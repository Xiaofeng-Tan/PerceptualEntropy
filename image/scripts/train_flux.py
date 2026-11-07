from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import swanlab
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def stream_t5_blocks_from_host(text_encoder, device):
    """Keep T5 blocks in pinned host memory and copy each to `device` only for its own forward.

    T5-XXL is 9.5 GB in bf16; next to the 23.8 GB FLUX transformer it does not fit a 40 GB card.
    The embedding and final norm stay resident so `text_encoder.device` is still the GPU.
    """
    text_encoder.shared.to(device)
    text_encoder.encoder.final_layer_norm.to(device)
    for block in text_encoder.encoder.block:
        params = list(block.parameters())
        host = [p.data.pin_memory() for p in params]
        for p, h in zip(params, host):
            p.data = h

        def to_device(module, args, params=params, host=host):
            for p, h in zip(params, host):
                p.data = h.to(device, non_blocking=True)

        def to_host(module, args, output, params=params, host=host):
            for p, h in zip(params, host):
                p.data = h

        block.register_forward_pre_hook(to_device)
        block.register_forward_hook(to_host)

def gather_to_host(accelerator, tensor, chunk_size):
    """Same result as accelerator.gather(tensor).cpu(), without the full gathered tensor on the GPU."""
    chunks = [accelerator.gather(c.contiguous()).cpu() for c in tensor.split(chunk_size)]
    # each gathered chunk is rank-major; restore the rank-major order of the whole tensor
    chunks = [c.reshape(accelerator.num_processes, -1, *c.shape[1:]) for c in chunks]
    return torch.cat(chunks, dim=1).reshape(-1, *tensor.shape[1:])

def log_cuda_memory(accelerator, tag):
    if accelerator.is_local_main_process:
        gib = 1024 ** 3
        print(
            f"[mem] rank{accelerator.process_index} {tag}: "
            f"peak_alloc={torch.cuda.max_memory_allocated() / gib:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved() / gib:.2f}GiB",
            flush=True,
        )
    torch.cuda.reset_peak_memory_stats()

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        reward_std_mean: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def calculate_group_reward_stats_per_timestep(prompts, gathered_rewards_avg):
    """
    Calculate per-group reward avg and std for each timestep.
    
    Args:
        prompts: List of prompts.
        gathered_rewards_avg: numpy array of shape (N, T) - rewards for each sample at each timestep.
        
    Returns:
        stats: Dictionary containing per-timestep group reward avg and std (mean across groups).
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Sort data by prompt groups
    sorted_indices = np.argsort(inverse_indices)
    sorted_rewards = gathered_rewards_avg[sorted_indices]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(sorted_rewards, split_indices)
    
    num_timesteps = gathered_rewards_avg.shape[1]
    stats = {}
    
    for t in range(num_timesteps):
        group_avgs = []
        group_stds = []
        for group in reward_groups:
            if len(group) > 0:
                group_t = group[:, t]
                group_avgs.append(np.mean(group_t))
                group_stds.append(np.std(group_t))
        
        # Mean across groups
        stats[f"group_reward_avg_t{t}"] = np.mean(group_avgs) if group_avgs else 0.0
        stats[f"group_reward_std_t{t}"] = np.mean(group_stds) if group_stds else 0.0
    
    return stats

def calculate_group_variance(prompts, data, num_processes):
    """
    Calculate the mean variance within each group (samples with the same prompt).
    
    Args:
        prompts: List of prompts.
        data: numpy array of shape (N,) or (N, D) - can be rewards, embeddings, latents, or log_probs.
        num_processes: Number of processes for gathering.
        
    Returns:
        mean_variance: Mean variance across all groups.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Sort data by prompt groups
    sorted_indices = np.argsort(inverse_indices)
    sorted_data = data[sorted_indices]
    split_indices = np.cumsum(counts)[:-1]
    data_groups = np.split(sorted_data, split_indices)
    
    # Calculate variance for each group
    if len(data.shape) == 1:
        # For 1D data (rewards, log_probs)
        group_variances = np.array([np.var(group) for group in data_groups if len(group) > 1])
    else:
        # For 2D data (embeddings, latents) - compute mean variance across dimensions
        group_variances = np.array([np.mean(np.var(group, axis=0)) for group in data_groups if len(group) > 1])
    
    return group_variances.mean() if len(group_variances) > 0 else 0.0

def calculate_group_log_prob_stats(prompts, log_probs, initial_latents=None):
    """
    Calculate cumulative log_prob statistics for each group (skip per-timestep stats).
    
    Args:
        prompts: List of prompts.
        log_probs: numpy array of shape (N, T) - log_prob for each sample at each timestep.
        initial_latents: numpy array of shape (N, D) - initial latents for prior probability (optional).
        
    Returns:
        stats: Dictionary containing cumulative statistics (mean only).
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Sort data by prompt groups
    sorted_indices = np.argsort(inverse_indices)
    sorted_log_probs = log_probs[sorted_indices]
    split_indices = np.cumsum(counts)[:-1]
    log_prob_groups = np.split(sorted_log_probs, split_indices)
    
    num_timesteps = log_probs.shape[1]
    stats = {}
    
    # Skip per-timestep statistics (too verbose)
    # for t in range(num_timesteps):
    #     group_means = []
    #     for group in log_prob_groups:
    #         if len(group) > 0:
    #             group_t = group[:, t]
    #             group_means.append(np.mean(group_t))
    #     stats[f"log_p_t{t}_group_mean"] = np.mean(group_means) if group_means else 0.0
    
    # 2. Cumulative log_prob (sum across timesteps, excluding prior)
    total_log_probs = log_probs.sum(axis=1)  # (N,)
    sorted_total = total_log_probs[sorted_indices]
    total_groups = np.split(sorted_total, split_indices)
    
    group_means = []
    for group in total_groups:
        if len(group) > 0:
            group_means.append(np.mean(group))
    
    stats["total_log_p_group_mean"] = np.mean(group_means) if group_means else 0.0
    
    # 3. Cumulative log_prob with prior (if initial_latents provided)
    if initial_latents is not None:
        # Prior: log p(z_0) = -0.5 * ||z_0||^2 - D/2 * log(2*pi)
        latents_flat = initial_latents.reshape(initial_latents.shape[0], -1)
        D = latents_flat.shape[1]
        log_prior = -0.5 * np.sum(latents_flat ** 2, axis=1) - 0.5 * D * np.log(2 * np.pi)
        
        total_with_prior = total_log_probs + log_prior
        sorted_total_prior = total_with_prior[sorted_indices]
        total_prior_groups = np.split(sorted_total_prior, split_indices)
        
        group_means = []
        for group in total_prior_groups:
            if len(group) > 0:
                group_means.append(np.mean(group))
        
        stats["total_log_p_with_prior_group_mean"] = np.mean(group_means) if group_means else 0.0
        
        # Also log the prior separately
        sorted_prior = log_prior[sorted_indices]
        prior_groups = np.split(sorted_prior, split_indices)
        group_means = []
        for group in prior_groups:
            if len(group) > 0:
                group_means.append(np.mean(group))
        stats["log_prior_group_mean"] = np.mean(group_means) if group_means else 0.0
    
    return stats

def calculate_reward_emb_log_prob(prompts, embeddings, return_per_sample=False):
    """
    Calculate log probability of each sample in the reward model embedding space.
    
    For each group (same prompt), we estimate the distribution as a Gaussian with:
    - μ = mean of embeddings in the group
    - σ² = variance of embeddings in the group (per dimension, then averaged)
    
    Then compute log p(x) = -0.5 * ||x - μ||² / σ² - D/2 * log(2πσ²)
    
    Args:
        prompts: List of prompts.
        embeddings: numpy array of shape (N, D) - reward model embeddings for each sample.
        return_per_sample: If True, also return per-sample log_prob in original order.
        
    Returns:
        stats: Dictionary containing log probability statistics in reward embedding space.
        per_sample_log_prob: (optional) numpy array of shape (N,) - log_prob for each sample in original order.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Sort data by prompt groups
    sorted_indices = np.argsort(inverse_indices)
    sorted_embs = embeddings[sorted_indices]
    split_indices = np.cumsum(counts)[:-1]
    emb_groups = np.split(sorted_embs, split_indices)
    
    # Also split sorted_indices to track original positions
    sorted_indices_groups = np.split(sorted_indices, split_indices)
    
    D = embeddings.shape[1]  # embedding dimension
    N = embeddings.shape[0]  # number of samples
    stats = {}
    
    all_log_probs = []
    group_log_prob_means = []
    
    # Per-sample log_prob in original order
    per_sample_log_prob = np.zeros(N)
    
    for group, orig_indices in zip(emb_groups, sorted_indices_groups):
        if len(group) < 2:
            # Need at least 2 samples to estimate variance
            # Set log_prob to 0 for these samples
            per_sample_log_prob[orig_indices] = 0.0
            continue
        
        # Estimate μ and σ² for this group
        mu = np.mean(group, axis=0)  # (D,)
        # Use sample variance with small epsilon for numerical stability
        var = np.var(group, axis=0) + 1e-8  # (D,)
        
        # Compute log probability for each sample in the group
        residual_sq = (group - mu) ** 2  # (K, D)
        normalized_residual_sq = residual_sq / var  # (K, D)
        
        # Per-sample log prob (averaged over dimensions for interpretability)
        log_prob_per_dim = -0.5 * normalized_residual_sq - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(var)  # (K, D)
        log_prob_per_sample = np.mean(log_prob_per_dim, axis=1)  # (K,) - average over dimensions
        
        # Store in original order
        per_sample_log_prob[orig_indices] = log_prob_per_sample
        
        all_log_probs.extend(log_prob_per_sample.tolist())
        group_log_prob_means.append(np.mean(log_prob_per_sample))
    
    if len(group_log_prob_means) > 0:
        stats["reward_emb_log_p_group_mean"] = np.mean(group_log_prob_means)
        stats["reward_emb_log_p_overall_mean"] = np.mean(all_log_probs) if all_log_probs else 0.0
        # Entropy in reward embedding space: H = -E[log p]
        stats["reward_emb_entropy_mean"] = -stats["reward_emb_log_p_group_mean"]
    else:
        stats["reward_emb_log_p_group_mean"] = 0.0
        stats["reward_emb_log_p_overall_mean"] = 0.0
        stats["reward_emb_entropy_mean"] = 0.0
    
    if return_per_sample:
        return stats, per_sample_log_prob
    return stats

def compute_intermediate_x0_embeddings(pipeline, latents_list, noise_preds_list, timesteps, reward_fn, height, width):
    """
    Compute x_0 predictions at each timestep using single-step Euler denoising,
    then decode to images and get reward model embeddings.
    
    For flow matching, x_0 prediction from x_t is:
        x_0 = x_t - sigma_t * v_t
    where v_t is the velocity prediction (noise_pred).
    
    Args:
        pipeline: The diffusion pipeline with VAE decoder.
        latents_list: List of latents at each timestep [x_T, x_{T-1}, ..., x_0], length = num_steps + 1
        noise_preds_list: List of noise predictions at each timestep, length = num_steps
        timesteps: Tensor of timesteps.
        reward_fn: Reward function that returns image embeddings.
        height, width: Image dimensions.
        
    Returns:
        all_x0_embs: List of embeddings for x_0 predictions at each timestep, each shape (batch, D)
    """
    all_x0_embs = []
    sigmas = pipeline.scheduler.sigmas
    
    with torch.no_grad():
        for i, (x_t, noise_pred) in enumerate(zip(latents_list[:-1], noise_preds_list)):
            # Get sigma for this timestep
            step_index = pipeline.scheduler.index_for_timestep(timesteps[i])
            sigma_t = sigmas[step_index].to(x_t.device)
            
            # Single-step Euler to predict x_0: x_0 = x_t - sigma_t * v_t
            # For flow matching: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
            # v_t predicts the direction from x_t to x_0
            # So: x_0 = x_t + (1 - sigma_t) * v_t / sigma_t  (if we interpret v as velocity)
            # But in this scheduler, noise_pred is the velocity v = (x_0 - x_t) / (1 - sigma_t) approximately
            # Actually for flow matching: x_0_pred = x_t - sigma_t * v_t where v_t = noise_pred
            
            # For FlowMatchEulerDiscreteScheduler, the update is:
            # x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_t
            # To get x_0 directly (sigma_0 = 0):
            # x_0 = x_t - sigma_t * v_t
            x_0_pred = x_t.float() - sigma_t * noise_pred.float()
            
            # Decode x_0 to image
            x_0_unpacked = pipeline._unpack_latents(x_0_pred, height, width, pipeline.vae_scale_factor)
            x_0_unpacked = (x_0_unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            x_0_unpacked = x_0_unpacked.to(dtype=pipeline.vae.dtype)
            
            images = pipeline.vae.decode(x_0_unpacked, return_dict=False)[0]
            images = pipeline.image_processor.postprocess(images, output_type="pt")
            
            # Get reward model embeddings (only need embeddings, not rewards)
            # We need to call reward_fn in a way that returns embeddings
            # For now, we'll store the images and compute embeddings later
            all_x0_embs.append(images)  # Store decoded images for now
    
    return all_x0_embs

def calculate_intermediate_reward_emb_log_prob(prompts, embeddings_per_step):
    """
    Calculate log probability and variance in reward embedding space for intermediate x_0 predictions.
    
    Args:
        prompts: List of prompts.
        embeddings_per_step: List of embeddings arrays, one per timestep.
                            Each array has shape (N, D).
        
    Returns:
        stats: Dictionary containing per-timestep log probability and variance statistics.
    """
    stats = {}
    
    for t, embeddings in enumerate(embeddings_per_step):
        if embeddings is None or len(embeddings) == 0:
            continue
            
        # Use the same logic as calculate_reward_emb_log_prob
        prompt_array = np.array(prompts)
        unique_prompts, inverse_indices, counts = np.unique(
            prompt_array, 
            return_inverse=True,
            return_counts=True
        )
        
        sorted_indices = np.argsort(inverse_indices)
        sorted_embs = embeddings[sorted_indices]
        split_indices = np.cumsum(counts)[:-1]
        emb_groups = np.split(sorted_embs, split_indices)
        
        group_log_prob_means = []
        group_variances = []  # Store group variances
        
        for group in emb_groups:
            if len(group) < 2:
                continue
            
            mu = np.mean(group, axis=0)
            var = np.var(group, axis=0) + 1e-8
            
            # Store mean variance across dimensions for this group
            group_variances.append(np.mean(var))
            
            residual_sq = (group - mu) ** 2
            normalized_residual_sq = residual_sq / var
            
            log_prob_per_dim = -0.5 * normalized_residual_sq - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(var)
            log_prob_per_sample = np.mean(log_prob_per_dim, axis=1)
            
            group_log_prob_means.append(np.mean(log_prob_per_sample))
        
        if len(group_log_prob_means) > 0:
            stats[f"reward_emb_log_p_t{t}_group_mean"] = np.mean(group_log_prob_means)
            stats[f"reward_emb_var_t{t}_group_mean"] = np.mean(group_variances)
        else:
            stats[f"reward_emb_log_p_t{t}_group_mean"] = 0.0
            stats[f"reward_emb_var_t{t}_group_mean"] = 0.0
    
    return stats

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, config):
    packed_noisy_model_input = sample["latents"][:, j]
    device = packed_noisy_model_input.device
    dtype = packed_noisy_model_input.dtype
    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(packed_noisy_model_input.shape[0])
    else:
        guidance = None

    # Predict the noise residual
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids= torch.zeros(sample["prompt_embeds"].shape[1], 3).to(device=device, dtype=dtype),
        img_ids=sample["image_ids"][0],
        return_dict=False,
    )[0]
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            swanlab.log(
                {
                    "eval_images": [
                        swanlab.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    resume_run_name = os.environ.get("FLOWGRPO_RUN_NAME", "").strip()
    if resume_run_name:
        config.run_name = resume_run_name
    elif not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        swanlab_id = os.environ.get("FLOWGRPO_SWANLAB_ID", "").strip()
        swanlab_kwargs = dict(
            project=config.swanlab_project,
            name=config.run_name,
            config=config.to_dict(),
        )
        if swanlab_id:
            swanlab_kwargs["id"] = swanlab_id
            swanlab_kwargs["resume"] = "must"
            logger.info(f"[resume] SwanLab continuing run {swanlab_id}")
        swanlab.init(**swanlab_kwargs)
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"swanlab": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.mem40g:
        pipeline.text_encoder_2.to(dtype=inference_dtype)
        stream_t5_blocks_from_host(pipeline.text_encoder_2, accelerator.device)
        pipeline.vae.enable_slicing()
    else:
        pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)

    # LoRA training disables autocast, so the frozen transformer must already be bf16.
    # Leaving it in fp32 filled a 95 GiB H20 before the first optimizer step.
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        resume_lora = os.environ.get("FLOWGRPO_RESUME_LORA", "").strip()
        lora_path = resume_lora or config.train.lora_path
        if lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
            if accelerator.is_main_process and resume_lora:
                logger.info(f"[resume] loaded LoRA from {resume_lora}")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
        if hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
            pipeline.transformer.enable_gradient_checkpointing()
        elif hasattr(pipeline.transformer, "gradient_checkpointing_enable"):
            pipeline.transformer.gradient_checkpointing_enable()
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.sample.global_std,
            buffer_size=int(getattr(config.train, "ddpo_buffer_size", 64)),
            min_count=int(getattr(config.train, "ddpo_min_count", 16)),
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    # prepare prompt and reward fn
    if is_deepspeed_zero3_enabled():
        # Using deepspeed zero3 will cause the model parameter `weight.shape` to be empty.
        unset_hf_deepspeed_config()
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        set_hf_deepspeed_config(accelerator.state.deepspeed_plugin.dschf)
    else:
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        if config.mem40g:
            # eval_reward_fn is only called synchronously, after all reward futures have resolved
            eval_reward_fn = reward_fn
        else:
            eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    # Create save directory at the start
    if accelerator.is_main_process and config.save_dir:
        os.makedirs(config.save_dir, exist_ok=True)

    epoch = int(os.environ.get("FLOWGRPO_RESUME_EPOCH", "0") or 0)
    global_step = int(os.environ.get("FLOWGRPO_RESUME_STEP", "0") or 0)
    if accelerator.is_main_process and (epoch or global_step):
        logger.info(f"[resume] continuing at epoch={epoch} global_step={global_step}")
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        all_image_embs = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            if config.sample.mix:
                if i % 40 <= 30:
                    config.sample.same_latent = True
                else:
                    config.sample.same_latent = False
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, image_ids, text_ids, log_probs, noise_preds = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            noise_preds = torch.stack(noise_preds, dim=1)  # (batch_size, num_steps, ...)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "image_ids": image_ids.unsqueeze(0).repeat(len(prompt_ids),1,1),
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "noise_preds": noise_preds,  # noise predictions for x_0 computation
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "images": images,  # decoded images for variance calculation
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            # Collect image embeddings if available
            if "image_embs" in reward_metadata:
                all_image_embs.append(reward_metadata["image_embs"].to(accelerator.device))

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                swanlab.log(
                    {
                        "images": [
                            swanlab.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Compute intermediate x_0 rewards for each timestep
        # For each timestep j, predict x_0 from x_j using single-step Euler, decode to image, and compute reward
        if config.train.use_intermediate_rewards:
            intermediate_rewards = []
            sigmas = pipeline.scheduler.sigmas.cpu()
            total_samples = samples["latents"].shape[0]
            
            # Number of early steps to skip (use final reward instead)
            skip_early_steps = getattr(config.train, 'intermediate_skip_early_steps', 0)
            
            # Decode prompts once outside the loop
            batch_prompt_ids = samples["prompt_ids"].cpu().numpy()
            batch_prompts = pipeline.tokenizer.batch_decode(batch_prompt_ids, skip_special_tokens=True)
            
            # Move large tensors to CPU to free GPU memory for reward computation
            latents_cpu = samples["latents"].cpu()
            noise_preds_cpu = samples["noise_preds"].cpu()
            timesteps_cpu = samples["timesteps"].cpu()
            images_cpu = samples["images"].cpu()
            final_reward = samples["rewards"]["avg"].clone()  # Save final reward for early steps
            
            # Delete GPU references and clear cache
            del samples["latents"], samples["noise_preds"], samples["images"]
            torch.cuda.empty_cache()
            
            # Process in mini-batches to avoid OOM
            mini_batch_size = config.sample.train_batch_size  # Use same size as sampling batch
            
            for j in tqdm(
                range(num_train_timesteps),
                desc="Computing intermediate rewards",
                disable=not accelerator.is_local_main_process,
            ):
                # Skip early steps: use final reward instead of intermediate x_0 reward
                if j < skip_early_steps:
                    intermediate_rewards.append(final_reward.to(accelerator.device))
                    continue
                
                rewards_j_all = []
                
                # Get sigma for this timestep (same for all samples)
                timestep_j = timesteps_cpu[0, j].item()
                step_index = pipeline.scheduler.index_for_timestep(torch.tensor(timestep_j))
                sigma_t = sigmas[step_index]
                
                # Process in mini-batches
                for start_idx in range(0, total_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, total_samples)
                    
                    # Get x_t and noise_pred for this mini-batch (load from CPU to GPU)
                    x_t = latents_cpu[start_idx:end_idx, j].to(accelerator.device)
                    noise_pred_j = noise_preds_cpu[start_idx:end_idx, j].to(accelerator.device)
                    
                    # Single-step Euler to predict x_0: x_0 = x_t - sigma_t * v_t
                    x_0_pred = x_t.float() - sigma_t.to(accelerator.device) * noise_pred_j.float()
                    
                    # Decode x_0 to image
                    with torch.no_grad():
                        x_0_unpacked = pipeline._unpack_latents(
                            x_0_pred, config.resolution, config.resolution, pipeline.vae_scale_factor
                        )
                        x_0_unpacked = (x_0_unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                        x_0_unpacked = x_0_unpacked.to(dtype=pipeline.vae.dtype)
                        
                        images_j = pipeline.vae.decode(x_0_unpacked, return_dict=False)[0]
                        images_j = pipeline.image_processor.postprocess(images_j, output_type="pt")
                    
                    # Compute reward for this mini-batch
                    mini_batch_prompts = batch_prompts[start_idx:end_idx]
                    rewards_j_mini, _ = reward_fn(images_j, mini_batch_prompts, None, only_strict=True)
                    rewards_j_all.append(torch.as_tensor(rewards_j_mini['avg'], device=accelerator.device).float())
                    
                    # Free memory
                    del x_t, noise_pred_j, x_0_pred, x_0_unpacked, images_j
                    torch.cuda.empty_cache()
                
                # Concatenate all mini-batch rewards for this timestep
                intermediate_rewards.append(torch.cat(rewards_j_all, dim=0))
            
            # Stack intermediate rewards: (batch, num_train_timesteps)
            samples["rewards"]["avg"] = torch.stack(intermediate_rewards, dim=1)
            
            # Move latents and noise_preds back to GPU for training
            samples["latents"] = latents_cpu.to(accelerator.device)
            samples["noise_preds"] = noise_preds_cpu.to(accelerator.device)
            samples["images"] = images_cpu.to(accelerator.device)
            del latents_cpu, noise_preds_cpu, images_cpu
        else:
            # Original behavior: repeat final reward to all timesteps
            samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            swanlab.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            
            # Gather log_probs for covariance computation
            gathered_log_probs_for_cov = accelerator.gather(samples["log_probs"]).cpu().float().numpy()
            # Use mean log_prob across timesteps for covariance computation
            mean_log_probs_for_cov = gathered_log_probs_for_cov.mean(axis=1)
            
            # Calculate entropy bonus (if enabled)
            entropy_weight = getattr(config.train, 'entropy_weight', 0.0)
            use_latent_entropy = getattr(config.train, 'use_latent_entropy', False)
            entropy_target = getattr(config.train, 'entropy_target', 'reward')
            use_dual_entropy = getattr(config.train, 'use_dual_entropy', False)
            per_sample_entropy = None
            
            # Get advanced sample selection settings
            use_extreme_samples_only = getattr(config.train, 'use_extreme_samples_only', False)
            extreme_samples_ratio = getattr(config.train, 'extreme_samples_ratio', 0.25)
            extreme_samples_mode = getattr(config.train, 'extreme_samples_mode', 'drop_high')  # 'drop_high' or 'keep_extreme'
            use_entropy_kl_constraint = getattr(config.train, 'use_entropy_kl_constraint', False)
            entropy_kl_ratio = getattr(config.train, 'entropy_kl_ratio', 0.25)
            
            # DDPO_IS keeps the clipped likelihood-ratio loss below, but the baseline
            # is a per-prompt running mean/std (Black et al., 2023, Appendix D.3),
            # not the current-group baseline used by GRPO.
            algorithm = getattr(config, "algorithm", "grpo")
            if algorithm == "ddpo" and not use_dual_entropy:
                advantages = stat_tracker.update_running(prompts, gathered_rewards["avg"])
                if accelerator.is_local_main_process:
                    print(
                        f"[DDPO] running per-prompt advantage, "
                        f"prompts_in_buffer={len(stat_tracker.history)}, "
                        f"buffer_size={stat_tracker.buffer_size}, "
                        f"adv_mean={advantages.mean():.4f}, adv_std={advantages.std():.4f}"
                    )
            # Dual entropy mode: r' = r + w1 * H(reward_emb) - w2 * H(latent)
            # DDPO uses the same augmented reward, then its running baseline.
            elif use_dual_entropy:
                reward_emb_entropy_weight = getattr(config.train, 'reward_emb_entropy_weight', 0.1)
                latent_entropy_weight = getattr(config.train, 'latent_entropy_weight', 0.1)
                dual_entropy_target = getattr(config.train, 'dual_entropy_target', 'advantage')
                use_importance_weighting = getattr(config.train, 'use_importance_weighting', False)
                
                # 1. Compute H(reward_emb) - reward embedding space entropy
                H_reward_emb = None
                if len(all_image_embs) > 0:
                    all_embs = torch.cat(all_image_embs, dim=0)
                    gathered_embs = accelerator.gather(all_embs).cpu().float().numpy()
                    _, reward_emb_log_prob = calculate_reward_emb_log_prob(prompts, gathered_embs, return_per_sample=True)
                    H_reward_emb = -reward_emb_log_prob  # (N,)
                
                # 2. Compute H(latent) - latent space entropy
                gathered_latents = accelerator.gather(samples["next_latents"][:, -1]).cpu().float().numpy()
                gathered_latents_flat = gathered_latents.reshape(gathered_latents.shape[0], -1)
                _, latent_log_prob = calculate_reward_emb_log_prob(prompts, gathered_latents_flat, return_per_sample=True)
                H_latent = -latent_log_prob  # (N,)
                
                # 3. Compute entropy bonus
                if use_importance_weighting:
                    # New formula: w * (p / π_old) * H
                    log_pi_old = gathered_log_probs_for_cov.sum(axis=1)  # log π_old(x), (N,)
                    
                    # Latent importance weight: p_latent / π_old
                    log_importance_latent = latent_log_prob - log_pi_old
                    importance_weight_latent = np.exp(log_importance_latent)
                    importance_weight_latent = np.clip(importance_weight_latent, 1e-6, 1e6)
                    
                    if H_reward_emb is not None:
                        # Reward emb importance weight: p_emb / π_old
                        log_importance_emb = reward_emb_log_prob - log_pi_old
                        importance_weight_emb = np.exp(log_importance_emb)
                        importance_weight_emb = np.clip(importance_weight_emb, 1e-6, 1e6)
                        
                        entropy_bonus = reward_emb_entropy_weight * importance_weight_emb * H_reward_emb \
                                      - latent_entropy_weight * importance_weight_latent * H_latent
                    else:
                        entropy_bonus = -latent_entropy_weight * importance_weight_latent * H_latent
                else:
                    # Original formula: w * H (no importance weighting)
                    if H_reward_emb is not None:
                        entropy_bonus = reward_emb_entropy_weight * H_reward_emb \
                                      - latent_entropy_weight * H_latent
                    else:
                        entropy_bonus = -latent_entropy_weight * H_latent
                
                # 4. Apply entropy bonus to reward or advantage based on dual_entropy_target
                if dual_entropy_target == "reward":
                    # Apply to reward first, then compute advantage
                    augmented_rewards = gathered_rewards['avg'].copy()
                    if len(augmented_rewards.shape) == 1:
                        augmented_rewards = augmented_rewards + entropy_bonus
                    else:
                        augmented_rewards = augmented_rewards + entropy_bonus[:, None]
                    if algorithm == "ddpo":
                        advantages = stat_tracker.update_running(prompts, augmented_rewards)
                    else:
                        advantages = stat_tracker.update(prompts, augmented_rewards)
                    
                    if accelerator.is_local_main_process:
                        log_parts = [
                            f"[Dual Entropy on Reward] H_reward_emb={H_reward_emb.mean() if H_reward_emb is not None else 'N/A':.4f}",
                            f"H_latent={H_latent.mean():.4f}",
                            f"bonus_mean={entropy_bonus.mean():.4f}",
                        ]
                        if use_importance_weighting:
                            iw_emb_str = f"iw_emb=[mean={importance_weight_emb.mean():.4f}, min={importance_weight_emb.min():.4f}, max={importance_weight_emb.max():.4f}]" if H_reward_emb is not None else "iw_emb=N/A"
                            iw_lat_str = f"iw_latent=[mean={importance_weight_latent.mean():.4f}, min={importance_weight_latent.min():.4f}, max={importance_weight_latent.max():.4f}]"
                            log_parts.extend([iw_emb_str, iw_lat_str])
                        log_parts.extend([
                            f"original_reward_mean={gathered_rewards['avg'].mean():.4f}",
                            f"augmented_reward_mean={augmented_rewards.mean():.4f}",
                        ])
                        print(", ".join(log_parts))
                else:
                    # Original behavior: apply to advantage
                    advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
                    
                    if len(advantages.shape) == 1:
                        advantages = advantages + entropy_bonus
                    else:
                        advantages = advantages + entropy_bonus[:, None]
                    
                    if accelerator.is_local_main_process:
                        print(f"[Dual Entropy on Advantage] H_reward_emb={H_reward_emb.mean() if H_reward_emb is not None else 'N/A':.4f}, "
                              f"H_latent={H_latent.mean():.4f}, bonus_mean={entropy_bonus.mean():.4f}")
            
            elif entropy_weight > 0:
                if use_latent_entropy:
                    # Use latent space entropy instead of reward embedding entropy
                    # Gather final latents
                    gathered_latents = accelerator.gather(samples["next_latents"][:, -1]).cpu().float().numpy()
                    # Flatten latents for entropy calculation
                    gathered_latents_flat = gathered_latents.reshape(gathered_latents.shape[0], -1)
                    
                    # Get per-sample log_prob in latent space and compute entropy = -log_prob
                    _, per_sample_log_prob = calculate_reward_emb_log_prob(prompts, gathered_latents_flat, return_per_sample=True)
                    per_sample_entropy = -per_sample_log_prob  # (N,)
                    
                    if accelerator.is_local_main_process:
                        print(f"[Latent Entropy Bonus] weight={entropy_weight}, entropy_mean={per_sample_entropy.mean():.4f}, entropy_std={per_sample_entropy.std():.4f}")
                elif len(all_image_embs) > 0:
                    # Use reward embedding space entropy (original behavior)
                    all_embs = torch.cat(all_image_embs, dim=0)
                    gathered_embs = accelerator.gather(all_embs).cpu().float().numpy()
                    
                    # Get per-sample log_prob and compute entropy = -log_prob
                    _, per_sample_log_prob = calculate_reward_emb_log_prob(prompts, gathered_embs, return_per_sample=True)
                    per_sample_entropy = -per_sample_log_prob  # (N,)
                    
                    if accelerator.is_local_main_process:
                        print(f"[Reward Emb Entropy Bonus] weight={entropy_weight}, entropy_mean={per_sample_entropy.mean():.4f}, entropy_std={per_sample_entropy.std():.4f}")
                
                if per_sample_entropy is not None:
                    if entropy_target == "reward":
                        # Original behavior: entropy added to reward before computing advantage
                        # augmented_reward = reward + entropy_weight * entropy
                        augmented_rewards = gathered_rewards['avg'].copy()
                        if len(augmented_rewards.shape) == 1:
                            augmented_rewards = augmented_rewards + entropy_weight * per_sample_entropy
                        else:
                            augmented_rewards = augmented_rewards + entropy_weight * per_sample_entropy[:, None]
                        
                        if accelerator.is_local_main_process:
                            print(f"[Entropy on Reward] original_reward_mean={gathered_rewards['avg'].mean():.4f}, augmented_reward_mean={augmented_rewards.mean():.4f}")
                        
                        advantages = stat_tracker.update(prompts, augmented_rewards)
                    else:
                        # New behavior: entropy added to advantage after computing from original reward
                        # A = normalize(r) + entropy_weight * entropy
                        advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
                        
                        if len(advantages.shape) == 1:
                            advantages = advantages + entropy_weight * per_sample_entropy
                        else:
                            advantages = advantages + entropy_weight * per_sample_entropy[:, None]
                        
                        if accelerator.is_local_main_process:
                            print(f"[Entropy on Advantage] entropy_weight={entropy_weight}, entropy_mean={per_sample_entropy.mean():.4f}")
                else:
                    advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            else:
                advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            
            # Compute covariances for extreme sample selection (Setting 1)
            covariances = None
            extreme_sample_mask = None
            
            # Determine which log_probs to use for covariance computation
            covariance_source = getattr(config.train, 'covariance_source', 'latent')
            if covariance_source == 'reward_emb' and len(all_image_embs) > 0:
                # Use reward embedding log prob for covariance computation
                all_embs = torch.cat(all_image_embs, dim=0)
                gathered_embs_for_cov = accelerator.gather(all_embs).cpu().float().numpy()
                _, log_probs_for_cov = calculate_reward_emb_log_prob(prompts, gathered_embs_for_cov, return_per_sample=True)
                if accelerator.is_local_main_process:
                    print(f"[Covariance] Using reward embedding log prob (source={covariance_source})")
            else:
                # Use VAE latent log prob (default behavior)
                log_probs_for_cov = mean_log_probs_for_cov
                if accelerator.is_local_main_process and (use_extreme_samples_only or use_entropy_kl_constraint):
                    print(f"[Covariance] Using VAE latent log prob (source={covariance_source})")
            
            if use_extreme_samples_only or use_entropy_kl_constraint:
                # Clear stats before recomputing with log_probs to avoid extend error
                # (stats were already converted to np.ndarray in first update call)
                stat_tracker.clear()
            
            if use_extreme_samples_only:
                # Compute covariance: Cov(yi) = (log πθ(yi) - mean(log πθ)) · (A(yi) - mean(A))
                # Re-compute with log_probs to get covariances
                _, covariances = stat_tracker.update(prompts, gathered_rewards['avg'], log_probs=log_probs_for_cov)
                extreme_sample_mask = stat_tracker.select_extreme_samples(
                    prompts, covariances, ratio=extreme_samples_ratio, mode=extreme_samples_mode
                )
                if accelerator.is_local_main_process:
                    num_selected = extreme_sample_mask.sum()
                    print(f"[Extreme Samples] Selected {num_selected}/{len(extreme_sample_mask)} samples (ratio={extreme_samples_ratio}, mode={extreme_samples_mode})")
            
            # Get high covariance samples mask for KL constraint (Setting 2)
            high_cov_mask = None
            if use_entropy_kl_constraint:
                # Need covariances for high covariance sample selection
                if covariances is None:
                    # Compute covariances if not already computed
                    _, covariances = stat_tracker.update(prompts, gathered_rewards['avg'], log_probs=log_probs_for_cov)
                high_cov_mask = stat_tracker.get_high_covariance_samples(
                    prompts, covariances, ratio=entropy_kl_ratio
                )
                if accelerator.is_local_main_process:
                    num_high_cov = high_cov_mask.sum()
                    print(f"[High Covariance KL] {num_high_cov}/{len(high_cov_mask)} samples will have KL constraint (ratio={entropy_kl_ratio})")
            
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
            
            # Calculate group variances for different metrics
            variance_logs = {}
            
            # Skip per-timestep group reward logging (too verbose)
            # if config.train.use_intermediate_rewards and 'avg' in gathered_rewards and len(gathered_rewards['avg'].shape) == 2:
            #     group_reward_stats = calculate_group_reward_stats_per_timestep(prompts, gathered_rewards['avg'])
            #     variance_logs.update(group_reward_stats)
            
            # 1. Reward model embedding variance (if available)
            if len(all_image_embs) > 0:
                # Reuse gathered_embs if already computed, otherwise compute now
                if entropy_weight <= 0 or per_sample_entropy is None or use_latent_entropy:
                    all_embs = torch.cat(all_image_embs, dim=0)
                    gathered_embs = accelerator.gather(all_embs).cpu().float().numpy()
                
                emb_group_var = calculate_group_variance(prompts, gathered_embs, accelerator.num_processes)
                variance_logs["group_emb_var"] = emb_group_var
                
                # 1.1 Calculate log probability in reward embedding space (for final clean images)
                reward_emb_log_prob_stats = calculate_reward_emb_log_prob(prompts, gathered_embs)
                variance_logs.update(reward_emb_log_prob_stats)
            
            # Log entropy bonus stats if enabled
            if entropy_weight > 0 and per_sample_entropy is not None:
                entropy_type = "latent" if use_latent_entropy else "reward_emb"
                variance_logs[f"entropy_bonus_mean_{entropy_type}"] = per_sample_entropy.mean()
                variance_logs[f"entropy_bonus_std_{entropy_type}"] = per_sample_entropy.std()
                variance_logs[f"entropy_target"] = 0 if entropy_target == "reward" else 1  # 0=reward, 1=advantage
            
            # Log extreme sample selection stats
            if use_extreme_samples_only and extreme_sample_mask is not None:
                variance_logs["extreme_samples_selected_ratio"] = extreme_sample_mask.sum() / len(extreme_sample_mask)
                if covariances is not None:
                    variance_logs["covariance_mean"] = np.mean(covariances)
                    variance_logs["covariance_std"] = np.std(covariances)
            
            # Log high covariance KL constraint stats
            if use_entropy_kl_constraint and high_cov_mask is not None:
                variance_logs["high_cov_samples_ratio"] = high_cov_mask.sum() / len(high_cov_mask)
            
            # Log dual entropy stats if enabled
            if use_dual_entropy:
                variance_logs["dual_entropy_H_reward_emb_mean"] = H_reward_emb.mean() if H_reward_emb is not None else 0
                variance_logs["dual_entropy_H_latent_mean"] = H_latent.mean()
                variance_logs["dual_entropy_bonus_mean"] = entropy_bonus.mean()
            
            # 1.2 Calculate intermediate x_0 reward embedding log probability (if enabled)
            if config.train.compute_intermediate_reward_emb_log_p and len(all_image_embs) > 0:
                # Compute x_0 predictions at each timestep and get reward embeddings
                gathered_latents_all = accelerator.gather(samples["latents"]).cpu()  # (N, T, ...)
                gathered_noise_preds = accelerator.gather(samples["noise_preds"]).cpu()  # (N, T, ...)
                
                if accelerator.is_main_process:
                    num_samples = gathered_latents_all.shape[0]
                    num_steps = gathered_latents_all.shape[1]
                    
                    intermediate_embs_per_step = []
                    sigmas = pipeline.scheduler.sigmas.cpu()
                    
                    # Use mini-batch processing to avoid OOM
                    mini_batch_size = config.sample.train_batch_size
                    
                    for t in tqdm(
                        range(num_steps),
                        desc="Computing intermediate embs",
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Get sigma for this timestep
                        timestep_t = samples["timesteps"][0, t].item()
                        step_index = pipeline.scheduler.index_for_timestep(torch.tensor(timestep_t))
                        sigma_t = sigmas[step_index]
                        
                        all_embs_t = []
                        
                        # Process in mini-batches
                        for batch_start in range(0, num_samples, mini_batch_size):
                            batch_end = min(batch_start + mini_batch_size, num_samples)
                            
                            # Load from CPU to GPU
                            x_t = gathered_latents_all[batch_start:batch_end, t].to(accelerator.device)
                            noise_pred_t = gathered_noise_preds[batch_start:batch_end, t].to(accelerator.device)
                            
                            # Single-step Euler to predict x_0: x_0 = x_t - sigma_t * v_t
                            x_0_pred = x_t.float() - sigma_t.to(accelerator.device) * noise_pred_t.float()
                            
                            # Decode x_0 to image
                            with torch.no_grad():
                                x_0_unpacked = pipeline._unpack_latents(
                                    x_0_pred, config.resolution, config.resolution, pipeline.vae_scale_factor
                                )
                                x_0_unpacked = (x_0_unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                                x_0_unpacked = x_0_unpacked.to(dtype=pipeline.vae.dtype)
                                
                                images_batch = pipeline.vae.decode(x_0_unpacked, return_dict=False)[0]
                                images_batch = pipeline.image_processor.postprocess(images_batch, output_type="pt")
                            
                            # Get reward model embeddings for this mini-batch
                            mini_batch_prompts = prompts[batch_start:batch_end]
                            with torch.no_grad():
                                _, reward_meta_batch = reward_fn(images_batch, mini_batch_prompts, None, only_strict=True)
                                if "image_embs" in reward_meta_batch:
                                    all_embs_t.append(reward_meta_batch["image_embs"].cpu().float().numpy())
                            
                            # Free memory
                            del x_t, noise_pred_t, x_0_pred, x_0_unpacked, images_batch
                            torch.cuda.empty_cache()
                        
                        # Concatenate all mini-batch embeddings for this timestep
                        if len(all_embs_t) > 0:
                            intermediate_embs_per_step.append(np.concatenate(all_embs_t, axis=0))
                        else:
                            intermediate_embs_per_step.append(None)
                    
                    # Skip per-timestep intermediate reward embedding log probability logging (too verbose)
                    # intermediate_log_p_stats = calculate_intermediate_reward_emb_log_prob(
                    #     prompts, intermediate_embs_per_step
                    # )
                    # variance_logs.update(intermediate_log_p_stats)
            
            # 2. Latent variance (final latent)
            gathered_latents = accelerator.gather(samples["next_latents"][:, -1]).cpu().float().numpy()
            # Flatten latents for variance calculation
            gathered_latents_flat = gathered_latents.reshape(gathered_latents.shape[0], -1)
            latent_group_var = calculate_group_variance(prompts, gathered_latents_flat, accelerator.num_processes)
            variance_logs["group_latent_var"] = latent_group_var
            
            # 3. Decoded image variance
            if config.mem40g:
                gathered_images = gather_to_host(accelerator, samples["images"], config.sample.train_batch_size).float().numpy()
            else:
                gathered_images = accelerator.gather(samples["images"]).cpu().float().numpy()
            # Flatten images for variance calculation (batch, C, H, W) -> (batch, C*H*W)
            gathered_images_flat = gathered_images.reshape(gathered_images.shape[0], -1)
            image_group_var = calculate_group_variance(prompts, gathered_images_flat, accelerator.num_processes)
            variance_logs["group_image_var"] = image_group_var
            
            # 4. log_prob variance (p_old)
            gathered_log_probs = accelerator.gather(samples["log_probs"]).cpu().float().numpy()
            # Use mean log_prob across timesteps
            mean_log_probs = gathered_log_probs.mean(axis=1)
            log_prob_group_var = calculate_group_variance(prompts, mean_log_probs, accelerator.num_processes)
            variance_logs["group_log_p_old_var"] = log_prob_group_var
            
            # 5. Per-timestep and cumulative log_prob statistics
            # Get initial latents for prior calculation (first latent in the sequence)
            gathered_initial_latents = accelerator.gather(samples["latents"][:, 0]).cpu().float().numpy()
            log_prob_stats = calculate_group_log_prob_stats(
                prompts, 
                gathered_log_probs, 
                initial_latents=gathered_initial_latents
            )
            variance_logs.update(log_prob_stats)

            if accelerator.is_main_process:
                swanlab.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        **variance_logs,
                    },
                    step=global_step,
                )
            if getattr(config, "algorithm", "grpo") != "ddpo":
                stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        
        # Store extreme_sample_mask and high_entropy_mask for training
        if config.per_prompt_stat_tracking:
            if extreme_sample_mask is not None:
                extreme_sample_mask = torch.as_tensor(extreme_sample_mask)
                samples["extreme_sample_mask"] = (
                    extreme_sample_mask.reshape(accelerator.num_processes, -1)[accelerator.process_index]
                    .to(accelerator.device)
                )
            if high_cov_mask is not None:
                high_cov_mask = torch.as_tensor(high_cov_mask)
                samples["high_cov_mask"] = (
                    high_cov_mask.reshape(accelerator.num_processes, -1)[accelerator.process_index]
                    .to(accelerator.device)
                )
        
        # Store reward embedding log_prob for reward_emb entropy regularization
        use_reward_emb_entropy_reg = getattr(config.train, 'use_reward_emb_entropy_reg', False)
        if use_reward_emb_entropy_reg and len(all_image_embs) > 0:
            all_embs = torch.cat(all_image_embs, dim=0)
            gathered_embs = accelerator.gather(all_embs).cpu().float().numpy()
            _, reward_emb_log_prob = calculate_reward_emb_log_prob(prompts, gathered_embs, return_per_sample=True)
            reward_emb_log_prob = torch.as_tensor(reward_emb_log_prob, dtype=torch.float32)
            samples["reward_emb_log_prob"] = (
                reward_emb_log_prob.reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )
            if accelerator.is_local_main_process:
                print(f"[Reward Emb Entropy Reg] Stored reward_emb_log_prob, mean={reward_emb_log_prob.mean():.4f}, std={reward_emb_log_prob.std():.4f}")
        
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        del samples["images"]  # no longer needed after variance calculation
        del samples["noise_preds"]  # no longer needed after intermediate x_0 computation

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps

        log_cuda_memory(accelerator, f"epoch{epoch} sampling+reward")
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        print("ratio", ratio)
                        
                        # Setting 1: Apply extreme sample mask (only keep highest and lowest covariance samples)
                        use_extreme_samples_only = getattr(config.train, 'use_extreme_samples_only', False)
                        if use_extreme_samples_only and "extreme_sample_mask" in sample:
                            extreme_mask = sample["extreme_sample_mask"].bool()
                            # Apply mask to advantages, ratio, and log_prob
                            advantages_masked = advantages[extreme_mask]
                            ratio_masked = ratio[extreme_mask]
                            log_prob_masked = log_prob[extreme_mask]
                            log_probs_old_masked = sample["log_probs"][:, j][extreme_mask]
                        else:
                            advantages_masked = advantages
                            ratio_masked = ratio
                            log_prob_masked = log_prob
                            log_probs_old_masked = sample["log_probs"][:, j]
                        
                        if config.train.maximize_log_p_only:
                            # Only maximize log p (minimize negative log p)
                            # This ignores reward and just maximizes the probability of the sampled trajectory
                            loss = -torch.mean(log_prob_masked)
                            policy_loss = loss  # for logging
                            info["maximize_log_p_loss"].append(loss)
                        else:
                            unclipped_loss = -advantages_masked * ratio_masked
                            clipped_loss = -advantages_masked * torch.clamp(
                                ratio_masked,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            
                            # reinforce loss: -p_r / p_old * log p_theta
                            if config.train.reinforce_weight > 0:
                                p_r = torch.softmax(sample["advantages"][:, j], dim=0)
                                # p_old = exp(log_prob) then softmax to normalize
                                p_old_unnorm = torch.exp(sample["log_probs"][:, j])
                                p_old = p_old_unnorm / (p_old_unnorm.sum() + 1e-8)
                                # importance weight: p_r / p_old
                                importance_weight = p_r / (p_old + 1e-8)
                                # p_r / p_old * log p_theta, gradient w.r.t. theta only through log_prob
                                reinforce_loss = -torch.mean(importance_weight.detach() * log_prob)
                                # Log importance weight stats
                                info["importance_weight_mean"].append(torch.mean(importance_weight))
                                info["importance_weight_max"].append(torch.max(importance_weight))
                                info["importance_weight_min"].append(torch.min(importance_weight))
                                info["importance_weight_std"].append(torch.std(importance_weight))
                            
                            if config.train.beta > 0:
                                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                                kl_loss = torch.mean(kl_loss)
                                loss = policy_loss + config.train.beta * kl_loss
                            else:
                                loss = policy_loss
                            
                            # Setting 2: KL constraint for high covariance samples
                            use_entropy_kl_constraint = getattr(config.train, 'use_entropy_kl_constraint', False)
                            entropy_kl_weight = getattr(config.train, 'entropy_kl_weight', 0.001)
                            if use_entropy_kl_constraint and "high_cov_mask" in sample:
                                high_cov_mask = sample["high_cov_mask"].bool()
                                if high_cov_mask.any():
                                    # Compute KL loss only for high covariance samples
                                    if config.train.beta > 0:
                                        # Reuse prev_sample_mean_ref from above
                                        high_cov_kl = ((prev_sample_mean[high_cov_mask] - prev_sample_mean_ref[high_cov_mask]) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t[high_cov_mask] ** 2)
                                        high_cov_kl_loss = torch.mean(high_cov_kl)
                                    else:
                                        # Need to compute reference model output
                                        with torch.no_grad():
                                            with transformer.module.disable_adapter():
                                                _, _, prev_sample_mean_ref_hc, _ = compute_log_prob(transformer, pipeline, sample, j, config)
                                        high_cov_kl = ((prev_sample_mean[high_cov_mask] - prev_sample_mean_ref_hc[high_cov_mask]) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t[high_cov_mask] ** 2)
                                        high_cov_kl_loss = torch.mean(high_cov_kl)
                                    
                                    loss = loss + entropy_kl_weight * high_cov_kl_loss
                                    info["high_cov_kl_loss"].append(high_cov_kl_loss)
                            
                            # Entropy regularization: encourage higher entropy (more exploration)
                            # H = -E[log p], so we add entropy_reg_weight * H to maximize entropy
                            # This is equivalent to subtracting entropy_reg_weight * mean(log_prob)
                            entropy_reg_weight = getattr(config.train, 'entropy_reg_weight', 0.0)
                            if entropy_reg_weight > 0:
                                # entropy_loss = -H = mean(log_prob), we want to minimize this (maximize entropy)
                                entropy_loss = torch.mean(log_prob)
                                loss = loss + entropy_reg_weight * entropy_loss
                                info["entropy_reg_loss"].append(entropy_loss)
                            
                            # Reward embedding entropy regularization: encourage diversity in reward space
                            # H(reward_emb) = -log p(image_embedding | prompt)
                            # We want to maximize H(reward_emb), which means minimizing mean(reward_emb_log_prob)
                            use_reward_emb_entropy_reg = getattr(config.train, 'use_reward_emb_entropy_reg', False)
                            reward_emb_entropy_reg_weight = getattr(config.train, 'reward_emb_entropy_reg_weight', 0.0)
                            if use_reward_emb_entropy_reg and reward_emb_entropy_reg_weight > 0 and "reward_emb_log_prob" in sample:
                                # reward_emb_log_prob is per-sample, shape (batch_size,)
                                # We want to maximize entropy = -log_prob, so we minimize log_prob
                                reward_emb_entropy_loss = torch.mean(sample["reward_emb_log_prob"])
                                loss = loss + reward_emb_entropy_reg_weight * reward_emb_entropy_loss
                                info["reward_emb_entropy_reg_loss"].append(reward_emb_entropy_loss)
                                # Also track reward embedding entropy
                                reward_emb_entropy = -torch.mean(sample["reward_emb_log_prob"])
                                info["reward_emb_entropy"].append(reward_emb_entropy)
                            
                            if config.train.reinforce_weight > 0:
                                loss = loss + config.train.reinforce_weight * reinforce_loss
                                info["reinforce_loss"].append(reinforce_loss)

                        # info["approx_kl"].append(
                        #     0.5
                        #     * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        # )
                        # Entropy tracking: H = -E[log p]
                        H_old = -torch.mean(sample["log_probs"][:, j])  # entropy of old model's samples
                        H_new = -torch.mean(log_prob)  # entropy of new model's samples
                        info["H_old"].append(H_old)
                        info["H_new"].append(H_new)
                        info["entropy_diff"].append(H_new - H_old)  # positive = entropy increase, negative = entropy decrease
                        info["entropy"].append(H_new)
                        
                        # Print entropy info
                        if accelerator.is_local_main_process:
                            print(f"[t={j}] H_old: {H_old.item():.4f}, H_new: {H_new.item():.4f}, entropy_diff: {(H_new - H_old).item():.4f}, ratio_mean: {ratio.mean().item():.4f}")
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # Compute true H_new after gradient update (without gradient)
                        # Use importance sampling correction: H_new = -E_{p_old}[p_new/p_old * log p_new]
                        with torch.no_grad():
                            with autocast():
                                # Recompute log_prob with updated model
                                _, log_prob_after_update, _, _ = compute_log_prob(transformer, pipeline, sample, j, config)
                                log_p_old = sample["log_probs"][:, j]
                                log_p_new = log_prob_after_update
                                
                                # Importance weight: p_new / p_old = exp(log_p_new - log_p_old)
                                importance_weight = torch.exp(log_p_new - log_p_old)
                                
                                # Corrected entropy: -E_{p_old}[p_new/p_old * log p_new]
                                H_new_corrected = -torch.mean(importance_weight * log_p_new)
                                H_old_current = -torch.mean(log_p_old)
                                
                                info["H_new_corrected"].append(H_new_corrected)
                                info["entropy_diff_corrected"].append(H_new_corrected - H_old_current)
                                info["importance_weight_mean"].append(torch.mean(importance_weight))
                                info["importance_weight_std"].append(torch.std(importance_weight))
                        
                        # Sample K images from validation set with updated model and record log_prob/entropy
                        # Only do this every 20 global steps to save time
                        pipeline.transformer.eval()
                        val_info = {}
                        val_images_for_log = None
                        val_prompts_for_log = None
                        val_log_probs_for_log = None
                        if global_step % 20 == 0:
                            with torch.no_grad():
                                with autocast():
                                    # Get 1 prompt and generate K samples
                                    try:
                                        val_prompts_batch, val_metadata = next(iter(test_dataloader))
                                        val_prompt = val_prompts_batch[0]  # Take 1 prompt
                                    except StopIteration:
                                        val_prompt = prompts[0]
                                    
                                    # Repeat the same prompt K times
                                    K = config.sample.num_image_per_prompt  # Use same K as training
                                    val_prompts_repeated = [val_prompt] * K
                                    
                                    val_prompt_embeds, val_pooled_prompt_embeds = compute_text_embeddings(
                                        val_prompts_repeated, text_encoders, tokenizers, 
                                        max_sequence_length=128, device=accelerator.device
                                    )
                                    
                                    # Sample K images with updated model (SDE sampling for valid log_prob)
                                    val_images, val_latents, val_image_ids, _, val_log_probs, val_noise_preds = pipeline_with_logprob(
                                        pipeline,
                                        prompt_embeds=val_prompt_embeds,
                                        pooled_prompt_embeds=val_pooled_prompt_embeds,
                                        num_inference_steps=config.sample.num_steps,
                                        guidance_scale=config.sample.guidance_scale,
                                        output_type="pt",
                                        height=config.resolution,
                                        width=config.resolution,
                                        noise_level=config.sample.noise_level,
                                    )
                                    
                                    val_log_probs = torch.stack(val_log_probs, dim=1)  # (K, num_steps)
                                    val_latents_tensor = torch.stack(val_latents, dim=1)  # (K, num_steps+1, ...)
                                    val_noise_preds_tensor = torch.stack(val_noise_preds, dim=1)  # (K, num_steps, ...)
                                    
                                    # Compute x_0 prediction and reward at each timestep
                                    sigmas = pipeline.scheduler.sigmas
                                    val_x0_rewards_per_step = []  # (num_steps,) list of (K,) tensors
                                    
                                    for t_idx in range(config.sample.num_steps):
                                        x_t = val_latents_tensor[:, t_idx]  # (K, ...)
                                        noise_pred_t = val_noise_preds_tensor[:, t_idx]  # (K, ...)
                                        timestep_t = pipeline.scheduler.timesteps[t_idx]
                                        
                                        step_index = pipeline.scheduler.index_for_timestep(timestep_t)
                                        sigma_t = sigmas[step_index].to(x_t.device)
                                        
                                        # Single-step Euler to predict x_0: x_0 = x_t - sigma_t * v_t
                                        x_0_pred = x_t.float() - sigma_t * noise_pred_t.float()
                                        
                                        # Decode x_0 to image
                                        x_0_unpacked = pipeline._unpack_latents(
                                            x_0_pred, config.resolution, config.resolution, pipeline.vae_scale_factor
                                        )
                                        x_0_unpacked = (x_0_unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                                        x_0_unpacked = x_0_unpacked.to(dtype=pipeline.vae.dtype)
                                        
                                        x_0_images = pipeline.vae.decode(x_0_unpacked, return_dict=False)[0]
                                        x_0_images = pipeline.image_processor.postprocess(x_0_images, output_type="pt")
                                        
                                        # Compute reward for x_0 predictions
                                        x_0_rewards_dict, _ = eval_reward_fn(x_0_images, val_prompts_repeated, None)
                                        x_0_rewards = torch.as_tensor(x_0_rewards_dict['avg'], device=accelerator.device).float()
                                        val_x0_rewards_per_step.append(x_0_rewards)  # (K,)
                                    
                                    val_x0_rewards_per_step = torch.stack(val_x0_rewards_per_step, dim=1)  # (K, num_steps)
                                    
                                    # Compute final image rewards
                                    val_final_rewards_dict, _ = eval_reward_fn(val_images, val_prompts_repeated, None)
                                    val_final_rewards = torch.as_tensor(val_final_rewards_dict['avg'], device=accelerator.device).float()  # (K,)
                                    
                                    # Compute advantages for each timestep's x_0 reward
                                    # advantage = (reward - mean) / (std + eps)
                                    val_x0_advantages_per_step = (val_x0_rewards_per_step - val_x0_rewards_per_step.mean(dim=0, keepdim=True)) / (val_x0_rewards_per_step.std(dim=0, keepdim=True) + 1e-4)
                                    val_final_advantages = (val_final_rewards - val_final_rewards.mean()) / (val_final_rewards.std() + 1e-4)
                                    
                                    # Log final image reward and advantage statistics (skip per-timestep logging)
                                    val_info = {}
                                    val_info["val_final_reward_mean"] = val_final_rewards.mean()
                                    val_info["val_final_reward_std"] = val_final_rewards.std()
                                    val_info["val_final_adv_std"] = val_final_advantages.std()
                                    
                                    # Store for logging images
                                    val_images_for_log = val_images
                                    val_prompts_for_log = val_prompt
                                    
                                    if accelerator.is_local_main_process:
                                        print(f"[Val K={K}] prompt: {val_prompt[:50]}...")
                                        print(f"[Val] final_reward: mean={val_final_rewards.mean().item():.4f}, std={val_final_rewards.std().item():.4f}")
                        
                        pipeline.transformer.train()
                        
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        info.update({k: v.item() if torch.is_tensor(v) else v for k, v in val_info.items()})
                        
                        # Log validation images (simplified)
                        if accelerator.is_main_process and val_images_for_log is not None:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                num_val_images = min(len(val_images_for_log), 8)  # Log at most 8 images
                                info["val_images"] = []
                                for idx in range(num_val_images):
                                    img = val_images_for_log[idx]
                                    pil = Image.fromarray(
                                        (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                                    )
                                    pil.save(os.path.join(tmpdir, f"val_{idx}.jpg"))
                                    info["val_images"].append(
                                        swanlab.Image(
                                            os.path.join(tmpdir, f"val_{idx}.jpg"),
                                            caption=f"prompt: {val_prompts_for_log[:50]}...",
                                        )
                                    )
                            swanlab.log(info, step=global_step)
                        elif accelerator.is_main_process:
                            swanlab.log(info, step=global_step)
                        
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        log_cuda_memory(accelerator, f"epoch{epoch} train")
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

