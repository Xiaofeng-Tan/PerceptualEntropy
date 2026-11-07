import os

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs between saving model checkpoints.
    config.save_freq = 20
    # number of epochs between evaluating the model.
    config.eval_freq = 20
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 5
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # whether or not to use LoRA.
    config.use_lora = True
    config.dataset = ""
    config.resolution = 768
    config.activation_checkpointing = False
    config.fsdp_optimizer_offload = False
    # Fit FLUX LoRA training on a 40 GB card: stream T5 blocks from pinned host memory,
    # share one PickScore between train and eval rewards, slice VAE decode, and gather
    # images for the variance logs to host in chunks. Numerics are unchanged.
    config.mem40g = False
    # SwanLab project to log into; override with SWANLAB_PROJECT.
    config.swanlab_project = os.environ.get("SWANLAB_PROJECT", "perceptual-entropy")

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps for collecting dataset.
    sample.num_steps = 40
    # number of sampler inference steps for evaluation.
    sample.eval_num_steps = 40
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 4.5
    # classifier-free guidance weight for evaluation. 1.0 is no guidance.
    sample.eval_guidance_scale = 4.5
    # batch size (per GPU!) to use for sampling.
    sample.train_batch_size = 1
    sample.num_image_per_prompt = 1
    sample.test_batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 2
    # Whether use all samples in a batch to compute std
    sample.global_std = True
    # noise level
    sample.noise_level = 0.7
    # Whether to use the same noise for the same prompt
    sample.same_latent = False
    # Whether to mix same_latent mode
    sample.mix = False
    # sde window size
    sample.sde_window_size = 2
    # sde window range
    sample.sde_window_range = (0, 10)
    
    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the PPO clip range.
    train.clip_range = 1e-4
    # the high PPO clip range (for rationorm).
    train.highclip_range = 1e-4
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # kl ratio
    train.beta = 0.0
    # reinforce loss weight: -p_r / p_old * log p_theta
    train.reinforce_weight = 0.0
    # maximize log_p only (ignore reward, just maximize entropy)
    train.maximize_log_p_only = False
    # compute intermediate x_0 reward embedding log probability (expensive, requires VAE decode at each step)
    train.compute_intermediate_reward_emb_log_p = False
    # use intermediate x_0 rewards instead of final image reward for each timestep
    train.use_intermediate_rewards = False
    # entropy weight for reward augmentation: augmented_reward = reward + entropy_weight * entropy
    # entropy is computed as -log_prob in reward embedding space (higher entropy = more diverse)
    train.entropy_weight = 0.0
    # use latent space entropy instead of reward embedding entropy
    # if True, entropy is computed from the original latent space (final latents)
    train.use_latent_entropy = False
    # entropy regularization weight for differentiable entropy loss: loss = loss - entropy_reg_weight * H
    # H = -mean(log_prob), so this encourages higher entropy (more exploration)
    # This is different from entropy_weight which augments rewards non-differentiably
    train.entropy_reg_weight = 0.0
    # entropy target: "reward" (add entropy to reward before computing advantage) or "advantage" (add entropy to advantage after)
    train.entropy_target = "reward"
    # dual entropy mode: A = A + reward_emb_entropy_weight * H(reward_emb) - latent_entropy_weight * H(latent)
    # This encourages semantic diversity while constraining latent space from over-exploration
    train.use_dual_entropy = False
    train.reward_emb_entropy_weight = 0.1  # Weight for H(reward_emb), encourages semantic diversity
    train.latent_entropy_weight = 0.1  # Weight for H(latent), penalizes latent space over-dispersion
    # pretrained lora path
    train.lora_path = None
    # save ema model
    train.ema = False
    # rationorm: normalize ratio by sqrt(dt) * sigma_t for GRPO Guard
    config.rationorm = False
    
    # ====== Advanced Sample Selection Settings ======
    # Setting 1: Only keep extreme samples (highest and lowest k/2 samples based on covariance)
    train.use_extreme_samples_only = False  # Whether to only use extreme samples
    train.extreme_samples_ratio = 0.5  # Ratio of samples to keep (k/2 from top + k/2 from bottom)
    
    # Setting 2: KL constraint for high entropy samples
    train.use_entropy_kl_constraint = False  # Whether to apply KL constraint on high entropy samples
    train.entropy_kl_ratio = 0.25  # k/4 ratio of high entropy samples to apply KL constraint
    train.entropy_kl_weight = 0.001  # Weight for the KL constraint on high entropy samples

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = "imagenet_animals"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = ml_collections.ConfigDict()
    config.save_dir = ''

    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = True

    return config
