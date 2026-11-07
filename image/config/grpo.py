import importlib.util
import os

import ml_collections

# importlib instead of imp: imp was removed in Python 3.12.
_base_spec = importlib.util.spec_from_file_location(
    "base", os.path.join(os.path.dirname(__file__), "base.py")
)
base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(base)

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def _pickscore_flux_base():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = os.path.join(os.getcwd(), "pretrained_ckpt/flux")
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.mix = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_ours_2'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def _pickscore_flux_ddpo_base():
    """DDPO_IS (Black et al., 2023) on FLUX.

    The flow SDE log-prob is unchanged: FLUX has no DDPM Gaussian reverse step.
    The policy loss is the clipped importance-sampling objective with clip 1e-4.
    Advantages use each prompt's running mean and std, and each sampling round
    takes 4 optimizer steps.
    """
    config = _pickscore_flux_base()
    config.algorithm = "ddpo"
    config.train.clip_range = 1e-4
    config.train.beta = 0.0
    config.train.reinforce_weight = 0.0
    config.train.num_inner_epochs = 1
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 4
    config.train.ddpo_buffer_size = 64
    config.train.ddpo_min_count = 16
    config.per_prompt_stat_tracking = True
    config.run_name = "flux-ddpo"
    config.save_dir = "logs/pec_ddpo"
    return config

def pickscore_flux_8gpu_ddpo_pec_003():
    """DDPO advantage on PEC reward: r' = r + 0.03 * H(reward_emb), no importance weight."""
    config = _pickscore_flux_ddpo_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.0
    config.train.use_importance_weighting = False
    config.run_name = "flux-ddpo-pec-a003"
    config.save_dir = "logs/ddpo_pec_003"
    return config

def pickscore_flux_8gpu_ddpo_pcvae_003():
    """DDPO advantage on PCVAE reward: r' = r + 0.03 * H(reward_emb) - 0.03 * H(latent)."""
    config = _pickscore_flux_ddpo_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.03
    config.train.use_importance_weighting = False
    config.run_name = "flux-ddpo-pcvae-a003"
    config.save_dir = "logs/ddpo_pcvae_003"
    return config

def pickscore_flux_8gpu_ddpo_pcvae_008():
    """DDPO advantage on PCVAE reward: r' = r + 0.08 * H(reward_emb) - 0.08 * H(latent)."""
    config = _pickscore_flux_ddpo_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.08
    config.train.latent_entropy_weight = 0.08
    config.train.use_importance_weighting = False
    config.run_name = "flux-ddpo-pcvae-a008"
    config.save_dir = "logs/ddpo_pcvae_008"
    return config

def pickscore_flux_8gpu_dual_entropy():
    """
    Dual entropy mode: A = A + w1 * H(reward_emb) - w2 * H(latent)
    Encourages semantic diversity (high reward embedding entropy) while
    constraining latent space from over-dispersion (penalize high latent entropy).
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.reward_emb_entropy_weight = 0.1  # Encourage semantic diversity
    config.train.latent_entropy_weight = 0.1  # Constrain latent space
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy'
    return config

def pickscore_flux_8gpu_dual_entropy_001_001():
    """
    Dual entropy mode with equal weights: A = A + 0.1 * H(reward_emb) - 0.1 * H(latent)
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.reward_emb_entropy_weight = 0.01
    config.train.latent_entropy_weight = 0.01
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_001_001'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_005_005():
    """
    Dual entropy mode applied to REWARD instead of advantage:
    r' = r + w1 * H(reward_emb) - w2 * H(latent)
    A = normalize(r')
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"  # Apply to reward, not advantage
    config.train.reward_emb_entropy_weight = 0.05
    config.train.latent_entropy_weight = 0.05
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_005_005'
    return config

def pickscore_flux_8gpu_pcvae_01():
    """PCVAE: r' = r + 0.1 * H(reward_emb) - 0.1 * H(latent), no importance weighting."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.1
    config.train.latent_entropy_weight = 0.1
    config.train.use_importance_weighting = False
    config.run_name = "flux-pcvae-a01"
    config.save_dir = "logs/pcvae_01"
    return config

def pickscore_flux_8gpu_pcvae_003():
    """PCVAE reproduction of dual_entropy_on_reward_003_003, without importance weighting."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.03
    config.train.use_importance_weighting = False
    config.run_name = "flux-pcvae-a003"
    config.save_dir = "logs/pcvae_003"
    return config

def pickscore_flux_8gpu_pcvae_005():
    """PCVAE: r' = r + 0.05 * H(reward_emb) - 0.05 * H(latent)."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.05
    config.train.latent_entropy_weight = 0.05
    config.run_name = "flux-pcvae-a005"
    config.save_dir = "logs/pcvae_005"
    return config

def pickscore_flux_8gpu_pcvae_001():
    """PCVAE: r' = r + 0.01 * H(reward_emb) - 0.01 * H(latent)."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.01
    config.train.latent_entropy_weight = 0.01
    config.run_name = "flux-pcvae-a001"
    config.save_dir = "logs/pcvae_001"
    return config

def pickscore_flux_8gpu_pcvae_0001():
    """PCVAE: r' = r + 0.001 * H(reward_emb) - 0.001 * H(latent)."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.001
    config.train.latent_entropy_weight = 0.001
    config.train.use_importance_weighting = False
    config.run_name = "flux-pcvae-a0001"
    config.save_dir = "logs/pcvae_0001"
    return config

def pickscore_flux_8gpu_pcvae_00001():
    """PCVAE: r' = r + 0.0001 * H(reward_emb) - 0.0001 * H(latent)."""
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.0001
    config.train.latent_entropy_weight = 0.0001
    config.train.use_importance_weighting = False
    config.run_name = "flux-pcvae-a00001"
    config.save_dir = "logs/pcvae_00001"
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_003_003():
    """
    Dual entropy mode applied to REWARD instead of advantage:
    r' = r + w1 * H(reward_emb) - w2 * H(latent)
    A = normalize(r')
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"  # Apply to reward, not advantage
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.03
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_003_003'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_01_01():
    """
    Dual entropy mode applied to REWARD instead of advantage:
    r' = r + w1 * H(reward_emb) - w2 * H(latent)
    A = normalize(r')
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"  # Apply to reward, not advantage
    config.train.reward_emb_entropy_weight = 0.1
    config.train.latent_entropy_weight = 0.1
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_01_01'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_008_008():
    """
    Dual entropy mode applied to REWARD instead of advantage:
    r' = r + w1 * H(reward_emb) - w2 * H(latent)
    A = normalize(r')
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"  # Apply to reward, not advantage
    config.train.reward_emb_entropy_weight = 0.08
    config.train.latent_entropy_weight = 0.08
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_008_008'
    return config

def pickscore_flux_8gpu_reward_entropy_only_01():
    """
    Reward entropy constraint WITHOUT latent entropy penalty:
    r' = r + w * H(reward_emb)
    A = normalize(r')
    
    This encourages semantic diversity in reward embedding space
    while allowing free exploration in latent space.
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"  # Apply to reward
    config.train.reward_emb_entropy_weight = 0.1   # Encourage reward embedding entropy
    config.train.latent_entropy_weight = 0.0       # No latent penalty
    config.run_name = "flux-pec-a01"
    config.save_dir = "logs/pec_01"
    return config

def pickscore_flux_8gpu_reward_entropy_only_005():
    """
    Reward entropy constraint WITHOUT latent entropy penalty:
    r' = r + w * H(reward_emb)
    A = normalize(r')
    
    Smaller weight version (0.05).
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.05
    config.train.latent_entropy_weight = 0.0
    config.train.use_importance_weighting = False
    config.run_name = "flux-pec-a005"
    config.save_dir = "logs/pec_005"
    return config

def pickscore_flux_8gpu_reward_entropy_only_001():
    """
    Reward entropy constraint WITHOUT latent entropy penalty:
    r' = r + w * H(reward_emb)
    A = normalize(r')

    Smaller weight version (0.01).
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.01
    config.train.latent_entropy_weight = 0.0
    config.run_name = "flux-pec-a001"
    config.save_dir = "logs/pec_001"
    return config

def pickscore_flux_8gpu_reward_entropy_only_003():
    """
    Reward entropy constraint WITHOUT latent entropy penalty:
    r' = r + w * H(reward_emb)
    A = normalize(r')
    
    Smaller weight version (0.03).
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.0
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_reward_entropy_only_003'
    return config

# ── No-importance-weighting variants (original formula: w * H) ──

def pickscore_flux_8gpu_reward_entropy_only_005_no_iw():
    """
    PEC without importance weighting (original formula):
    r' = r + w * H(reward_emb)
    (No p_emb / π_old multiplication)
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.05
    config.train.latent_entropy_weight = 0.0
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_reward_entropy_only_005_no_iw'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_01_01_no_iw():
    """
    PCVAE without importance weighting (original formula):
    r' = r + w1 * H(reward_emb) - w2 * H(latent)
    (No importance weight multiplication)
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.1
    config.train.latent_entropy_weight = 0.1
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_01_01_no_iw'
    return config

def pickscore_flux_8gpu_reward_entropy_only_003_no_iw():
    """
    PEC without importance weighting (original formula):
    r' = r + w * H(reward_emb), w=0.03
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.0
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_reward_entropy_only_003_no_iw'
    return config

def pickscore_flux_8gpu_reward_entropy_only_01_no_iw():
    """
    PEC without importance weighting (original formula):
    r' = r + w * H(reward_emb), w=0.1
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.1
    config.train.latent_entropy_weight = 0.0
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_reward_entropy_only_01_no_iw'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_008_008_no_iw():
    """
    PCVAE without importance weighting (original formula):
    r' = r + w1 * H(reward_emb) - w2 * H(latent), w1=w2=0.08
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.08
    config.train.latent_entropy_weight = 0.08
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_008_008_no_iw'
    return config

def pickscore_flux_8gpu_dual_entropy_on_reward_003_003_no_iw():
    """
    PCVAE without importance weighting (original formula):
    r' = r + w1 * H(reward_emb) - w2 * H(latent), w1=w2=0.03
    """
    config = _pickscore_flux_base()
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = 0.03
    config.train.latent_entropy_weight = 0.03
    config.train.use_importance_weighting = False
    config.save_dir = 'logs/pickscore/flux-group24-8gpu_dual_entropy_on_reward_003_003_no_iw'
    return config

PEC40G_SAVE_ROOT = "logs/40g"

def _pickscore_flux_40g(gpu_number, name, batch_size=3):
    """pickscore_flux_8gpu rescaled to `gpu_number` cards with the 40 GB memory path on.

    48 prompts x 24 images per epoch and two optimizer steps per epoch are kept, so the
    global_step axis matches the 8-GPU H20 runs. gpu_number * batch_size must divide by 24.
    """
    config = _pickscore_flux_base()
    config.sample.train_batch_size = batch_size
    config.train.batch_size = batch_size
    config.mem40g = True
    config.swanlab_project = "flux_pec_40g"
    config.pretrained.model = os.path.join(os.getcwd(), "pretrained_ckpt/flux")
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.use_importance_weighting = False
    config.run_name = name
    config.save_dir = os.path.join(PEC40G_SAVE_ROOT, name)
    return config

def _pec_40g(gpu_number, name, reward_emb_w, latent_w, batch_size=3):
    """r' = r + reward_emb_w * H(reward_emb) - latent_w * H(latent), no importance weighting."""
    config = _pickscore_flux_40g(gpu_number, name, batch_size)
    config.train.use_dual_entropy = True
    config.train.dual_entropy_target = "reward"
    config.train.reward_emb_entropy_weight = reward_emb_w
    config.train.latent_entropy_weight = latent_w
    return config

def pickscore_flux_16gpu_40g_pec_005():
    return _pec_40g(16, "flux-pec-a005-16gpu40g", 0.05, 0.0)

def pickscore_flux_16gpu_40g_pec_01():
    return _pec_40g(16, "flux-pec-a01-16gpu40g", 0.1, 0.0)

def pickscore_flux_16gpu_40g_pcvae_003():
    return _pec_40g(16, "flux-pcvae-a003-16gpu40g", 0.03, 0.03)

def pickscore_flux_16gpu_40g_pcvae_01():
    return _pec_40g(16, "flux-pcvae-a01-16gpu40g", 0.1, 0.1)

def pickscore_flux_8gpu_40g_pec_005():
    """Same setting as the H20 reference rm38ec3k (8 cards), on 40 GB cards."""
    return _pec_40g(8, "flux-pec-a005-8gpu40g", 0.05, 0.0)

def pickscore_flux_12gpu_40g_pec_005():
    """12 cards x batch 2 = 24 images = one prompt per sampling round, exactly like 8 x 3, so the
    prompt sequence is the H20 reference rm38ec3k's and the curves compare point by point."""
    return _pec_40g(12, "flux-pec-a005-12gpu40g", 0.05, 0.0, batch_size=2)

def get_config(name):
    return globals()[name]()
