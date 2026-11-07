import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False, buffer_size=64, min_count=16):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        # DDPO Appendix D.3: running mean/std per prompt, kept across sampling rounds.
        self.history = {}
        self.buffer_size = buffer_size
        self.min_count = min_count

    def update(self, prompts, rewards, log_probs=None, type='grpo'):
        """
        Update statistics and compute advantages.
        
        Args:
            prompts: List of prompts
            rewards: numpy array of rewards, shape (N,) or (N, T)
            log_probs: numpy array of log probabilities, shape (N,) or (N, T), optional
            type: Type of advantage computation ('grpo', 'rwr', 'sft', 'dpo')
            
        Returns:
            advantages: numpy array of advantages
            covariances: numpy array of covariances (if log_probs provided), else None
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        
        # Initialize covariances if log_probs provided
        covariances = None
        if log_probs is not None:
            log_probs = np.array(log_probs, dtype=np.float64)
            covariances = np.empty_like(rewards) * 0.0
        
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[prompt_mask] = (prompt_rewards - mean) / std
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompt_mask] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompt_mask] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompt_mask] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
            # Compute covariance if log_probs provided
            # Cov(yi) = (log πθ(yi) - mean(log πθ)) · (A(yi) - mean(A))
            if log_probs is not None:
                prompt_log_probs = log_probs[prompt_mask]
                prompt_advantages = advantages[prompt_mask]
                
                # Compute centered values
                log_prob_mean = np.mean(prompt_log_probs, axis=0, keepdims=True)
                adv_mean = np.mean(prompt_advantages, axis=0, keepdims=True)
                
                log_prob_centered = prompt_log_probs - log_prob_mean
                adv_centered = prompt_advantages - adv_mean
                
                # Handle shape mismatch: if log_probs is 1D and advantages is 2D
                # (e.g., log_probs shape (N,) and advantages shape (N, T) with intermediate rewards)
                if len(log_prob_centered.shape) == 1 and len(adv_centered.shape) == 2:
                    log_prob_centered = log_prob_centered[:, None]  # (N,) -> (N, 1) for broadcasting
                
                # Covariance = (log_p - mean_log_p) * (A - mean_A)
                cov = log_prob_centered * adv_centered
                covariances[prompt_mask] = cov
            
        if log_probs is not None:
            return advantages, covariances
        return advantages

    def update_running(self, prompts, rewards):
        """DDPO per-prompt baseline: normalize by a running mean and std.

        History is a bounded deque per prompt and is not wiped between epochs.
        Until a prompt has ``min_count`` rewards, fall back to the current
        batch mean and std, matching ddpo-pytorch.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        # Final reward is repeated over timesteps; the baseline is on that scalar.
        timestep_rewards = rewards[:, 0] if rewards.ndim == 2 else rewards
        unique = np.unique(prompts)
        advantages_1d = np.zeros(len(prompts), dtype=np.float64)
        global_mean = float(np.mean(timestep_rewards))
        global_std = float(np.std(timestep_rewards)) + 1e-4

        for prompt in unique:
            mask = prompts == prompt
            prompt_rewards = timestep_rewards[mask]
            if prompt not in self.history:
                self.history[prompt] = deque(maxlen=self.buffer_size)
            self.history[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))
            hist = np.asarray(self.history[prompt], dtype=np.float64)
            if len(hist) < self.min_count:
                mean, std = global_mean, global_std
            else:
                mean = float(np.mean(hist))
                std = float(np.std(hist)) + 1e-4
            advantages_1d[mask] = (prompt_rewards - mean) / std

        if rewards.ndim == 2:
            return np.repeat(advantages_1d[:, None], rewards.shape[1], axis=1)
        return advantages_1d
    
    def select_extreme_samples(self, prompts, covariances, ratio=0.25, mode='drop_high'):
        """
        Select samples based on covariance filtering.
        
        Args:
            prompts: List of prompts
            covariances: numpy array of covariances, shape (N,) or (N, T)
            ratio: Ratio of samples to drop (for mode='drop_high') or keep from each extreme (for mode='keep_extreme')
            mode: 'drop_high' - drop top ratio samples with highest covariance
                  'keep_extreme' - keep top+bottom ratio/2 samples (original behavior)
            
        Returns:
            mask: Boolean mask indicating which samples to keep
        """
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        
        # Handle multi-dimensional covariances by taking mean across timesteps
        if len(covariances.shape) > 1:
            cov_for_selection = np.mean(covariances, axis=1)
        else:
            cov_for_selection = covariances
        
        mask = np.zeros(len(prompts), dtype=bool)
        
        for prompt in unique:
            prompt_mask = prompts == prompt
            prompt_indices = np.where(prompt_mask)[0]
            prompt_cov = cov_for_selection[prompt_mask]
            
            group_size = len(prompt_cov)
            
            if mode == 'drop_high':
                # Drop top ratio samples with highest covariance
                k_drop = max(1, int(group_size * ratio))  # At least 1 sample to drop
                
                if group_size <= k_drop:
                    # If group is too small, keep all samples
                    mask[prompt_indices] = True
                else:
                    # Get indices sorted by covariance
                    sorted_indices = np.argsort(prompt_cov)
                    # Keep all except top k_drop (highest covariance)
                    keep_indices = sorted_indices[:-k_drop]
                    mask[prompt_indices[keep_indices]] = True
            else:
                # Original behavior: keep extreme samples (top + bottom)
                k_half = max(1, int(group_size * ratio / 2))
                
                if group_size <= 2 * k_half:
                    mask[prompt_indices] = True
                else:
                    sorted_indices = np.argsort(prompt_cov)
                    bottom_indices = sorted_indices[:k_half]
                    top_indices = sorted_indices[-k_half:]
                    selected_local_indices = np.concatenate([bottom_indices, top_indices])
                    mask[prompt_indices[selected_local_indices]] = True
        
        return mask
    
    def get_high_covariance_samples(self, prompts, covariances, ratio=0.25):
        """
        Get mask for high covariance samples (highest |covariance| = most informative for KL constraint).
        
        Covariance is computed as:
        Cov(yi) = (log πθ(yi) - mean(log πθ)) · (A(yi) - mean(A))
        
        High |covariance| samples are where log probability and advantage are most correlated,
        which are the most informative samples that may benefit from KL constraint.
        
        Args:
            prompts: List of prompts
            covariances: numpy array of covariances, shape (N,) or (N, T)
            ratio: Ratio of high covariance samples to select (k/4)
            
        Returns:
            mask: Boolean mask indicating high covariance samples
        """
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        
        # Handle multi-dimensional covariances by taking mean across timesteps
        if len(covariances.shape) > 1:
            cov_for_selection = np.mean(covariances, axis=1)
        else:
            cov_for_selection = covariances
        
        # Use absolute value of covariance for selection
        abs_cov = np.abs(cov_for_selection)
        
        mask = np.zeros(len(prompts), dtype=bool)
        
        for prompt in unique:
            prompt_mask = prompts == prompt
            prompt_indices = np.where(prompt_mask)[0]
            prompt_abs_cov = abs_cov[prompt_mask]
            
            group_size = len(prompt_abs_cov)
            k_quarter = max(1, int(group_size * ratio))  # At least 1 sample
            
            if group_size <= k_quarter:
                # If group is too small, select all as high covariance
                mask[prompt_indices] = True
            else:
                # Higher |covariance| = more informative, so get top k_quarter
                sorted_indices = np.argsort(prompt_abs_cov)
                high_cov_indices = sorted_indices[-k_quarter:]  # Highest |covariance|
                
                mask[prompt_indices[high_cov_indices]] = True
        
        return mask

    def get_stats(self):
        groups = self.history if self.history else self.stats
        avg_group_size = sum(len(v) for v in groups.values()) / len(groups) if groups else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    log_probs = [-0.5, -0.3, -0.8, -0.2, -0.6, -0.4]
    
    # Test with covariance computation
    advantages, covariances = tracker.update(prompts, rewards, log_probs=log_probs)
    print("Advantages:", advantages)
    print("Covariances:", covariances)
    
    # Test extreme sample selection
    tracker.clear()
    tracker.update(prompts, rewards)  # Reset stats
    mask = tracker.select_extreme_samples(prompts, covariances, ratio=0.5)
    print("Extreme samples mask:", mask)
    
    # Test high covariance sample selection
    high_cov_mask = tracker.get_high_covariance_samples(prompts, covariances, ratio=0.25)
    print("High covariance mask:", high_cov_mask)
    
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()