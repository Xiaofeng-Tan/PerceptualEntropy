# Image-Reward: Copied from https://github.com/THUDM/ImageReward
import os
from typing import Union, List
from PIL import Image

import torch
try:
    import ImageReward as RM
except:
    raise Warning("ImageReward is required to be installed (`pip install image-reward`) when using ImageReward for post-training.")

# 本地模型缓存路径
LOCAL_MODEL_CACHE = "./pretrained_ckpt/reward_models"


def get_local_or_hf(model_name, local_subdir=None, model_dir=None):
    """检查本地是否有模型，没有则返回 huggingface/默认路径"""
    # 尝试多个可能的子目录名（支持拼写变体）
    possible_subdirs = []
    if local_subdir:
        possible_subdirs.append(local_subdir)
    # 添加常见的变体
    possible_subdirs.extend(["imagerward", "imagereward_v1", "imagereward"])
    
    base_dir = model_dir if model_dir else LOCAL_MODEL_CACHE
    
    for subdir in possible_subdirs:
        if subdir:
            local_path = os.path.join(base_dir, subdir)
            if os.path.exists(local_path) and os.listdir(local_path):
                print(f"✅ 找到本地 ImageReward 模型: {local_path}")
                return local_path
    
    print(f"⚠️  本地未找到 ImageReward 模型，将使用: {model_name}")
    return model_name


class ImageRewardScorer(object):
    def __init__(self, model_name=None, device="cuda", dtype=torch.float32, 
                 http_proxy=None, https_proxy=None, med_config=None, model_dir=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        
        # 验证和规范化设备
        device = self._validate_device(device)
        
        self.model_name = model_name if model_name else "ImageReward-v1.0"
        self.device = device
        self.dtype = dtype
        self.med_config = med_config
        
        # 检查本地缓存
        model_path = get_local_or_hf(self.model_name, "imagereward_v1", model_dir)
        self.build_reward_model(model_path)

    def _validate_device(self, device):
        """验证 CUDA 设备是否有效"""
        if isinstance(device, str):
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    if device == "cuda":
                        device = "cuda:0"
                    try:
                        gpu_idx = int(device.split(':')[1])
                        if gpu_idx >= torch.cuda.device_count():
                            print(f"⚠️  GPU {gpu_idx} 不存在，可用 GPU 数: {torch.cuda.device_count()}")
                            device = "cuda:0"
                    except (ValueError, IndexError):
                        device = "cuda:0"
                else:
                    print("⚠️  CUDA 不可用，使用 CPU")
                    device = "cpu"
        return device

    def build_reward_model(self, model_path):
        self.model = RM.load(model_path, device=self.device, med_config=self.med_config)

    @torch.no_grad()
    def __call__(
            self,
            images: List[Image.Image],
            texts: Union[str, List[str]],
    ) -> List[float]:
        """
        Calculate ImageReward scores for images given texts.
        
        Args:
            images: List of PIL Images to evaluate
            texts: Text prompt(s) to evaluate images against
            
        Returns:
            List of reward scores for each image
        """
        if isinstance(texts, str):
            texts = [texts] * len(images)
        
        rewards = []
        for image, text in zip(images, texts):
            ranking, reward = self.model.inference_rank(text, [image])
            rewards.append(reward)
        return rewards


# 保持向后兼容的别名
ImageRewardModel = ImageRewardScorer


# Usage example
def main():
    scorer = ImageRewardScorer(
        device="cuda",
        dtype=torch.float32
    )

    images = ["astronaut.jpg"]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'A astronaut\'s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(pil_images, prompts))


if __name__ == "__main__":
    main()
