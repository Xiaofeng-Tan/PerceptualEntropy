# Aesthetic Predictor v2.5: Based on SigLIP
import os
from typing import Union, List
from PIL import Image

import torch
try:
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
except:
    raise Warning("aesthetic_predictor_v2_5 is required to be installed when using Aesthetic Predictor.")

# 本地模型缓存路径
LOCAL_MODEL_CACHE = "./pretrained_ckpt/reward_models"


class AestheticScorer(object):
    def __init__(self, model_name=None, device="cuda", dtype=torch.bfloat16,
                 http_proxy=None, https_proxy=None, model_dir=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        
        # 验证和规范化设备
        device = self._validate_device(device)
        
        self.model_name = model_name if model_name else "Aesthetic_PredictorModel-v2.5"
        self.device = device
        self.dtype = dtype
        self.model_dir = model_dir
        self.build_reward_model()

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

    def build_reward_model(self):
        # 如果提供了 model_dir，设置缓存目录
        if self.model_dir:
            os.environ['HF_HOME'] = self.model_dir
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.model_dir, "cache")
            print(f"✅ Aesthetic 模型将使用缓存目录: {self.model_dir}")
        
        try:
            model, preprocessor = convert_v2_5_from_siglip(
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = model.to(dtype=self.dtype, device=self.device)
            self.preprocessor = preprocessor
            print(f"✅ Aesthetic 模型加载成功")
        except Exception as e:
            print(f"⚠️  Aesthetic 模型加载失败: {e}")
            raise

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], texts=None) -> List[float]:
        """
        Calculate aesthetic scores for images.
        
        Args:
            images: List of PIL.Image objects
            texts: Placeholder parameter (not used)
            
        Returns:
            List[float]: aesthetic scores
        """
        pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        logits = self.model(pixel_values).logits.squeeze()
        scores = logits.float().cpu().numpy()

        if scores.ndim == 0:  # 单张图
            scores = [float(scores)]
        else:  # 多张图
            scores = scores.tolist()
        return scores


# 保持向后兼容的别名
Aesthetic_PredictorModel = AestheticScorer


# Usage example
def main():
    scorer = AestheticScorer(device="cuda")
    
    images = ["test.jpg"]
    pil_images = [Image.open(img).convert("RGB") for img in images]
    scores = scorer(pil_images)
    print(f"Aesthetic scores: {scores}")


if __name__ == "__main__":
    main()
