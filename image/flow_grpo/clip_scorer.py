import os
import torch
import torch.nn.functional as F
from typing import List, Union
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 本地模型缓存路径
LOCAL_MODEL_CACHE = "./pretrained_ckpt/reward_models"


def get_local_or_hf(model_name, local_subdir=None, model_dir=None):
    """检查本地是否有模型，没有则返回 huggingface 路径"""
    if local_subdir is None:
        local_subdir = model_name.replace("/", "_")
    
    if model_dir:
        local_path = os.path.join(model_dir, local_subdir)
    else:
        local_path = os.path.join(LOCAL_MODEL_CACHE, local_subdir)
    
    if os.path.exists(local_path) and os.listdir(local_path):
        return local_path
    return model_name


class ClipScorer:
    def __init__(self, clip_model_path="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", device="cuda", 
                 dtype=torch.float32, http_proxy=None, https_proxy=None, clip_model_type='ViT-H-14',
                 model_dir=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        
        # 验证和规范化设备
        device = self._validate_device(device)
        
        self.clip_model_path = clip_model_path
        self.clip_model_type = clip_model_type
        self.device = device
        self.dtype = dtype
        self.model_dir = model_dir
        self.load_model()

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

    def load_model(self):
        model_loaded = False
        
        # 优先检查 ModelScope 下载的路径
        if self.model_dir:
            modelscope_path = os.path.join(self.model_dir, "clip/openai-mirror/clip-vit-base-patch32")
            if os.path.exists(modelscope_path):
                print(f"✅ 从 ModelScope 加载 CLIP: {modelscope_path}")
                self.model, self.preprocess = create_model_from_pretrained(modelscope_path)
                model_loaded = True
        
        # 其次检查用户提供的自定义路径
        if not model_loaded and self.model_dir and not self.clip_model_path.startswith("hf-hub:"):
            local_path = os.path.join(self.model_dir, self.clip_model_path)
            if os.path.exists(local_path):
                print(f"✅ 从本地加载 CLIP: {local_path}")
                self.model, self.preprocess = create_model_from_pretrained(local_path)
                model_loaded = True
        
        # 最后使用默认路径
        if not model_loaded:
            print(f"从 HuggingFace 加载 CLIP: {self.clip_model_path}")
            self.model, self.preprocess = create_model_from_pretrained(self.clip_model_path)
        
        self.tokenizer = get_tokenizer(self.clip_model_type)
        self.model.to(self.device)
        
    @torch.no_grad()
    def __call__(
        self, 
        prompts: Union[str, List[str]], 
        images: List[Image.Image]
    ) -> List[float]:
        """
        Calculate CLIP scores for images given prompts.
        
        Args:
            prompts: Text prompt(s) to evaluate images against
            images: List of PIL Images to evaluate
            
        Returns:
            List of CLIP scores for each image
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        scores = []
        for prompt, image in zip(prompts, images):
            image_proc = self.preprocess(image).unsqueeze(0).to(self.device)
            text = self.tokenizer(
                [prompt], 
                context_length=self.model.context_length
            ).to(self.device)

            image_features = self.model.encode_image(image_proc)
            text_features = self.model.encode_text(text)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            clip_score = image_features @ text_features.T

            scores.append(clip_score.item())

        return scores


# 保持向后兼容的别名
CLIPScoreRewardModel = ClipScorer


def main():
    clip_model_path = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
    scorer = ClipScorer(
        clip_model_path=clip_model_path,
        device="cuda"
    )
    
    image_path = "assets/test.jpg"
    prompt = "A 3D rendering of anime schoolgirls with a sad expression underwater, surrounded by dramatic lighting."
    
    image = Image.open(image_path).convert("RGB")
    clip_score = scorer(prompt, [image])

    print(f"CLIP Score: {clip_score}")


if __name__ == "__main__":
    main()
