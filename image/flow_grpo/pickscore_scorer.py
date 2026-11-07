from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, model_dir=None):
        super().__init__()
        
        # Determine model path
        if model_dir:
            base_dir = model_dir
        else:
            # Default: pretrained_ckpt/ relative to project root
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_ckpt")
        
        # Check ModelScope download paths
        modelscope_clip_path = os.path.join(base_dir, "clip/openai-mirror/clip-vit-base-patch32")
        modelscope_ps_path = os.path.join(base_dir, "pickscore/AI-ModelScope/PickScore_v1")
        
        # Default paths
        processor_path = os.path.join(base_dir, "clip_processor")
        model_path = os.path.join(base_dir, "pickscore_model")
        
        # Prefer ModelScope-downloaded models
        if os.path.exists(modelscope_clip_path):
            processor_path = modelscope_clip_path
            print(f"Using ModelScope CLIP: {processor_path}")
        
        if os.path.exists(modelscope_ps_path):
            model_path = modelscope_ps_path
            print(f"Using ModelScope PickScore: {model_path}")
        elif not os.path.exists(model_path):
            # Fallback to ps/ directory
            ps_path = os.path.join(base_dir, "ps")
            if os.path.exists(ps_path):
                model_path = ps_path
                print(f"Using ps/: {model_path}")
        
        # Validate paths exist
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"PickScore processor not found: {processor_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PickScore model not found: {model_path}")
        
        print(f"Loading PickScore processor: {processor_path}")
        print(f"Loading PickScore model: {model_path}")
        
        # Validate and normalize device
        if isinstance(device, str):
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    if device == "cuda":
                        device = "cuda:0"
                    try:
                        gpu_idx = int(device.split(':')[1])
                        if gpu_idx >= torch.cuda.device_count():
                            print(f"WARNING: GPU {gpu_idx} not available, {torch.cuda.device_count()} GPU(s) found")
                            print(f"WARNING: Falling back to cuda:0")
                            device = "cuda:0"
                    except (ValueError, IndexError):
                        device = "cuda:0"
                else:
                    print("WARNING: CUDA not available, using CPU")
                    device = "cpu"
        
        self.device = device
        self.dtype = dtype

        print(f"Loading processor: {processor_path}")
        print(f"Loading model: {model_path}")
        
        self.processor = CLIPProcessor.from_pretrained(processor_path, local_files_only=True, trust_remote_code=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores/26
        return scores.cpu().tolist(), image_embs

# Usage example
def main():
    scorer = PickScoreScorer(
        device="cuda",
        dtype=torch.float32
    )
    images=[
    "nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()