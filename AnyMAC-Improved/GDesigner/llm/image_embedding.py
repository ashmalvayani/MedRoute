"""
Image embedding module using frozen vision encoders.

Supported encoders:
  - google/siglip-so400m-patch14-384 (SigLIP, 1152-dim)
  - openai/clip-vit-large-patch14-336 (CLIP, 1024-dim)
  - facebook/dinov2-large (DINOv2, 1024-dim)
  - microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (BiomedCLIP, 768-dim)

Mirrors the interface of profile_embedding.py:
  - Lazy model loading (first call)
  - In-memory cache keyed by image path
  - Returns numpy arrays
"""

import logging
import numpy as np
import torch
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)

# Default vision encoder
DEFAULT_VISION_MODEL = "google/siglip-so400m-patch14-384"

# Cached model + processor
_model = None
_processor = None
_model_name = None
_embedding_dim = None
_model_type = None  # "siglip", "clip", "dino", "biomedclip"

# Embedding cache: image_path -> numpy array
_embedding_cache: dict = {}


def _detect_model_type(model_name: str) -> str:
    name_lower = model_name.lower()
    if "biomedclip" in name_lower:
        return "biomedclip"
    elif "dinov2" in name_lower or "dino" in name_lower:
        return "dino"
    elif "clip" in name_lower and "siglip" not in name_lower:
        return "clip"
    else:
        return "siglip"


def _load_model(model_name: str = None):
    global _model, _processor, _model_name, _embedding_dim, _model_type
    model_name = model_name or DEFAULT_VISION_MODEL
    if _model is not None and _model_name == model_name:
        return

    _model_type = _detect_model_type(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[image_embedding] Loading vision encoder: {model_name} (type={_model_type})")

    if _model_type == "biomedclip":
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:' + model_name
        )
        model.eval()
        model = model.to(device)
        _model = model
        _processor = preprocess_val
        # Detect dim
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            emb = model.encode_image(dummy)
            _embedding_dim = emb.shape[-1]
    elif _model_type == "dino":
        from transformers import AutoModel, AutoImageProcessor
        _processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _model.eval()
        _model = _model.to(device)
        # DINOv2: use CLS token from last_hidden_state
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = _model(dummy)
            _embedding_dim = out.last_hidden_state[:, 0].shape[-1]
    else:
        # SigLIP / CLIP — both use AutoModel with .vision_model
        from transformers import AutoModel, AutoImageProcessor
        _processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _model.eval()
        _model = _model.to(device)
        # Detect dim via dummy forward
        img_size = 384 if "384" in model_name else 336 if "336" in model_name else 224
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            out = _model.vision_model(dummy)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                _embedding_dim = out.pooler_output.shape[-1]
            else:
                _embedding_dim = out.last_hidden_state[:, 0].shape[-1]

    _model_name = model_name
    print(f"[image_embedding] Loaded. Embedding dim: {_embedding_dim}, device: {device}")


def get_embedding_dim(model_name: str = None) -> int:
    """Return the embedding dimension for the vision encoder."""
    _load_model(model_name)
    return _embedding_dim


def _extract_embedding(images_tensor=None, pil_images=None):
    """Extract embeddings based on model type. Returns tensor on model device."""
    device = next(_model.parameters()).device

    if _model_type == "biomedclip":
        # open_clip: preprocess returns tensor, use encode_image
        if pil_images is not None:
            batch = torch.stack([_processor(img) for img in pil_images]).to(device)
        else:
            batch = images_tensor
        with torch.no_grad():
            return _model.encode_image(batch)
    elif _model_type == "dino":
        # DINOv2: direct forward, CLS token
        if pil_images is not None:
            inputs = _processor(images=pil_images, return_tensors="pt").to(device)
        else:
            inputs = {"pixel_values": images_tensor}
        with torch.no_grad():
            out = _model(**inputs)
            return out.last_hidden_state[:, 0]
    else:
        # SigLIP / CLIP
        if pil_images is not None:
            inputs = _processor(images=pil_images, return_tensors="pt").to(device)
        else:
            inputs = {"pixel_values": images_tensor}
        with torch.no_grad():
            out = _model.vision_model(**inputs)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                return out.pooler_output
            else:
                return out.last_hidden_state[:, 0]


def get_image_embedding(image_path: str, model_name: str = None) -> np.ndarray:
    """
    Get image embedding from the vision encoder.

    Args:
        image_path: Path to the image file
        model_name: Optional model name override

    Returns:
        numpy array of shape (embedding_dim,)
    """
    if image_path in _embedding_cache:
        return _embedding_cache[image_path]

    _load_model(model_name)

    from PIL import Image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[image_embedding] Failed to load {image_path}: {e}")
        emb = np.zeros(_embedding_dim, dtype=np.float32)
        _embedding_cache[image_path] = emb
        return emb

    emb = _extract_embedding(pil_images=[img]).squeeze(0)
    emb = emb.cpu().numpy().astype(np.float32)
    _embedding_cache[image_path] = emb
    return emb


def precompute_embeddings(image_paths: list, model_name: str = None) -> dict:
    """
    Batch-precompute embeddings for a list of image paths.
    Returns dict mapping path -> numpy array.
    """
    _load_model(model_name)

    missing = [p for p in image_paths if p not in _embedding_cache]
    if not missing:
        return {p: _embedding_cache[p] for p in image_paths}

    from PIL import Image
    print(f"[image_embedding] Pre-computing embeddings for {len(missing)} images...")
    batch_size = 32

    for i in range(0, len(missing), batch_size):
        batch_paths = missing[i:i + batch_size]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"  WARNING: {p}: {e}")
                _embedding_cache[p] = np.zeros(_embedding_dim, dtype=np.float32)

        if not images:
            continue

        embs = _extract_embedding(pil_images=images)
        for p, emb in zip(valid_paths, embs):
            _embedding_cache[p] = emb.cpu().numpy().astype(np.float32)

    print(f"[image_embedding] Done. {len(_embedding_cache)} embeddings cached.")
    return {p: _embedding_cache[p] for p in image_paths if p in _embedding_cache}


def clear_embedding_cache():
    _embedding_cache.clear()
