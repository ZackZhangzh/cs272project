import os
from glob import glob
from typing import List, Dict, Optional
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
import csv

# transformers==4.43
# ====== Global variables ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model & text embeddings only once
print("🔄 Loading CLIP model and encoding prompts...")
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def tagging_and_grouping(
    image_dir: str, LABEL_PROMPTS, save_csv: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Tag and group images based on fixed CLIP prompts.
    Args:
        image_dir: Folder containing images.
        save_csv: Optional path to save CSV file.
    Returns:
        Dictionary {label_prompt: [image_path1, image_path2, ...]}
    """
    image_paths = sorted(glob(os.path.join(image_dir, "*")))
    grouped = defaultdict(list)
    results = []
    with torch.no_grad():
        TEXT_INPUTS = CLIP_PROCESSOR(
            text=LABEL_PROMPTS, return_tensors="pt", padding=True
        ).to(DEVICE)
        TEXT_FEATURES = CLIP_MODEL.get_text_features(**TEXT_INPUTS)
        TEXT_FEATURES = TEXT_FEATURES / TEXT_FEATURES.norm(dim=-1, keepdim=True)
    print("✅ CLIP model & text features ready.")
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            print(f"⚠️ Cannot read image: {path}")
            continue
        inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = CLIP_MODEL.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ TEXT_FEATURES.T)[0]
            best_idx = similarity.argmax().item()
            best_label = LABEL_PROMPTS[best_idx]
        grouped[best_label].append(path)
        results.append((path, best_label))
    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "predicted_label"])
            writer.writerows(results)
    return grouped


if __name__ == "__main__":
    LABEL_PROMPTS = [
        "a photo of a person",
    ]
    grouped = tagging_and_grouping(
        "/home/zhihao/cs272project/Gradio/examples/photos",
        LABEL_PROMPTS,
        # save_csv="output.csv"
    )
    print("Grouped images by labels:", grouped)
