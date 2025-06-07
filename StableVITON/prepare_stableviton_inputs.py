import itertools
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional
from tqdm import tqdm  
# import requests # Not strictly needed if images are local, but good for a general example

# --- Configuration ---
UPPER_BODY_LABELS = [4, 7] # For "Upper-clothes" and "Dress"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# DensePose Color Palette for Segment Mask (24 parts + 1 background)
# Index 0 is background (black). Indices 1-24 are parts.
DENSEPOSE_COLOR_PALETTE = np.array([ # R, G, B
    [0, 0, 0],        # 0 - Background
    [255, 128, 128],  # 1 - Torso Front
    [128, 255, 128],  # 2 - Torso Back
    [128, 128, 255],  # 3 - Right Hand
    [255, 255, 128],  # 4 - Left Hand
    [255, 128, 255],  # 5 - Left Foot
    [128, 255, 255],  # 6 - Right Foot
    [200, 0, 0],      # 7 - Right Upper Leg Front
    [0, 200, 0],      # 8 - Left Upper Leg Front
    [0, 0, 200],      # 9 - Right Upper Leg Back
    [200, 200, 0],    # 10 - Left Upper Leg Back
    [200, 0, 200],    # 11 - Right Lower Leg Front
    [0, 200, 200],    # 12 - Left Lower Leg Front
    [150, 150, 150],  # 13 - Right Lower Leg Back
    [100, 0, 0],      # 14 - Left Lower Leg Back
    [0, 100, 0],      # 15 - Left Upper Arm Front
    [0, 0, 100],      # 16 - Right Upper Arm Front
    [100, 100, 0],    # 17 - Left Upper Arm Back
    [100, 0, 100],    # 18 - Right Upper Arm Back
    [0, 100, 100],    # 19 - Left Lower Arm Front
    [220, 100, 100],  # 20 - Right Lower Arm Front
    [100, 220, 100],  # 21 - Left Lower Arm Back
    [100, 100, 220],  # 22 - Right Lower Arm Back
    [250, 190, 190],  # 23 - Head Front
    [190, 250, 190]   # 24 - Head Back
], dtype=np.uint8)

# --- Utility Functions ---
def load_image_pil(image_path):
    return Image.open(image_path).convert("RGB")

def save_image_np(np_array, path):
    # 1. 数据类型标准化
    if np_array.dtype != np.uint8:
        if np.issubdtype(np_array.dtype, np.floating):
            if np.max(np_array) <= 1.0 and np.min(np_array) >= 0.0:
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        else:
            np_array = np_array.astype(np.uint8)

    # 2. 调整尺寸到768x1024（保持宽高比强制拉伸）
    target_size = (768,1024)  # 注意OpenCV的尺寸格式是(width, height)
    resized_array = cv2.resize(np_array, target_size, interpolation=cv2.INTER_AREA)

    # 3. 保存处理
    if resized_array.ndim == 2:
        # 灰度图保存
        Image.fromarray(resized_array, mode='L').save(path)
    elif resized_array.ndim == 3:
        if resized_array.shape[2] == 1:
            # 单通道图保存
            Image.fromarray(resized_array.squeeze(), mode='L').save(path)
        elif resized_array.shape[2] == 3:
            # RGB彩色图保存（直接使用OpenCV的BGR顺序）
            # 如果需要RGB格式，可以添加转换：cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB)
            Image.fromarray(resized_array).save(path)
        else:
            raise ValueError(f"Unsupported channel count: {resized_array.shape[2]}")
    else:
        raise ValueError(f"Unsupported array shape: {resized_array.shape}")
# --- Model Wrapper Classes ---
class HumanParser:
    def __init__(self, model_name="matei-dorian/segformer-b5-finetuned-human-parsing"):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print(f"Loading Human Parser: {model_name}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("Human Parser loaded.")

    def get_parsing(self, image_pil: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
        )
        parsing_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return parsing_map

class ClothSegmenter:
    def __init__(self):
        try:
            from rembg import remove
            self.rembg_remove = remove
            print("Cloth Segmenter (rembg) initialized.")
        except ImportError:
            print("CRITICAL: 'rembg' not installed. Please install: pip install rembg[gpu]")
            self.rembg_remove = None
            
    def get_mask(self, image_pil: Image.Image) -> np.ndarray:
        if not self.rembg_remove:
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
        try:
            output_pil_with_alpha = self.rembg_remove(image_pil, alpha_matting=True, post_process_mask=True)
        except Exception as e:
            print(f"ClothSegmenter Error: {e}. Basic removal.")
            output_pil_with_alpha = self.rembg_remove(image_pil)
        if output_pil_with_alpha.mode == 'RGBA':
            alpha_mask_np = np.array(output_pil_with_alpha.split()[-1])
            _, binary_mask_np = cv2.threshold(alpha_mask_np, 128, 255, cv2.THRESH_BINARY)
            return binary_mask_np
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

class DensePosePredictor:
    def __init__(self,
                 model_config_path="densepose_rcnn_R_50_FPN_s1x.yaml",
                 model_weights_url="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
                 device="cuda:0"):
        self.predictor = None
        self.device = device
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config
        except ImportError as e:
            print(f"CRITICAL: Missing detectron2 or densepose package: {e}")
            return

        print("Loading DensePose Predictor...")
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(model_config_path)
        cfg.MODEL.WEIGHTS = model_weights_url
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        print("DensePose Predictor loaded.")

    def get_densepose_colored_segment_mask(self, image_pil: Image.Image) -> np.ndarray:
        """
        Returns a colored DensePose segmentation mask aligned with the input image.

        Args:
            image_pil: Input PIL image.

        Returns:
            mask_np: np.ndarray of shape (H, W, 3), colored DensePose part segmentation.
        """
        # Step 1: Run predictor
        
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        outputs = self.predictor(image_cv2)
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_densepose"):
            raise ValueError("No DensePose output found. Make sure you are using a DensePose model.")

        # Prepare blank canvas
        dp_out = instances.pred_densepose[0]
        fine_segm = dp_out.fine_segm[0]  # torch.Tensor
        
        probs = torch.nn.functional.softmax(fine_segm, dim=0)  # shape: [25, H, W]
        max_probs, final_mask = probs.max(0)
        final_mask = final_mask.cpu().numpy()
        max_probs = max_probs.cpu().numpy()

        # 设置置信度阈值，小于 threshold 的点视为背景
        final_mask[max_probs < 0.6] = 0
        final_mask = final_mask.astype(np.uint8)

        
        # 6. 计算每个像素所属 body part：取 argmax over K channels
        # final_mask = fine_segm.argmax(0).cpu().numpy().astype(np.uint8)  # shape: (H, W)
        final_mask_resized = cv2.resize(
            final_mask,
            (image_pil.width, image_pil.height),  # 注意 cv2.resize的尺寸是 (宽, 高)
            interpolation=cv2.INTER_NEAREST
        )
        mask_rgb = DENSEPOSE_COLOR_PALETTE[final_mask_resized]
        # Convert part index map to color
        # mask_rgb = np.transpose(mask_rgb, (1, 0, 2))  # width->height transpose
        return mask_rgb

# --- Core Logic Functions ---
def generate_agnostic(image_pil: Image.Image, parsing_map: np.ndarray, upper_body_labels: list) -> tuple[np.ndarray, np.ndarray]:
    np_img = np.array(image_pil)
    agnostic = np_img.copy()
    mask_upper_body = np.isin(parsing_map, upper_body_labels)
    agnostic[mask_upper_body] = [127, 127, 127]
    agnostic_binary_mask = (mask_upper_body).astype(np.uint8) * 255
    return agnostic, agnostic_binary_mask
# --- 新增：独立图片处理函数 ---
def process_person_image(person_path: str, output_dir: str,
                        human_parser: HumanParser = None,
                        densepose_predictor: DensePosePredictor = None):
    """处理单张人物图片"""
    try:
        # 创建独立输出目录
        output_subdir = os.path.join(output_dir, "person_results", os.path.basename(person_path))
        os.makedirs(output_subdir, exist_ok=True)
 
        # 加载图片
        image_pil = load_image_pil(person_path)
 
        # 人类解析
        if human_parser:
            parsing_map = human_parser.get_parsing(image_pil)
            save_image_np(parsing_map, os.path.join(output_subdir, "image-parse-map.png"))
 
        # 生成无性别图像
        if 'parsing_map' in locals():
            agnostic_np, agnostic_mask_np = generate_agnostic(image_pil, parsing_map, UPPER_BODY_LABELS)
        else:
            agnostic_np = np.array(image_pil)
            agnostic_mask_np = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
 
        save_image_np(agnostic_np, os.path.join(output_subdir, "agnostic-image.png"))
        save_image_np(agnostic_mask_np, os.path.join(output_subdir, "agnostic-mask.png"))
 
        # DensePose预测
        if densepose_predictor:
            densepose_mask_np = densepose_predictor.get_densepose_colored_segment_mask(image_pil)
            save_image_np(densepose_mask_np, os.path.join(output_subdir, "image-densepose.png"))
 
        # 保存原始图片
        image_pil = image_pil.resize((768,1024))
        image_pil.save(os.path.join(output_subdir, "original-person.png"))
 
        return True, f"Processed person image: {person_path}"
 
    except Exception as e:
        return False, f"Error processing person {person_path}: {str(e)}"
 
def process_cloth_image(cloth_path: str, output_dir: str,
                        cloth_segmenter: ClothSegmenter = None):
    """处理单张服装图片"""
    try:
        # 创建独立输出目录
        output_subdir = os.path.join(output_dir, "cloth_results", os.path.basename(cloth_path))
        os.makedirs(output_subdir, exist_ok=True)
 
        # 加载图片
        cloth_pil = load_image_pil(cloth_path)
 
        # 服装分割
        if cloth_segmenter:
            cloth_mask_np = cloth_segmenter.get_mask(cloth_pil)
            save_image_np(cloth_mask_np, os.path.join(output_subdir, "cloth-mask.png"))
 
        # 保存原始图片
        cloth_pil = cloth_pil.resize((768,1024))
        cloth_pil.save(os.path.join(output_subdir, "original-cloth.png"))
 
        return True, f"Processed cloth image: {cloth_path}"
 
    except Exception as e:
        return False, f"Error processing cloth {cloth_path}: {str(e)}"
 
# --- 修改后的目录处理函数 ---
def process_directories(person_dir: str, cloth_dir: str, output_base_dir: str):
    """独立处理人物和服装目录"""
    # 验证输入目录
    if not os.path.isdir(person_dir):
        raise ValueError(f"Person directory not found: {person_dir}")
    if not os.path.isdir(cloth_dir):
        raise ValueError(f"Cloth directory not found: {cloth_dir}")
 
    # 获取所有图片文件路径
    person_files = sorted([
        os.path.join(person_dir, f) for f in os.listdir(person_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    cloth_files = sorted([
        os.path.join(cloth_dir, f) for f in os.listdir(cloth_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
 
    # 初始化模型（保持原有逻辑）
    print("\nInitializing models (this might take a while)...")
    human_parser_model = HumanParser()
    cloth_segmenter_model = ClothSegmenter()
    densepose_predictor_model = DensePosePredictor()
 
    # 检查模型初始化
    model_status = []
    if human_parser_model.model is None:
        model_status.append("Human Parser")
    if cloth_segmenter_model.rembg_remove is None:
        model_status.append("Cloth Segmenter")
    if densepose_predictor_model.predictor is None:
        model_status.append("DensePose Predictor")
    if model_status:
        print(f"WARNING: Some models failed to initialize: {', '.join(model_status)}")
 
    # 处理人物图片
    print(f"\nProcessing {len(person_files)} person images...")
    with tqdm(total=len(person_files), desc="Processing persons") as pbar:
        for person_path in person_files:
            pbar.set_postfix({"Person": os.path.basename(person_path)})
            success, _ = process_person_image(
                person_path=person_path,
                output_dir=output_base_dir,
                human_parser=human_parser_model,
                densepose_predictor=densepose_predictor_model
            )
            pbar.update(1 if success else 0)
 
    # 处理服装图片
    print(f"\nProcessing {len(cloth_files)} cloth images...")
    with tqdm(total=len(cloth_files), desc="Processing clothes") as pbar:
        for cloth_path in cloth_files:
            pbar.set_postfix({"Cloth": os.path.basename(cloth_path)})
            success, _ = process_cloth_image(
                cloth_path=cloth_path,
                output_dir=output_base_dir,
                cloth_segmenter=cloth_segmenter_model
            )
            pbar.update(1 if success else 0)
 
    print("\n--- Batch Processing Completed ---")
    print(f"All results saved to: {os.path.abspath(output_base_dir)}")
 
# --- 主程序入口 ---
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
    # 配置输入输出路径
    INPUT_PERSON_DIR = "dataset/person"  # 修改为你的实际路径
    INPUT_CLOTH_DIR = "dataset/clothes"    # 修改为你的实际路径
    OUTPUT_BASE_DIR = "output_viton_colored_dp"  # 修改为你的输出根目录
 
    # 运行批量处理
    try:
        process_directories(
            person_dir=INPUT_PERSON_DIR,
            cloth_dir=INPUT_CLOTH_DIR,
            output_base_dir=OUTPUT_BASE_DIR
        )
    except Exception as e:
        print(f"\n❌ Critical error occurred: {str(e)}")