import os
import shutil
import face_recognition
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm

def cluster_faces(image_dir, output_dir, face_crop_dir=None, eps=0.2, min_samples=2):
    """
    image_dir: 原始照片目录
    output_dir: 聚类后分类存放照片的目录
    face_crop_dir: 可选，保存crop下的人脸图片用于调试
    eps: DBSCAN的距离阈值，调节聚类“严格程度”
    min_samples: 同一聚类最小人脸数
    """

    os.makedirs(output_dir, exist_ok=True)
    if face_crop_dir:
        os.makedirs(face_crop_dir, exist_ok=True)

    encodings = []
    face_infos = []  # 保存图片路径和face位置信息

    print("[INFO] Step 1: 遍历图片并提取所有人脸特征...")

    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image)
        faces = face_recognition.face_encodings(image, locations)

        for loc, encoding in zip(locations, faces):
            encodings.append(encoding)
            face_infos.append((img_path, loc))

            # 可选保存人脸crop
            if face_crop_dir:
                y1, x2, y2, x1 = loc
                face_img = Image.fromarray(image[y1:y2, x1:x2])
                face_img.save(os.path.join(face_crop_dir, f"{len(encodings)}.jpg"))

    print(f"[INFO] Step 2: 聚类 {len(encodings)} 张人脸...")

    # Step 2: DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(encodings)
    labels = clustering.labels_

    print(f"[INFO] 得到 {len(set(labels)) - (1 if -1 in labels else 0)} 个人")

    # Step 3: 将照片按聚类标签保存
    for i, (label, (img_path, face_loc)) in enumerate(zip(labels, face_infos)):
        if label == -1:
            continue  # 忽略噪声点

        person_dir = os.path.join(output_dir, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True)

        # 拷贝原图进去（可避免重复拷贝）
        dst_img_path = os.path.join(person_dir, os.path.basename(img_path))
        if not os.path.exists(dst_img_path):
            shutil.copy(img_path, dst_img_path)

    print("[INFO] 聚类完成！结果保存在：", output_dir)

