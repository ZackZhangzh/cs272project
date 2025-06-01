from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps
import base64
import io
import os
from PIL import ImageDraw


app = Flask(__name__)


def handle_imageprocess(image_array):
    """使用 PIL 给 numpy 格式的图像数组加上遮罩"""
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(image_array).convert('RGB')

    # 创建一个与图像大小相同的遮罩
    mask = Image.new('L', image.size, 0)  # 'L' 模式表示单通道灰度图像
    draw = ImageDraw.Draw(mask)

    # 在遮罩上绘制一个圆形
    width, height = image.size
    radius = min(width, height) // 3  # 圆的半径
    center = (width // 2, height // 2)  # 圆的中心
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

    # 将遮罩应用到图像上
    masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)

    # 如果需要返回 numpy 数组格式的结果
    return np.array(masked_image)



@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    

    # image processing code goes here...
    img_masked = handle_imageprocess(img)

    # Convert the processed image back to bytes and base64 encode it
    img_masked = Image.fromarray(img_masked)

    # save to bytes
    img_bytes = io.BytesIO()
    img_masked.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Return the processed image as a base64 string
    return jsonify({'message': 'Image processed successfully', 'image_data': base64.b64encode(img_bytes).decode('utf-8')})

if __name__ == '__main__':
    app.run(port=9090)
