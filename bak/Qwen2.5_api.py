from openai import OpenAI
from PIL import Image
import io
import base64


def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format="webp")
            base64_str = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None


def get_images_base64_list(image_paths):
    base64_list = []
    for path in image_paths:
        base64_img = convert_image_to_webp_base64(path)
        if base64_img:
            base64_list.append(base64_img)
        else:
            print(f"警告：图片 {path} 处理失败，将跳过。")
    return base64_list


def qwen25_chat(message, image_paths, history=None):
    """
    message: str, 用户输入文本
    image_paths: str 或 list[str]，图片路径或路径列表
    history: list, 历史对话（每项为dict: {"role":..., "content":[...]}），可为空
    返回: reply 文本
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    base64_images = get_images_base64_list(image_paths)
    # 构建本轮 user 消息
    content = [{"type": "text", "text": message}]
    for base64_img in base64_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
            }
        )
    user_msg = {"role": "user", "content": content}
    # 合成完整消息历史
    messages = history.copy() if history else []
    messages.append(user_msg)
    client = OpenAI(
        base_url="https://api-inference.modelscope.cn/v1/",
        api_key="14a623da-3226-41dc-bd0e-f0d439662565",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=messages,
        stream=True,
    )
    reply = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        reply += delta
    return reply


# 用法示例（可删除）
if __name__ == "__main__":
    msg = input("请输入问题：")
    img = input("请输入图片路径：")
    print(qwen25_chat(msg, img, history=[]))
