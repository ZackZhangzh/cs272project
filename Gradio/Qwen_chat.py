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


def chat_with_images(image_paths, messages):
    """
    image_paths: list of image file paths
    messages: list of dicts, each dict like {"role": "user"/"assistant", "content": [...]}
    Returns: 模型回复文本
    """
    base64_images = get_images_base64_list(image_paths)
    client = OpenAI(
        base_url="https://api-inference.modelscope.cn/v1/",
        api_key="14a623da-3226-41dc-bd0e-f0d439662565",
    )
    # 只在本轮 user 消消息中加图片
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        for base64_img in base64_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                }
            )
        messages[-1]["content"] = content

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
    image_paths = ["/home/zhihao/cs272project/data/output/Nvi.png"]
    messages = []
    print("进入对话模式，输入 exit 退出。")
    while True:
        prompt = input("你想问什么？> ")
        if prompt.strip().lower() == "exit":
            break
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        reply = chat_with_images(image_paths, messages)
        print("模型回复：", reply)
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": reply}]}
        )
