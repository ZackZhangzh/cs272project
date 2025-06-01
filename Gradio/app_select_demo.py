import gradio as gr
import os
import glob
import hashlib


global SELECT_IMAGE_PATH
SELECT_IMAGE_PATH = ""  # Initialize global variable to store selected image path


def load_image():
    path = "examples/photos"
    image_paths = glob.glob(os.path.join(path, "*.png")) + glob.glob(
        os.path.join(path, "*.jpg")
    )
    return image_paths


def select_image(evt: gr.SelectData):
    """Handle image selection"""
    print(evt.value)
    global SELECT_IMAGE_PATH
    SELECT_IMAGE_PATH = evt.value["image"]["path"]  # Get the path of the selected image
    return evt.value["image"]["path"]  # Return selected image path


# Launch Gradio interface
with gr.Blocks(title="智能相册", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📷 智能相册应用")
    gr.Markdown("相册展示，选择，上传，网络处理demo")

    with gr.Row():
        with gr.Column(scale=5):
            gallery = gr.Gallery(
                label="照片墙", columns=10, object_fit="cover", value=load_image()
            )

            # ratio = gr.Radio(
            #     label="图片比例",
            #     choices=["src", "dest"],
            #     value="src",
            #     type="value"
            # )

        with gr.Column(scale=2):
            selected_image = gr.Image(label="选中的图片1", interactive=False)
            process_btn = gr.Button("处理选中图片")
            selected_image2 = gr.Image(label="处理结果示意", interactive=False)

            # Track last selected image index
            selected_images = []

            def handle_selection(evt: gr.SelectData):
                path = select_image(evt)
                selected_images.append(path)
                return path, path

        with gr.Column(scale=2):
            upload = gr.Image(
                label="上传图片", type="pil"
            )  # 使用 gr.Image 上传图像，类型为 PIL 图像
            upload_txt = gr.Textbox(label="上传结果")  # 提示文本框
            caption_txt = gr.Textbox(label="图像总结")  # 提示文本框

            from PIL import Image, ImageDraw
            import numpy as np

            def handle_imageprocess(image_array):
                """使用 PIL 给 numpy 格式的图像数组加上遮罩"""
                # 将 numpy 数组转换为 PIL 图像
                image = Image.fromarray(image_array).convert("RGB")

                # 创建一个与图像大小相同的遮罩
                mask = Image.new("L", image.size, 0)  # 'L' 模式表示单通道灰度图像
                draw = ImageDraw.Draw(mask)

                # 在遮罩上绘制一个圆形
                width, height = image.size
                radius = min(width, height) // 3  # 圆的半径
                center = (width // 2, height // 2)  # 圆的中心
                draw.ellipse(
                    (
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    fill=255,
                )

                # 将遮罩应用到图像上
                masked_image = Image.composite(
                    image, Image.new("RGB", image.size, (0, 0, 0)), mask
                )

                # 如果需要返回 numpy 数组格式的结果
                return np.array(masked_image)

            def handle_network_imageprocess(image_array):
                """Placeholder function for network image processing"""
                # send http request to server with image_array
                import requests
                import json
                import base64
                from io import BytesIO

                # Convert the image array to PIL Image
                image = Image.fromarray(image_array).convert("RGB")
                # Convert the PIL Image to base64 string
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # NOTE Send the image to the server using request.files['image'] format
                response = requests.post(
                    "http://localhost:9090/process_image",  # Replace with your server URL
                    files={"image": buffered.getvalue()},
                )

                if response.status_code == 200:
                    # Assuming the server returns a processed image in base64 format
                    response_data = response.json()
                    img_data = response_data["image_data"]
                    img_bytes = base64.b64decode(img_data)
                    image_array = np.array(Image.open(BytesIO(img_bytes)))
                    return image_array, "caption test"
                return None

            # Event binding (dynamic based on Radio)

            process_btn.click(
                fn=handle_network_imageprocess,  # Placeholder function for processing
                inputs=selected_image,
                outputs=[selected_image2, caption_txt],
            )

        # NOTE UPLOAD FUNCTIONALITY
        # 定义保存目录
        SAVE_DIR = "./examples/photos"
        os.makedirs(SAVE_DIR, exist_ok=True)  # 如果目录不存在，则创建

        def save_uploaded_image(image):
            """保存上传的图像到指定目录"""
            # 随机生成64位哈希字符串
            image_name = hashlib.md5(os.urandom(16)).hexdigest()[:32] + ".png"
            file_path = os.path.join(SAVE_DIR, image_name)  # 固定保存为一个文件名
            image.save(file_path)  # 使用 PIL 的 save 方法保存图像
            return file_path  # 返回保存的文件路径

        def update_gallery():
            """更新图像库"""
            # 获取保存目录中的所有图像文件路径
            image_files = [
                os.path.join(SAVE_DIR, f)
                for f in os.listdir(SAVE_DIR)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            return image_files  # 返回图像文件列表

        # 增加上传组件

        def handle_upload(image):
            """处理上传的图像并保存，同时更新图像库"""
            saved_path = save_uploaded_image(image)  # 保存上传的图像
            updated_gallery = update_gallery()  # 更新图像库
            return (
                f"图像已保存到: {saved_path}",
                updated_gallery,
            )  # 返回文本和更新后的图像列表

        upload.upload(
            fn=handle_upload,
            inputs=upload,
            outputs=[upload_txt, gallery],  # 文本输出  # 图像库输出
        )

    gr.Markdown("# 📷 自动查重 demo")
    gr.Markdown("对库内所有图像进行")
    with gr.Row():
        dedup_btn = gr.Button("开始查重")
    with gr.Row():
        dup_result = gr.Gallery(
            label="查重结果",
            columns=5,
            height=200,
            object_fit="cover",
            interactive=False,
        )

    # NOTE  TAgging Grouping
    gr.Markdown("# 📷 聚类 分类 demo")
    gr.Markdown("图像检索")
    with gr.Row():
        with gr.Column(scale=5):
            txxt_input = gr.Textbox(label="输入文本", placeholder="Tag to search")
        with gr.Column(scale=2):
            txt_btn = gr.Button("查询")
    with gr.Row():
        gallery_output = gr.Gallery(
            label="查询结果",
            columns=5,
            height=200,
            object_fit="cover",
            interactive=False,
        )

    gr.Markdown("人脸自动聚类")
    with gr.Row():
        with gr.Column(scale=2):
            figure_image = gr.Image(label="待聚类图像", type="pil")
            cluster_btn = gr.Button("获取人脸聚类")
        with gr.Column(scale=5):
            cluster_gallery = gr.Gallery(
                label="人脸聚类结果",
                columns=5,
                height=200,
                object_fit="cover",
                interactive=False,
            )

    gallery.select(handle_selection, None, [selected_image, figure_image])

    gr.Markdown("# 📷 VLLM")
    import random
    from Qwen_chat import chat_with_images
    import json

    with gr.Row():

        def random_resp(message, history):
            print("=======")
            print(SELECT_IMAGE_PATH)

            messages.append(
                {"role": "user", "content": [{"type": "text", "text": message}]}
            )
            reply = chat_with_images([SELECT_IMAGE_PATH], messages)
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": reply}]}
            )
            print("=======")
            # print(json.dumps(messages, ensure_ascii=False, indent=2))
            return reply

        gr.ChatInterface(random_resp)


if __name__ == "__main__":
    messages = []
    app.launch(server_port=8081, share=False)
