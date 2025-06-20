# CV2 project


pip install transformers=4.43
flask
## Qwen

## image_captioning

```bash
conda create -n tf15 python=3.7
conda activate tf15
conda install --channel https://conda.anaconda.org/hanyucui tensorflow-gpu=1.15

pip install pandas matplotlib tqdm


cd image_captioning
python main.py --phase=test \
--model_file='./models/289999.npy' \
--beam_size=3


```

## Deduplicator

```bash

cd imagededup
python setup.py install

python duplicates.py
```

## Try on
```bash

cd StableVITON
python inference.py  --config_path ./configs/VITONHD.yaml  --batch_size 2  --model_load_path ./ckpts/VITONHD.ckpt  --save_dir ./save --unpair
```

## Face Recognition and Grouping

```bash
from face_detection_grouping import cluster_faces
cluster_faces(
    "<path to images>",
    "<path to save clustering result>",
    face_crop_dir=<path to save cropped faces>,
    eps=0.2,
    min_samples=1
)
```

























## Face2Face
https://github.com/kimoktm/Face2face
```bash

bzip2 -d
xvfb-run -a
```

```bash
conda install scikit-learn=0.23



cd Face2face
xvfb-run -a  python cli/initialize.py --input_dir /home/zhihao/cs272project/data/input --output_dir /home/zhihao/cs272project/data/identity


python cli/tracker.py --input_dir path_to_tracking_images --output_dir path_to_save_tracking --parameters path_to_save_identity/params.npy
```

```bash
cd Face2FaceRHO

python src/fitting.py --device cuda \
--src_img /home/zhihao/cs272project/Gradio/examples/photos/cam_0001_0000.jpg \
--drv_img /home/zhihao/cs272project/Gradio/examples/photos/0003_0000.jpg \
--output_src_headpose /home/zhihao/cs272project/data/output/sh \
--output_src_landmark /home/zhihao/cs272project/data/output/sl \
--output_drv_headpose /home/zhihao/cs272project/data/output/dh \
--output_drv_landmark /home/zhihao/cs272project/data/output/dl



python src/reenact.py --config ./config/test_face2facerho.ini \
--src_img /home/zhihao/cs272project/Gradio/examples/photos/cam_0001_0000.jpg \
--src_headpose /home/zhihao/cs272project/data/output/sh \
--src_landmark /home/zhihao/cs272project/data/output/sl \
--drv_headpose /home/zhihao/cs272project/data/output/dh \
--drv_landmark /home/zhihao/cs272project/data/output/dl \
--output_dir /home/zhihao/cs272project/data/output
```
