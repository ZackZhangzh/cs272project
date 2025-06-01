# CV2 project
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
--src_img /home/zhihao/cs272project/data/input/cam_0001_0000.jpg \
--drv_img /home/zhihao/cs272project/data/output/Nvi.png \
--output_src_headpose /home/zhihao/cs272project/data/output/h \
--output_src_landmark /home/zhihao/cs272project/data/output/h \
--output_drv_headpose /home/zhihao/cs272project/data/output/h \
--output_drv_landmark /home/zhihao/cs272project/data/output/h
```
