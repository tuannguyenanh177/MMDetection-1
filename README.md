# MMDetection

## Install
pip install -r requirements.txt -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html
example: pip install -r requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

## Dataset
python3 data/download_dataset.py --dataset-name coco2017 --save-dir data/coco --unzip --delete

## Train
### Double Head
python train/train.py /content/MMDetection/configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py \
&emsp;&emsp;&emsp;&emsp;--work-dir train/double_head_train
### Mask RCNN
python train/train.py /content/MMDetection/configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py \
&emsp;&emsp;&emsp;&emsp;--work-dir train/mask_rcnn_train
                      
## Evaluation
