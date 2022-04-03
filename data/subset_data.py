import json
from tqdm import tqdm
import shutil
from pathlib import Path

final_path = 'coco2017_subset/'
src_parent = 'train2017/'
dst_parent = final_path + 'train2017/'
anno_parent = final_path + 'annotations/'

shutil.rmtree(final_path, ignore_errors=True)
Path(dst_parent).mkdir(parents=True, exist_ok=True)
Path(anno_parent).mkdir(parents=True, exist_ok=True)

# 'person': 1, 'car': 3, 'cat': 17, 'bottle': 44, 'chair': 62
id_cls = [3, 17, 44, 62]

# Download full coco2017 (images+annotations) first
with open("annotations/instances_train2017.json", "r") as f:
  data = json.load(f)

  annotations = []
  images = []
  img_id = []

  dict_images = {img['id']:img for img in data['images']}

  for anno in tqdm(data['annotations']):
    if anno['category_id'] in id_cls:
      annotations.append(anno)
      
      if anno['image_id'] not in img_id:
        img_id.append(anno['image_id'])
        images.append(dict_images[anno['image_id']])
        shutil.copyfile(src_parent + dict_images[anno['image_id']]['file_name'], dst_parent + dict_images[anno['image_id']]['file_name'])
  data['images'] = images
  data['annotations'] = annotations

  with open(anno_parent + 'instances_train2017.json', 'w') as f:
    json.dump(data, f)