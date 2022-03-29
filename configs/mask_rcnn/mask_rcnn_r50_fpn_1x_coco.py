_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=1000)

checkpoint_config = dict(interval=100)

data = dict(
    samples_per_gpu=3)

optimizer_config = dict(
 _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
