import mxnet as mx
from mxnet import gluon
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

import os
import math

def load_voc(batch_size, input_size=416):
    width, height = input_size, input_size
    val_dataset = gdata.VOCDetection(splits=[('2007', 'test')])
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)
    return val_loader

def load_voc_metric():
    return VOC07MApMetric(iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)


def load_imagenet(batch_size):
    val_dataset = ImageNet(train=False)
    val_loader = gluon.data.DataLoader(
        val_dataset,
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)

def load_imagenet_rec(batch_size, input_size=224):
    rec_val = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.rec")
    rec_val_idx = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.idx")
    crop_ratio = 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    val_data = mx.io.ImageRecordIter(
	path_imgrec         = rec_val,
	path_imgidx         = rec_val_idx,
	preprocess_threads  = 24,
	shuffle             = False,
	batch_size          = batch_size,

	resize              = resize,
	data_shape          = (3, input_size, input_size),
	mean_r              = mean_rgb[0],
	mean_g              = mean_rgb[1],
	mean_b              = mean_rgb[2],
	std_r               = std_rgb[0],
	std_g               = std_rgb[1],
	std_b               = std_rgb[2],
    )
    return val_data

