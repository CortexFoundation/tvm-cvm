import mxnet as mx
from mxnet import gluon
from mxnet import nd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
import numpy as np
import requests
import shutil

import os
import math
import pickle

# max value: 2.64
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


def download_file(filename, src_path = "http://192.168.50.210:8827"):
    if os.path.exists(filename):
        return
    filedir, _ = os.path.split(filename);
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if "imagenet" in filename:
        if "val.rec" in filename:
            r = requests.get(src_path + "/imagenet/val.rec")
        elif "val.idx" in filename: 
            r = requests.get(src_path + "/imagenet/val.idx")
    elif "trec" in filename:
        if "TREC.test.pk" in filename:
            r = requests.get(src_path + "/trec/TREC.test.pk")
        if "TREC.train.pk" in filename:
            r = requests.get(src_path + "/trec/TREC.train.pk")
    else:
        return
    r.raise_for_status()
    nf = open(filename, "wb")
    for c in r.iter_content(10000):
        nf.write(c)
    nf.close()


def load_imagenet_rec(batch_size, input_size=224): 
    # rec_val = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.rec")
    rec_val = "./dataset/imagenet/rec/val.rec"
    download_file(rec_val)
    # rec_val_idx = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.idx")
    rec_val_idx = "./dataset/imagenet/rec/val.idx"
    download_file(rec_val_idx)
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

import pickle

def load_cifar10(batch_size, input_size=224, num_workers=4):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            yield data[0], label[0]
    return data_iter()

def load_quickdraw10(batch_size, num_workers=4):
    X = nd.array(np.load('/home/serving/cortex_ml_data/quickdraw_X_test.npy'))
    y = nd.array(np.load('/home/serving/cortex_ml_data/quickdraw_y_test.npy'))
    val_data = gluon.data.DataLoader(
             mx.gluon.data.dataset.ArrayDataset(X, y),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            data, label = data[0], label[0]
            yield data, label
    return data_iter()

def load_trec(batch_size, is_train = False):
    if is_train:
        # fname = "./data/TREC/TREC.train.pk"
        fname = os.path.expanduser("./dataset/trec/TREC.train.pk")
    else:
        # fname = "./data/TREC/TREC.test.pk"
        fname = os.path.expanduser("./dataset/trec/TREC.test.pk")
    download_file(fname) 
    dataset = pickle.load(open(fname, "rb"))
    data, label = [], []
    for x, y in dataset:
        if len(data) < batch_size:
            data.append(x)
            label.append(y)
        else:
            yield nd.transpose(nd.array(data)), nd.transpose(nd.array(label))
            data, label = [], []
