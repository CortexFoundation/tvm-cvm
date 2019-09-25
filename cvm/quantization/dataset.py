import mxnet as mx
from mxnet import gluon
from mxnet import nd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
import numpy as np
import requests
import tarfile

import os
import math
import pickle
import logging

dataset_dir = os.path.expanduser("~/.cvm")
src = "http://192.168.50.210:8827"

def extract_file(tar_path, target_path):
    if os.path.exists(target_path):
        return
    tar = tarfile.open(tar_path, "r")
    tar.extractall(target_path)
    tar.close()

def download_files(category, files, baseUrl=src, root=dataset_dir):
    logger = logging.getLogger("dataset")
    root_dir = os.path.join(root, category)
    os.makedirs(root_dir, exist_ok=True)

    for df in files:
        url = os.path.join(baseUrl, category, df)
        fpath = os.path.join(root_dir, df)
        if os.path.exists(fpath):
            continue
        fdir = os.path.dirname(fpath)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        logger.info("Downloading dateset %s into %s from url[%s]",
                df, root_dir, url)
        r = requests.get(url)
        if r.status_code != 200:
            logger.error("Url response invalid status code: %s",
                    r.status_code)
            exit()
        r.raise_for_status()
        with open(fpath, "wb") as fout:
            fout.write(r.content)
    return root_dir

# max value: 2.64
def load_voc(batch_size, input_size=416, **kwargs):
    fname = "VOCtest_06-Nov-2007.tar"
    root_dir = download_files("voc", [fname], **kwargs)
    extract_file(os.path.join(root_dir, fname), root_dir)
    width, height = input_size, input_size
    val_dataset = gdata.VOCDetection(root=os.path.join(root_dir,
                                                       'VOCdevkit'),
                                     splits=[('2007', 'test')])
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

def load_imagenet_rec(batch_size, input_size=224, **kwargs):
    files = ["rec/val.rec", "rec/val.idx"]
    root_dir = download_files("imagenet", files, **kwargs)
    crop_ratio = 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    rec_val = os.path.join(root_dir, files[0])
    rec_val_idx = os.path.join(root_dir, files[1])

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

def load_cifar10(batch_size, input_size=224, num_workers=4, **kwargs):
    flist = ["cifar-10-binary.tar.gz"]
    root_dir = download_files("cifar10", flist, **kwargs)
    extract_file(os.path.join(root_dir, flist[0]), root_dir)
    transform_test = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor(),
        gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                               [0.2023, 0.1994, 0.2010])])
    val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root=root_dir,
                train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            yield data[0], label[0]
    return data_iter()

def load_quickdraw10(batch_size, num_workers=4, **kwargs):
    files = ["quickdraw_X_test.npy", "quickdraw_y_test.npy"]
    root_dir = download_files("quickdraw", files, **kwargs)
    X = nd.array(np.load(os.path.join(root_dir, files[0])))
    y = nd.array(np.load(os.path.join(root_dir, files[1])))
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

def load_trec(batch_size, is_train = False, **kwargs):
    #  if is_train:
        #  fname = dataset_dir + "/trec/TREC.train.pk"
    #  else:
        #  fname = dataset_dir + "/trec/TREC.test.pk"
    files = ["TREC.train.pk", "TREC.test.pk"]
    root_dir = download_files("trec", files, **kwargs)
    fname = os.path.join(root_dir, files[0] if is_train else files[1])
    #  download_file(fname, dataset_dir=dataset_dir)
    with open(fname, "rb") as fin:
        dataset = pickle.load(fin)
        data, label = [], []
        for x, y in dataset:
            if len(data) < batch_size:
                data.append(x)
                label.append(y)
            else:
                yield nd.transpose(nd.array(data)), nd.transpose(nd.array(label))
                data, label = [], []

def load_mnist(batch_size, **kwargs):
    flist = ["t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz"]
    root_dir = download_files("mnist", flist, **kwargs)
    val_data = mx.gluon.data.vision.MNIST(root=root_dir, train=False).transform_first(data_xform)
    val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
    return val_loader

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255



