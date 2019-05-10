from mxnet import gluon
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

def load_voc(batch_size, input_size=416):
    width, height = input_size, input_size
    val_dataset = gdata.VOCDetection(splits=[('2007', 'test')])
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)
    return val_loader, val_metric

