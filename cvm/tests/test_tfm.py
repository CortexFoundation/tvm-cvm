import mxnet as mx

import utils
from transformer import validate_model

from os import path

def test_mobilenet1_0():
    sym_path = "./data/mobilenet1_0.json"
    prm_path = "./data/mobilenet1_0.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_mobilenet1_0()
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx=ctx)

def test_mobilenet_v2_1_0():
    sym_path = "./data/mobilenetv2_1.0.json"
    prm_path = "./data/mobilenetv2_1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx=ctx)

def test_tf_inceptionv3():
    sym_path = "./data/tf_inceptionv3.json"
    prm_path = "./data/tf_inceptionv3.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size=299, ctx=ctx)

def test_alexnet():
    sym_path = "./data/alexnet.json"
    prm_path = "./data/alexnet.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, batch_size=700, ctx=ctx)

def test_cifar10_resnet20_v1():
    sym_path = "./data/cifar_resnet20_v1.json"
    prm_path = "./data/cifar_resnet20_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size=32, ctx=ctx)

def test_resnet18_v1():
    sym_path = "./data/resnet18_v1.json"
    prm_path = "./data/resnet18_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, iter_num=100, ctx=ctx)

def test_yolo3_darknet53_voc():
    sym_path = "./data/yolo3_darknet53_voc.json"
    prm_path = "./data/yolo3_darknet53_voc.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size=416,
            batch_size=1, iter_num=100, ctx=ctx)

if __name__ == '__main__':
    utils.log_init()

    test_mobilenet1_0() # 81-71%
    test_mobilenet_v2_1_0() # quantized_model zero precision
    test_tf_inceptionv3() # 80-56%
    test_alexnet() # 56%
    test_cifar10_resnet20_v1() # org_model/quantized_model zero precision
    test_resnet18_v1() # 87-70%
    # test_yolo3_darknet53_voc()


