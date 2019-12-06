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
    validate_model(sym_path, prm_path, ctx)

def test_mobilenet_v2_1_0():
    sym_path = "./data/mobilenetv2_1.0.json"
    prm_path = "./data/mobilenetv2_1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_tf_inceptionv3():
    sym_path = "./data/tf_inceptionv3.json"
    prm_path = "./data/tf_inceptionv3.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, input_size=299)

def test_alexnet():
    sym_path = "./data/alexnet.json"
    prm_path = "./data/alexnet.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, batch_size=700, ctx=ctx)

def test_cifar10_resnet20_v1():
    sym_path = "./data/cifar_resnet20_v1.json"
    prm_path = "./data/cifar_resnet20_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, input_size=32)

def test_resnet18_v1():
    sym_path = "./data/resnet18_v1.json"
    prm_path = "./data/resnet18_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, iter_num=100, from_scratch=0)

def test_densenet161():
    sym_path = "./data/densenet161.json"
    prm_path = "./data/densenet161.params"
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, batch_size=350, from_scratch=2)

def test_qd10_resnetv1_20():

    sym_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.json"
    prm_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, num_channel=1, input_size=28)


if __name__ == '__main__':
    utils.log_init()

    # test_mobilenet1_0() # 71% --> 63%
    # test_mobilenet_v2_1_0() # 73% --> 0%
    # test_tf_inceptionv3() # 56% --> 55%
    # test_alexnet() # 56% --> 55%
    # test_cifar10_resnet20_v1() # 0% --> 0%
    # test_resnet18_v1() # 70% --> 69%

    # TODO: improve precision for densenet
    # test_densenet161() # 77% --> 0%

    test_qd10_resnetv1_20()


