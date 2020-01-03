import mxnet as mx

import utils
from transformer import validate_model
from gluon_zoo import save_mobilenet1_0
from from_tensorflow import tf_dump_model

from os import path

def test_tf_resnet50_v1():
    sym_path = "./data/tf_resnet50_v1.json"
    prm_path = "./data/tf_resnet50_v1.params"
    # if not path.exists(sym_path) or not path.exists(prm_path):
    if True:
        tf_dump_model("resnet50_v1")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_tf_mobilenet():
    sym_path = "./data/tf_mobilenet.json"
    prm_path = "./data/tf_mobilenet.params"
    # if not path.exists(sym_path) or not path.exists(prm_path):
    if True:
        tf_dump_model("mobilenet")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_mobilenet1_0():
    sym_path = "./data/mobilenet1_0.json"
    prm_path = "./data/mobilenet1_0.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_mobilenet1_0()
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, iter_num=999999)
    validate_model(sym_path, prm_path, ctx)

def test_mobilenet_v2_1_0():
    sym_path = "./data/mobilenetv2_1.0.json"
    prm_path = "./data/mobilenetv2_1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_tf_inceptionv3():
    sym_path = "./data/tf_inception_v3.json"
    prm_path = "./data/tf_inception_v3.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        tf_dump_model("inception_v3")
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
    validate_model(sym_path, prm_path, ctx, input_size=32, ds_name='cifar10')

def test_resnet(suffix):
    sym_path = "./data/resnet" + suffix + ".json"
    prm_path = "./data/resnet" + suffix + ".params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, iter_num=100, lambd=16)

def test_densenet161():
    sym_path = "./data/densenet161.json"
    prm_path = "./data/densenet161.params"
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, batch_size=16)

def test_qd10_resnetv1_20():
    sym_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.json"
    prm_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, num_channel=1,
            input_size=28, ds_name='quickdraw')

def test_shufflenet_v1():
    sym_path = "./data/shufflenet_v1.json"
    prm_path = "./data/shufflenet_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, iter_num=100)

def test_squeezenet():
    sym_path = "./data/squeezenet1.0.json"
    prm_path = "./data/squeezenet1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, batch_size=60, iter_num=100)

def test_vgg():
    sym_path = "./data/vgg19.json"
    prm_path = "./data/vgg19.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

if __name__ == '__main__':
    utils.log_init()

    # test_mobilenet1_0()
    # 200102 AM; 70.76% --> 63.18%; Iteration 3123; Total Sample 49984  

    # test_mobilenet_v2_1_0()       # 73% --> 0%
    # test_tf_inceptionv3()         # 56% --> 56%
    # test_alexnet()                # 56% --> 55%
    # test_cifar10_resnet20_v1()    # 91% --> 90%
    test_resnet("50_v1")          # 78% --> 76%
    # test_resnet("18_v1")          # 70% --> %
    # test_resnet("50_v1d_0.86")    # not valid: Pooling count_include_pad:True
    # test_resnet("18_v1b_0.89")    # 68% --> 65%
    # test_resnet("50_v2")          # 77% --> 74%
    # test_densenet161()            # 81% --> 77%
    # test_qd10_resnetv1_20()       # 83% --> 83%
    # test_shufflenet_v1()          # 64% --> 61%
    # test_squeezenet()             # 57% --> 55%
    # test_vgg()                    # 78% --> 78%

    # TODO: test
    # test_tf_mobilenet()           # 0% --> 0%, maybe due to pad
    # test_tf_resnet50_v1()         # 0% --> 0%



