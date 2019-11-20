import os
import sim_quant_helper as sim

import utils
import logging
import mxnet as mx
from transformer import validate_model
from gluon_zoo import save_model

def test_mobilenet1_0():
    sym_path = "./data/mobilenet1_0.json"
    prm_path = "./data/mobilenet1_0.params"
    if not os.path.exists(sym_path) or not os.path.exists(prm_path):
        save_mobilenet1_0()
    input_size = 224
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size, batch_size=16, ctx=ctx)

def test_mobilenet_v2_1_0():
    sym_path = "./data/mobilenetv2_1.0.json"
    prm_path = "./data/mobilenetv2_1.0.params"
    if not os.path.exists(sym_path) or not os.path.exists(prm_path):
        save_model('mobilenetv2_1.0')
    input_size = 224
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size, batch_size=16, ctx=ctx)

def test_tf_inceptionv3():
    sym_path = "./data/tf_inceptionv3.json"
    prm_path = "./data/tf_inceptionv3.params"
    input_size = 299
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size, batch_size=16, ctx=ctx)

def test_alexnet():
    utils.log_init()
    logger = logging.getLogger('log.validate.alexnet')
    logger.info('test alexnet started.')
    sym_path = "./data/alexnet.json"
    prm_path = "./data/alexnet.params"
    if not os.path.exists(sym_path) or not os.path.exists(prm_path):
        save_model('alexnet')
    input_size = 224
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, input_size, batch_size=700, ctx=ctx)
    logger.info('test alexnet end.')



if __name__ == '__main__':
    utils.log_init()
    # test_mobilenet1_0()
    # test_mobilenet_v2_1_0() # zero precision
    # test_tf_inceptionv3()
    test_alexnet()

