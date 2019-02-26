# -*- coding:utf-8 -*-
from layers import IntDense, IntConv
from CVMSymbol import *
from CVMQuantization import *
# from mxboard import SummaryWriter
import mxnet as mx
import numpy as np
import logging
from mxnet import nd
from mxnet import autograd
import tvm
import shutil

logging.getLogger().setLevel(logging.DEBUG)
import argparse

parser = argparse.ArgumentParser(description='model quant')
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--is_quant_train', action='store_true')
args = parser.parse_args()

global batch
batch = 0

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
            Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
            Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                    arg_params[name] = v
            if tp == 'aux':
                    arg_params[name] = v
    return arg_params, aux_params

if __name__ == '__main__':

    batch_size = 256  # 定义每次处理数据的数量为64
    mnist = mx.test_utils.get_mnist()  # 使用内置的api读取mnist数据
    mnist['train_data'] = (mnist['train_data'] * 256).astype(int) - 128
    mnist['test_data'] = (mnist['test_data'] * 256).astype(int) - 128
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


    data = mx.symbol.Variable('data')

    #mxnet 原版网络
    if args.is_train and not args.is_quant_train:
        x = (data + 128) / 256.0
        x = mx.sym.Flatten(data=x, name='flatten')
        x = mx.sym.FullyConnected(data=x, num_hidden=32, name='fc1')
        x = mx.symbol.Activation(data=x, name='relu1', act_type="relu")
        x = mx.sym.FullyConnected(data=x, num_hidden=10, name='fc2')
        lenet= mx.sym.SoftmaxOutput(data=x, name='softmax')
    else:
        sb = mx.sym.Variable('sb0', init=mx.init.Constant(0))
        #  x = mx.sym.Flatten(data=data, name='flatten')
        #  x, sb = mx.symbol.Custom(data=x, sbits=sb, op_type='cvm.dense', num_hidden=64,)
        #  x = mx.symbol.Activation(data=x, act_type="relu")
        #  x, sb = mx.symbol.Custom(data=x, sbits=sb, op_type='cvm.dense', num_hidden=32,)
        #  x = mx.symbol.Activation(data=x, act_type="relu")
        #  x, sb = mx.symbol.Custom(data=x, sbits=sb, op_type='cvm.dense', num_hidden=10,)
        #  x = mx.sym.broadcast_div(x, mx.sym.pow(2, sb))
        #  lenet = mx.sym.SoftmaxOutput(data=x, name='softmax')


        x, sb = mx.symbol.Custom(data=data, sbits=sb, name='conv1', op_type='cvm.conv',
                                    num_filter=32, kernel=(3, 3),
                                    pad=(1, 1), stride=(1, 1), no_bias=True,)
        x = mx.symbol.Activation(data=x, name='relu1', act_type="relu")
        #  x = mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool1')
        #  x, sb = mx.symbol.Custom(data=x, sbits=sb, name='conv2', op_type='cvm.conv',
                                    #  num_filter=64, kernel=(5, 5),
                                    #  stride=(1, 1), pad=(2, 2), no_bias=True,)
        #  x = mx.sym.Activation(data=x, act_type="relu", name='relu2')
        #  x = mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool2')
        x = mx.sym.Flatten(data=x, name='flatten')
        #  x, sb = mx.symbol.Custom(data=x, sbits=sb, op_type='cvm.dense', num_hidden=512)
        #  x = mx.sym.Activation(data=x, act_type="relu", name='relu3')
        x, sb = mx.symbol.Custom(data=x, sbits=sb, op_type='cvm.dense', num_hidden=10)
        x = mx.sym.broadcast_div(x, mx.sym.pow(2, sb))
        lenet = mx.sym.SoftmaxOutput(data=x, name='softmax')

    if args.is_train:
        mod = mx.mod.Module(lenet, context=mx.gpu(1))
        # mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
        mod.fit(train_data=train_iter,
                eval_data=val_iter,
                optimizer='adam',
                optimizer_params={'learning_rate':1e-3 , },
                num_epoch=5,
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(batch_size, 100))
        mod.save_checkpoint('int_dense_conv_mnist_7bit', 20)
    elif args.is_quant_train:
        # print (lenet.tojson())
        mod = mx.mod.Module(lenet, data_names = ['data'], label_names = ['softmax_label'], context=mx.gpu(1))
        mod.fit(train_data=train_iter,
                eval_data=val_iter,
                optimizer='adam',
                optimizer_params={'learning_rate':1e-3 , },
                num_epoch=5,
                eval_metric='acc',
                initializer=mx.init.Uniform(0.01),
                #  initializer=mx.init.Mixed(['.*shift_bit', '.*'], [mx.init.Constant(7), mx.init.Uniform(0.01)]),
                batch_end_callback=mx.callback.Speedometer(batch_size, 100))
        mod.save_checkpoint('int_dense_conv_mnist_7bit_test', 1)
    else:
        # arg_params, aux_params = load_checkpoint('mnist', 20)
        arg_params, aux_params = load_checkpoint('int_dense_conv_mnist_7bit_test', 1)
        mod = mx.mod.Module(lenet, context=mx.gpu(1))
        mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
        mod.set_params(arg_params, aux_params, allow_missing=True, force_init=True, allow_extra=True)
        print (lenet.tojson())
    score = mod.score(val_iter, ['acc', 'loss'])  # 进行inference
    print("Accuracy score is ", (score[0]))  # 输出预测精度

