# -*- encode: utf8 -*-

import mxnet as mx
import utils
from int_layer import *
from mxnet.gluon import nn
from mxnet import init, gluon, nd
from mxnet.gluon.data.vision.datasets import MNIST

# added
import nnvm
import tvm
from tvm import rpc

def main():
    ctx = mx.cpu()
    train_data, test_data = utils.load_data_mnist(batch_size=64)
    net = nn.HybridSequential()
    with net.name_scope():
        net.add( Dense(units=10, acti_bit=8, in_units=100352),)
    net.initialize()
    batch_size = 64
    #utils.train(train_data, test_data, net, loss, trainer, ctx=ctx, num_epochs=10)
    #如果不需要转为sym-module，将下一行注释掉即可。
    net.hybridize()
    #test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    #net.save_params('./mnist_quantize_bias_dense_conv.params')
    #print('test acc : ', test_acc)
    #net.export('mnist_quantize')
    #net.save_params('./mnist_quantize.params')
    print (net(mx.symbol.Variable('data')).tojson())
    # added
    sys, params = nnvm.frontend.from_mxnet(net)




if __name__ == '__main__':
	main()
