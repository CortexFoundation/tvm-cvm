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
        net.add(Convolution_int(acti_bit=8, channels=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), in_channels=1, use_bias=False),
                Activation_int(8, 'relu'),
                Convolution_int(acti_bit=8, channels=128, kernel_size=(3,3), strides=(1,1), padding=(1,1), in_channels=64, use_bias=False),
                Activation_int(8, 'relu'),
                Dense(units=10, acti_bit=8, in_units=100352),)
    net.load_params('./mnist_quantize_bias_dense_conv.params', allow_missing=True, ignore_extra=True, ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    batch_size = 64
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})
    #utils.train(train_data, test_data, net, loss, trainer, ctx=ctx, num_epochs=10)
    #如果不需要转为sym-module，将下一行注释掉即可。
    net.hybridize()
    #test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    #net.save_params('./mnist_quantize_bias_dense_conv.params')
    #print('test acc : ', test_acc)
    #net.export('mnist_quantize')
    #net.save_params('./mnist_quantize.params')

    # added
    sym, params = nnvm.frontend.from_mxnet(net)
    graph, lib, params = nnvm.compiler.build(sym, 'llvm', shape={'data': (1, 784)}, params=params)

    remote = rpc.LocalSession()
    remote_ctx = remote.gpu(0)
    module = rumtime.create(graph, lib, ctx=remote_ctx)

    module.set_input('data', train_data[0])
    module.run()
    output = moudule.get_output(0, tvm.nd.empty(1,), dtype='float32')
    print(output)


if __name__ == '__main__':
	main()
