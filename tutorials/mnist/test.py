# -*- coding:utf-8 -*-
import os
import mxnet as mx
import numpy as np
import logging
from numpy import *
from mxnet import nd
from mxnet import autograd
from math import floor
from mxnet.test_utils import get_mnist_iterator
import int_symbol
import tvm
from tvm.contrib import util
import tvm.contrib.graph_runtime
import nnvm
from mxnet import gluon

if __name__ == '__main__':
    batch_size = 400
    length = 784
    data_shape = (batch_size, length)

    mnist = mx.test_utils.get_mnist()

    #train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True, label_name='fc3')
    #val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size, label_name='fc3')
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Custom(data, name='conv1', op_type='intconv', bit=8, num_filter=3, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
    flatten = mx.symbol.flatten(conv1, name='flatten')
    module = mx.mod.Module(flatten, context=mx.gpu(0))
    #module.fit(train_iter, val_iter, num_epoch=20, optimizer='adam', optimizer_params={'learning_rate':0.0001}, eval_metric='acc', initializer=mx.init.Xavier())

    #model = mx.model.FeedForward.create(net, X=train_iter, num_epoch=20, learning_rate=0.0001)
    #print(model)

    #module.save('model', 0)

    #sym = mx.sym.loads('int_mnist-symbol.json')
    #params = bytearray(open('int_mnist-0100.params', 'rb').read())
    #net = gluon.SymbolBlock.imports('int_mnist-symbol.json', ['data'], param_file='int_mnist-0100.params', ctx=mx.gpu())
    #print ('net', net)

    #graph, lib, params = nnvm.compiler.build(sym, 'llvm', shape={'data': batch_size}, params=params)
    #module = runtime.create(graph, lib, tvm.gpu())

    #sym, params = nnvm.frontend.from_mxnet(module(mx.symbol.Variable('data', dtype=dtype)))
    #graph, lib, params = nnvm.compiler.build(sym, target, shape={'data': batch_size}, params=params)

    #module = graph_runtime.create(graph, lib, tvm.gpu())
'''
	batch_size = 400  # 定义每次处理数据的数量为64
	mnist = mx.test_utils.get_mnist()  # 使用内置的api读取mnist数据

        # 划分训练集和验证集，本代码仅用验证集做inference
	train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True, label_name='fc3')
	val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size, label_name='fc3')

	data = mx.symbol.Variable('data')
	conv1 = mx.symbol.Custom(data, name='conv1', op_type='intconv', bit=8, num_filter=3, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
        flatten = mx.symbol.flatten(conv1, name='flatten')

	fc1 = mx.symbol.Custom(data=flatten, name='fc1', op_type='intdense', bit=8, num_hidden=128)
	act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
	fc2 = mx.symbol.Custom(data=act1, name='fc2', op_type='intdense', bit=8, num_hidden=64)
	act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
	fc3 = mx.symbol.Custom(data=act2, name='fc3', op_type='intdense', bit=8, num_hidden=10)
	#mlp = mx.symbol.Softmax(fc3, name='softmax')
	mod = mx.mod.Module(fc3, context=mx.gpu(0))
	mod.fit(train_iter, val_iter, num_epoch=20, optimizer='adam', optimizer_params={'learning_rate':0.0001}, eval_metric='acc', initializer=mx.init.Xavier())
	score = mod.score(val_iter, ['acc'])  # 进行inference
	mod.save_checkpoint('int_mnist', 100)
	print("Accuracy score is ", (score[0]))  # 输出预测精度
        '''
