from __future__ import print_function  # only relevant for Python 2
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import nnvm
import tvm
from tvm.contrib import graph_runtime

from quant_op import *
from quant_utils import *
import utils
import sym_annotate as anno
import sym_utils as sutils
import sym_pass as spass
import sym_calib as calib
import sim_quant_helper as sim
import gluon_zoo as zoo

import numpy as np

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    if with_ext:
        return "./data/mnist%s%s.json"%(version, suffix), \
            "./data/mnist%s%s.params"%(version, suffix), \
            "./data/mnist%s%s.ext"%(version, suffix)
    else:
        return "./data/mnist%s%s.json"%(version, suffix), \
            "./data/mnist%s%s.params"%(version, suffix)

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)

batch_size = 128
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)

version = ''
ctx = mx.gpu(2)
def train_mnist():
    # Select a fixed random seed for reproducibility
    mx.random.seed(42)

    if version == '':
        net = nn.HybridSequential(prefix='DApp_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=16, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=32, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=64, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(10, activation=None),
            )
    elif version == 'lenet':
        net = nn.HybridSequential(prefix='LeNet_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=20, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=50, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(500, activation='relu'),
                nn.Dense(10, activation=None),
            )
    elif version == 'mlp':
        net = nn.HybridSequential(prefix='MLP_')
        with net.name_scope():
            net.add(
                nn.Flatten(),
                nn.Dense(128, activation='relu'),
                nn.Dense(64, activation='relu'),
                nn.Dense(10, activation=None)  # loss function includes softmax already, see below
            )

    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.summary(nd.zeros((1, 1, 28, 28), ctx=ctx))

    trainer = gluon.Trainer(
	params=net.collect_params(),
	optimizer='adam',
	optimizer_params={'learning_rate': 1e-3},
    )
    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            metric.update(labels, outputs)

            trainer.step(batch_size=inputs.shape[0])

        name, acc = metric.get()
        print('After epoch {}: {} = {:5.2%}'.format(epoch + 1, name, acc))
        metric.reset()

    for inputs, labels in val_loader:
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        metric.update(labels, net(inputs))
    print('Validaton: {} = {}'.format(*metric.get()))
    assert metric.get()[1] > 0.96

    sym = net(mx.sym.var('data'))
    sym_file, param_file = load_fname(version)
    open(sym_file, "w").write(sym.tojson())
    net.collect_params().save(param_file)

def test_sym_pass(iter_num=10):
    inputs_ext = { 'data': {
            'shape': (batch_size, 1, 28, 28),
    } }
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = iter(val_loader)
    def data_iter_func():
        return next(data_iter)
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    def graph_func(data):
        return net1.forward(data.as_in_context(ctx))

    # sym_file, param_file = load_fname(version)
    # sym, params = mx.sym.load(sym_file), nd.load(param_file)
    # sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    # qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, ctx)
    # qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    # dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    # sim.save_ext(dump_ext, inputs_ext)
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())

    # dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    # sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    # (inputs_ext,) = sim.load_ext(dump_ext)
    # inputs = [mx.sym.var(n) for n in inputs_ext]
    # net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    # def cvm_quantize(data):
    #     data = sim.load_real_data(data, 'data', inputs_ext)
    #     return net2.forward(data.as_in_context(ctx))

    sym_file, param_file = load_fname(version)
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    print (sutils.sym_collect_attr(sym))
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    inputs_ext['data']['data'] = data
    qsym, qparams, _ = anno.mixed_precision(sym, params, inputs_ext,
            ctx=[mx.gpu(7)])
    # qsym, qparams, precs = anno.sym_annotate(sym, params, inputs_ext)
    # qsym, qparams, _ = anno.sym_simulate(qsym, qparams, inputs_ext, precs, ctx=[ctx])
    net3 = nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(net3, qparams, ctx=ctx)
    def mixed_precision(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return net3.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func,
            mixed_precision, # cvm_quantize,
            iter_num=iter_num)

def test_nnvm_pass(iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = { 'data': {
        'shape': (batch_size, 1, 28, 28),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = iter(val_loader)
    def data_iter_func():
        return next(data_iter)
    data, _ = data_iter_func()

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net1 = utils.load_model(dump_sym, dump_params, inputs, ctx=mx_ctx)
    def mnist_quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        np.save("/tmp/mnist/data.npy", data.asnumpy().astype('int8'))
        res = net1.forward(data.as_in_context(mx_ctx))
        np.save("/tmp/mnist/result.npy", res.asnumpy().astype('int8'))
        return res

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)

    use_dtype = "int32"
    for key, value in list(real_params.items()):
       real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)

    with nnvm.compiler.build_config(opt_level=0, runtime="cvm"):
       deploy_graph, lib, real_params = nnvm.compiler.build(
           nnvm_sym, target=target, shape=inputs_shape,
           params=real_params, dtype=use_dtype)

    real_params = spass.tvm_params_reduce(nnvm_sym, real_params, inputs_ext, tvm_ctx)
    dump_symbol, dump_params = '/tmp/mnist/symbol.json', '/tmp/mnist/params'
    with open(dump_symbol, "w") as fout:
       fout.write(deploy_graph.json())
    with open(dump_params, "wb") as fout:
       param_bytes = nnvm.compiler.save_param_dict(real_params)
       fout.write(param_bytes)

print ("Test mnist", version )
# train_mnist()
utils.log_init()
test_sym_pass(10000)
# test_nnvm_pass(10)
