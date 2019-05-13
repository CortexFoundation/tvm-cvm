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
import sym_utils as sutils
import sym_pass as spass
import sym_calib as calib
import sim_quant_helper as sim
import gluon_zoo as zoo

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

batch_size = 100
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)

def train_mnist():
    # Select a fixed random seed for reproducibility
    mx.random.seed(42)

    lenet = nn.HybridSequential(prefix='LeNet_')
    with lenet.name_scope():
        lenet.add(
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

    ctx = mx.gpu(3)
    lenet.initialize(mx.init.Xavier(), ctx=ctx)
    lenet.summary(nd.zeros((1, 1, 28, 28), ctx=ctx))

    trainer = gluon.Trainer(
        params=lenet.collect_params(),
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
                outputs = lenet(inputs)
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
        metric.update(labels, lenet(inputs))
    print('Validaton: {} = {}'.format(*metric.get()))
    assert metric.get()[1] > 0.985

    name = 'mnist'
    sym = lenet(mx.sym.var('data'))
    with open("./data/%s.json"%name, "w") as fout:
        fout.write(sym.tojson())
    lenet.collect_params().save("./data/%s.params"%name)

def test_sym_nnvm(iter_num=10):
    target = "cuda"
    ctx = mx.gpu(3)
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = { 'data': {
            'shape': (batch_size, 1, 28, 28),
    } }
    inputs = [mx.sym.var(n) for n in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = iter(val_loader)
    def data_iter_func():
        return next(data_iter)
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(""), inputs, ctx=ctx)
    def graph_func(data):
        return net1.forward(data.as_in_context(ctx))

    load_sym, load_params, load_ext = load_fname("", "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(load_ext)
    net2 = utils.load_model(*load_fname("", "sym.quantize"), inputs, ctx=ctx)
    def cvm_quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return net2.forward(data.as_in_context(ctx))

    sym, params = mx.sym.load(load_sym), nd.load(load_params)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)

    use_dtype = "int32"
    for key, value in list(real_params.items()):
       real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)
    with nnvm.compiler.build_config(opt_level=0, runtime="cvm"):
       deploy_graph, lib, real_params = nnvm.compiler.build(
           nnvm_sym, target=target, shape=inputs_shape,
           params=real_params, dtype=use_dtype)
    #  real_params = spass.tvm_params_reduce(nnvm_sym, real_params, inputs_ext, tvm_ctx)

    dump_symbol, dump_params = load_fname("", "nnvm.compile")
    open(dump_symbol, "w").write(deploy_graph.json())
    param_bytes = nnvm.compiler.save_param_dict(real_params)
    open(dump_params, "wb").write(param_bytes)

    # module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    # module.load_params(param_bytes)
    # def nnvm_real(data):
    #     data = sim.load_real_data(data, 'data', inputs_ext)
    #     data = tvm.nd.array(data.asnumpy(), tvm_ctx)
    #     module.run(data=data.asnumpy())
    #     return nd.array(module.get_output(0).asnumpy())

    utils.multi_eval_accuracy(graph_func, data_iter_func,
            cvm_quantize, # nnvm_real,
            iter_num=iter_num)


def test_sym_pass(iter_num=10):
    ctx = mx.gpu(3)
    inputs_ext = { 'data': {
            'shape': (batch_size, 1, 28, 28),
    } }
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = iter(val_loader)
    def data_iter_func():
        return next(data_iter)
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(""), inputs, ctx=ctx)
    def graph_func(data):
        return net1.forward(data.as_in_context(ctx))

    sym_file, param_file = load_fname("")
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    ops = spass.sym_calculate_ops(sym, params, inputs_ext)
    print (ops)
    # sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    # qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, ctx)
    # qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    # dump_sym, dump_params, dump_ext = load_fname("", "sym.quantize", True)
    # sim.save_ext(dump_ext, inputs_ext)
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname("", "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    ops = spass.sym_calculate_ops(sym, params, inputs_ext)
    print (ops)
    exit()
    (inputs_ext,) = sim.load_ext(dump_ext)
    inputs = [mx.sym.var(n) for n in inputs_ext]
    net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    def cvm_quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return net2.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func,
            cvm_quantize,
            iter_num=iter_num)

# train_mnist()
utils.log_init()
test_sym_pass(10)
test_sym_nnvm(10000)
