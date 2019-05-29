import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
import tvm
from tvm.contrib import graph_runtime
import nnvm
import pickle

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import utils

def load_fname(suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/trec%s" % (suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

batch_size = 32
ctx = mx.gpu()
inputs_ext = { 'data': {
    'shape': (38, batch_size)
}}
inputs = [mx.sym.var(n) for n in inputs_ext]

utils.log_init()

data_iter = ds.load_trec(batch_size)
def data_iter_func():
    return next(data_iter)
data, _ = data_iter_func()

sym_file, param_file = load_fname()
net1 = utils.load_model(sym_file, param_file, inputs, ctx=ctx)
def trec(data):
    res = net1(data.as_in_context(ctx))
    return res

sym, params = mx.sym.load(sym_file), nd.load(param_file)
sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
inputs_ext['data']['data'] = data
qsym, qparams, _ = calib.pure_int8_quantize(sym, params, inputs_ext, ctx=ctx)
net2 = gluon.nn.SymbolBlock(qsym, inputs)
utils.load_parameters(net2, qparams, ctx=ctx)
def quantize(data):
    data = sim.load_real_data(data, 'data', inputs_ext)
    res = net2(data.as_in_context(ctx))
    return res

utils.multi_eval_accuracy(trec, data_iter_func,
        quantize,
        iter_num=1000)

