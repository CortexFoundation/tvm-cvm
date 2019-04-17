
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

import sym_calib as calib
import utils
import gluon_zoo as zoo
import sym_pass as spass
import sim_quant_helper as sim

import logging

def get_dump_fname(suffix="quant"):
    return './data/mobilenet1_0.json.%s'%suffix, \
        './data/mobilenet1_0.params.%s'%suffix

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    symbol_file, params_file = "./data/mobilenet1_0.json", "./data/mobilenet1_0.params"
    sym, params = mx.sym.load(symbol_file), nd.load(params_file)
    graph_comp = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph_comp, params, ctx=ctx)
    def graph_func(data):
        return graph_comp.forward(data.as_in_context(ctx))

    qsym, qparams = spass.sym_quant_prepare(sym, params, inputs_ext)
    qsym, qparams, th_dict = calib.sym_calib_sim_quant(qsym,
            qparams, inputs_ext, data, ctx)
    dump_sym, dump_params = get_dump_fname('sym.sim.pass')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
       fout.write(qsym.tojson())
    qsym, qparams = mx.sym.load(dump_sym), nd.load(dump_params)
    qgraph = nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(qgraph, qparams, ctx=ctx)
    def simulate(data):
        data = sim.load_quant_data(data, 'data', qparams)
        data = sim.int_realize(data, 8)
        return qgraph.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func, simulate,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet1_0()
    test_sym_pass(16, 100000)
