import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet import ndarray as nd
import logging

from quant_utils import *
from quant_op import *
import quant_pass as qpass
from sym_pass import *
from utils import *


class Dense(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(Dense, self).__init__(prefix='fc0_', **kwargs)

        self.quant_flag = quant_flag

        setattr(self, 'bias',
            self.params.get('bias',
                    init='zeros',
                    allow_deferred_init=True))

        if not quant_flag.matrix_decomposition:
            setattr(self, 'weight',
                self.params.get('weight',
                        init='zeros',
                        allow_deferred_init=True))
            return

        self.matrix_len = 100352
        self.max_len = 1000

        start, step, idx = 0, self.max_len, 0
        while start < self.matrix_len:
            stop = min(start+step, self.matrix_len)

            weight_name = 'weight' + str(idx)
            setattr(self, weight_name,
                self.params.get(weight_name,
                            init='zeros',
                            allow_deferred_init=True))

            start, idx = stop, idx+1

        for i in range(idx-1):
            plus_name_f = '_plus' + str(i) + '_first_shift_bits'
            plus_name_s = '_plus' + str(i) + '_second_shift_bits'
            setattr(self, plus_name_f,
                self.params.get(plus_name_f,
                            init='zeros',
                            allow_deferred_init=True))
            setattr(self, plus_name_s,
                self.params.get(plus_name_s,
                            init='zeros',
                            allow_deferred_init=True))

            requant_name = '_plus' + str(i) + '_requant_shift_bits'
            setattr(self, requant_name,
                self.params.get(requant_name,
                            init='zeros',
                            allow_deferred_init=True))

    def hybrid_forward(self, F, x, **kwargs):
        if not self.quant_flag.matrix_decomposition:
            x = F.FullyConnected(x, kwargs['weight'],
                    kwargs['bias'], num_hidden=10)
        else:
            nodes = []
            start, step, idx = 0, self.max_len, 0
            while start < self.matrix_len:
                stop = min(start+step, self.matrix_len)

                weight_name = 'weight' + str(idx)
                tmp = F.slice(x, begin=(None, start), end=(None, stop))
                tmp = F.FullyConnected(tmp, kwargs[weight_name],
                        kwargs['bias'], num_hidden=10)
                nodes.append(tmp)

                start, idx = stop, idx+1

            i = 0
            while len(nodes) > 1:
                a, b = nodes.pop(0), nodes.pop(0)

                if self.quant_flag.calib_mode != CalibMode.NONE:
                    a_sb_name = '_plus' + str(i) + '_first_shift_bits'
                    a , _ = quant_helper(a, shift_bits=kwargs[a_sb_name], F=F)

                    b_sb_name = '_plus' + str(i) + '_second_shift_bits'
                    b , _ = quant_helper(b, shift_bits=kwargs[b_sb_name], F=F)

                out = a + b

                if self.quant_flag.calib_mode != CalibMode.NONE:
                    requant_name = '_plus' + str(i) + '_requant_shift_bits'
                    out, _ = quant_helper(out, shift_bits=kwargs[requant_name], F=F)

                nodes.append(out)

                i += 1

            x = nodes[0]

        return x

class SimpleNet(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(SimpleNet, self).__init__(**kwargs)

        self.forward = nn.HybridSequential(prefix='')

        self.forward.add(nn.Conv2D(128, kernel_size=3, strides=1,
            padding=1, use_bias=False))
        requant_helper(self.forward, quant_flag)

        self.forward.add(nn.Activation('relu'))

        # self.forward.add(nn.Dense(10, use_bias=True, prefix='fc0_'))
        self.forward.add(nn.Flatten())
        self.forward.add(Dense(quant_flag))

    def hybrid_forward(self, F, x):
        x = self.forward(x)
        return x

def get_dump_fname(suffix="quant"):
    return './data/simplenet.json.%s'%suffix, \
        './data/simplenet.params.%s'%suffix

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

def load_dataset(batch_size=10):
    data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
    loader = mx.gluon.data.DataLoader(data, shuffle=False, batch_size=batch_size)

    return iter(loader)

def test_load_simplenet(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.main")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 1, 28, 28),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = load_dataset(batch_size)
    data, label = next(data_iter)

    symbol_file, params_file = "./data/simplenet.json", "./data/simplenet.params"
    sym = mx.sym.load(symbol_file)
    qparams = nd.load(params_file)
    # qsym, qparams = mx_sym_rewrite(sym, qparams, quant_flag, inputs_ext=inputs_ext)
    # exit()

    logger.info("quantization")
    scope_graph = nn.HybridSequential(prefix='calib_')
    with scope_graph.name_scope():
        graph = SimpleNet(QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NONE,
                    matrix_decomposition=True))
        scope_graph.add(graph)

    qparams = qpass.matrix_decomposition(qparams, quant_flag)
    qparams = qpass.calibrate_parameters(scope_graph, qparams, ctx,
            data, quant_flag, name_scope='calib_')

    graph= SimpleNet(quant_flag)
    print ("SSSSSS", graph.collect_params().keys(), "\n", qparams.keys())
    qsym, qparams = graph(inputs[0]), load_parameters(graph, qparams, ctx=ctx)
    qsym, qparams = fold_cond_op(qsym, qparams, {}, quant_flag)

    dump_sym, dump_params = get_dump_fname("matrix")
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
        fout.write(qsym.tojson())

    logger.info("load model&quant_model")
    mx_graph = nn.SymbolBlock(sym, inputs)
    load_parameters(mx_graph, nd.load(params_file), ctx=ctx, prefix='calib_')

    qgraph = nn.SymbolBlock(qsym, inputs)
    load_parameters(qgraph, qparams, ctx=ctx)

    logger.info("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(iter_num):
        quant_data, _ = quant_helper(data)

        res = mx_graph.forward(data.as_in_context(ctx))

        if quant_flag.calib_mode == CalibMode.NONE:
            quant_data = data
        qres = qgraph.forward(quant_data.as_in_context(ctx))

        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            qres_label = qres[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()

            diff += 0 if res_label == qres_label else 1
            acc += 1 if res_label == data_label else 0
            qacc += 1 if qres_label == data_label else 0
            total += 1

        try:
            data, label = next(data_iter)
        except:
            exit()

        logger.info("Iteration: %5d | Accuracy: %.2f%% | Quant Acc: %.2f%%" +
                " | Difference: %.2f%% | Total Sample: %5d",
                i, 100.*acc/total, 100.*qacc/total, 100.*diff/total, total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)

    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    allows=["log.quant", "log.calib", "log.main", "log.test"]
    disables = ["log.quant.op.requant.helper.ddd", "autotvm"]

    log_filter = FilterList(
                allows=allows, disables=disables,
                # keywords=["layer=pool", "calib_pool"],
                log_level=logging.INFO,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG, use_scalar=False,
            matrix_decomposition=True,
            disabled_layers=["relu", "pool0", "activation"])

    test_load_simplenet(quant_flag, batch_size=10, iter_num=100000)
