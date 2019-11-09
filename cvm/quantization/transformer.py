import tfm_ops
from tfm_pass import *

import utils
import sym_utils as sutils
import cvm_op

import logging

#TODO(wlt): control available api for MRT

def init(symbol, params, input_shape=None):
    sym, params = graph_validate(symbol, params)
    if input_shape is not None:
        sym, params = attach_input_shape(sym, params, {'data': input_shape})
    infer_shape(sym, params) # check infer_shape is correct.
    sym, params = validate(sym, params)
    return sym, params


class MRT(object):
    def __init__(self, symbol, params,
            input_shape=None, data=None, input_prec=8):
        self._lgr = logging.getLogger('mrt')

        self._lgr.info("Graph initialize and reduce...")
        _sym, _prm = init(symbol, params, input_shape)
        orig_ops = calculate_ops(_sym, _prm)
        _sym, _prm = fuse_transpose(_sym, _prm)
        _sym, _prm = rewrite(_sym, _prm)
        _sym, _prm = fuse_constant(_sym, _prm)
        self._lgr.info("Original ops[%s] reduced into %s",
                orig_ops, calculate_ops(_sym, _prm))

        self._data = data
        self._fixed = set()

        self.precs = self._init_precs(input_prec)
        self.th_dict = None
        self.scales = {}

        self._qsym = None
        self._qprm = None

        self.op_input_precs = self._op_default_input_precs()
        self._sym, self._prm = _sym, _prm

    def set_data(self, data):
        self._data = data

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self._sym, self._prm, self._data,
                ctx=ctx, lambd=lambd, old_ths=old_ths)

    def set_input_prec(self, prec):
        self.precs['data'][OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self._sym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def quantize(self, no_realize=False):
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False

        qsym, qparams = quantize(self._sym, self._prm,
                self.th_dict, self.precs, self.scales, self.op_input_precs)
        '''
        if not no_realize:
            qsym, qparams = self._realize()
        qext = self._get_ext()
        '''
        return qsym, qparams

    def _op_default_input_precs(self):
        op_precs = {}
        for n in ['Convolution', 'FullyConnected', 'sigmoid', 'exp', 'softmax']:
            op_precs[n] = 8
        op_precs['sum'] = 8
        for n in ['broadcast_add', 'broadcast_sub', 'elemwise_add', 'elemwise_sub']:
            op_precs[n] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        return op_precs

    def _init_precs(self, input_prec):
        precs = {}
        for sym in topo_sort(self._sym):
            precs[sym.attr('name')] = {}
        precs['data'][OUT_KEY] = input_prec
        return precs

if __name__ == "__main__":
    import os
    import dataset as ds

    # set python-logging color format
    utils.log_init()

    # sym = mx.sym.load("/tmp/densenet/densenet161.json")
    # params = mx.nd.load("/tmp/densenet/densenet161.params")
    sym = mx.sym.load("./data/tf_inceptionv3.json")
    params = mx.nd.load("./data/tf_inceptionv3.params")
    print (collect_op_names(sym, params))
    print ("Registered Graph Pass")
    for k, v in pass_info().items():
        print ("%20s" % k, v)
    data_iter_func = ds.data_iter('imagenet', 1, input_size=224)
    data, _ = data_iter_func()
    mrt = MRT(sym, params, input_shape=(1, 3, 299, 299))
    mrt.set_data(data)
    mrt.calibrate()
    mrt.set_input_prec(8)
    mrt.set_output_prec(8)
    mrt.quantize()
    #sym, params = quantize(sym, params, th_dict, precs, scales)
    #print (calculate_ops(sym, params))

