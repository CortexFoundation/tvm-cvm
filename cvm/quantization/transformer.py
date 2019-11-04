from tfm_base import *
import tfm_ops
import cvm_op
from sym_utils import *
from tfm_pass import *
import utils

import logging

# set python-logging color format
utils.log_init()

def init(symbol, params, input_shape=None):
    sym, params = graph_validate(symbol, params)
    if input_shape is not None:
        sym, params = attach_input_shape(sym, params, input_shape)
    infer_shape(sym, params) # check infer_shape is correct.
    sym, params = validate(sym, params)
    return sym, params


class MRT(object):
    OUT_KEY = 'out_key'
    TARGET_KEY = 'target_key'

    def __init__(self, symbol, params,
            input_shape=None, data=None, input_prec=8):
        self._lgr = logging.getLogger('mrt')

        _sym, _prm = init(symbol, params, input_shape)
        _sym, _prm = fuse_constant(_sym, _prm)
        _sym, _prm = fuse_transpose(_sym, _prm)
        self._sym, self._prm = rewrite(_sym, _prm)
        self._lgr.info("ops=%s", calculate_ops(self._sym, self._prm))

        self._data = data
        self._fixed = set()

        self.precs = {}
        for sym in topo_sort(self._sym):
            self.precs[sym.attr('name')] = {}
        self.precs['data'][MRT.OUT_KEY] = input_prec

        self.th_dict = None

    def set_data(self, data):
        self._data = data

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self._sym, self._prm, self._data,
                ctx=ctx, lambd=lambd, old_ths=old_ths)

    def set_input_prec(self, prec):
        self.precs['data'][MRT.OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self._sym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def set_fixed(self, fixes):
        if isinstance(fixes, list):
            self._fixed.update(fixes)
        else:
            self._fixed.add(fixes)

if __name__ == "__main__":
    import os
    import dataset as ds

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
    mrt.set_fixed('data')
    mrt.set_output_prec(8)
    #sym, params = quantize(sym, params, th_dict, precs, scales)
    #print (calculate_ops(sym, params))

