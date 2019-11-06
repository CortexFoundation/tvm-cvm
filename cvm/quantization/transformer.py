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
    _out_key = 'out_key'
    _tartget_key = 'target_key'

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

        self.precs = {s.attr('name'): {} for s in topo_sort(_sym)}
        self.precs['data'][MRT._out_key] = input_prec

        self.th_dict = None
        self._sym, self._prm = _sym, _prm

    def set_data(self, data):
        self._data = data

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self._sym, self._prm, self._data,
                ctx=ctx, lambd=lambd, old_ths=old_ths)

    def set_input_prec(self, prec):
        self.precs['data'][MRT._out_key] = prec

    def set_output_prec(self, prec):
        for sym in self._sym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def set_fixed(self, fixes):
        if isinstance(fixes, list):
            self._fixed.update(fixes)
        else:
            self._fixed.add(fixes)
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False
        # Check Fixed
        for sym in topo_sort(self._sym):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if name not in self._fixed:
                continue
            assert op_name == 'null'
            if is_params(sym, self.prm):
                bit = _get_bit(self.prm[name])
                if MRT._out_key in self.precs[name]:
                    prec = self.precs[name][MRT._out_key]
                    assert prec >= bit
                    self.precs[name][out_key] = PREC(bit, LFIX)
            else:
                bit = self.precs[name][out_key].p
            self.th_dict[name] = _get_range(bit)

    def quantize(self, no_realize=False):
        qsym, qparams = self._simulate()
        '''
        if not no_realize:
            qsym, qparams = self._realize()
        qext = self._get_ext()
        '''
        return qsym, qparams, qext

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
    # mrt.calibrate()
    # mrt.set_input_prec(8)
    # mrt.set_fixed('data')
    # mrt.set_output_prec(8)
    # mrt.quantize()
    #sym, params = quantize(sym, params, th_dict, precs, scales)
    #print (calculate_ops(sym, params))

