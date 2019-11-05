from tfm_base import *
import tfm_ops
import cvm_op
from sym_utils import *
import utils
from tfm_pass import *

def init(symbol, params, input_shape=None):
    utils.log_init()
    sym, params = graph_validate(symbol, params)
    if input_shape is not None:
        sym, params = attach_var_shape(sym, params, input_shape)
    infer_shape(sym, params) # check infer_shape is correct.
    sym, params = validate(sym, params)
    return sym, params

if __name__ == "__main__":
    import os
    import dataset as ds

    # sym = mx.sym.load("/tmp/densenet/densenet161.json")
    # params = mx.nd.load("/tmp/densenet/densenet161.params")
    sym = mx.sym.load("./data/tf_inceptionv3.json")
    params = mx.nd.load("./data/tf_inceptionv3.params")
    sym, params = init(sym, params, (1, 3, 299, 299))
    # sym = mx.sym.load("./data/resnet18_v1.json")
    # params = mx.nd.load("./data/resnet18_v1.params")
    # sym, params = init(sym, params, (1, 3, 224, 224))
    print (collect_op_names(sym, params))
    print ("Registered Graph Pass")
    for k, v in pass_info().items():
        print ("%20s" % k, v)
    print (calculate_ops(sym, params))
    sym, params = fuse_constant(sym, params)
    print (calculate_ops(sym, params))
    sym, params = fuse_transpose(sym, params)
    print (calculate_ops(sym, params))
    sym, params = rewrite(sym, params)
    print (calculate_ops(sym, params))
    with open(os.path.expanduser("~/tvm-cvm/data/tmp_v2.json"), "w") as fout:
        fout.write(sym.tojson())
    data_iter_func = ds.data_iter('imagenet', 1, input_size=224)
    data, _ = data_iter_func()
    th_dict = sym_calibrate(sym, params, data, \
            ctx=mx.cpu(), old_ths=None, lambd=None)
    sym, params = quantize(sym, params, th_dict)
    print (calculate_ops(sym, params))

