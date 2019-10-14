from tfm_base import *
import tfm_ops
from sym_utils import *

def init(symbol, params, input_shape=None):
    sym, params = check_graph(symbol, params)
    if input_shape is not None:
        sym, params = attach_input_shape(sym, params, input_shape)
    infer_shape(sym, params) # check infer_shape is correct.
    sym, params = validate(sym, params)
    return sym, params

if __name__ == "__main__":
    sym = mx.sym.load("./data/tf_inceptionv3.json")
    params = mx.nd.load("./data/tf_inceptionv3.params")
    sym, params = init(sym, params, (1, 3, 644, 644))
    print (collect_op_names(sym, params))
    print (calculate_ops(sym, params))
    sym, params = fuse_constant(sym, params)
    print (calculate_ops(sym, params))
    sym, params = fuse_transpose(sym, params)
    print (calculate_ops(sym, params))
    sym, params = rewrite(sym, params)
    print (calculate_ops(sym, params))
    with open("/tmp/tmp_v2.json", "w") as fout:
        fout.write(sym.tojson())
