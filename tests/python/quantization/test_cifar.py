import tvm
from tvm.contrib import graph_runtime
import nnvm

import utils
import sim_quant_helper as sim
import sym_pass as spass
import cvm_op

def load_fname(prefix, suffix=None):
    suffix = "."+suffix if suffix is not None else ""
    load_prefix = prefix + suffix
    names = list(utils.extend_fname(load_prefix, True))
    dump_prefix = prefix + ".nnvm.compile"
    names.extend(utils.extend_fname(dump_prefix, False))
    return tuple(names)

names = load_fname("./data/cifar_resnet20_v2", "sym.quantize")
spass.to_nnvm(*names)


