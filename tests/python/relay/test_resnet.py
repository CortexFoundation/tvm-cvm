import tvm
from tvm import relay
from tvm.relay import quantize as qtz
from tvm.relay.frontend import mxnet as tvm_mxnet

import mxnet

def load_mxnet_resnet():
    symbol_file = "/home/wlt/tvm-cvm/data/resnet-152-symbol.json"
    params_file = "/home/wlt/tvm-cvm/data/resnet-152-0000.params"

    resnet_symbol = mxnet.symbol.load(symbol_file)
    resnet_params = mxnet.nd.load(params_file)
    input_shape = (1, 3, 224, 224)
    sym, params = tvm_mxnet.from_mxnet(resnet_symbol,
            shape={'data': input_shape},
            arg_params=resnet_params)

    with qtz.qconfig(skip_k_conv=0, global_scale=4.0,
                    round_for_shift=False, store_lowbit_output=False):
        qsym = qtz.quantize(sym, params=params, dataset=[])
        qsym = relay.ir_pass.infer_type(qsym)
        print (qsym)

    # print (sym)

    # with relay.build_config(opt_level=0):
        # symbol, mod, params = relay.build_module.build(sym, target="llvm", params=params)
        # print (symbol)

if __name__ == "__main__":
    load_mxnet_resnet()
