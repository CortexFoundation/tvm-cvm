from mxnet.gluon import HybridBlock, nn
from mxnet import ndarray as nd

import logging

from quant_utils import *

class ReQuant(HybridBlock):
    """Given a input data of type FP32, quantize it into a INT8 and shift bits

    Parameters
    ----------
    target_bits: int
        Quantize input data into target bits' int type
    """
    def __init__(self, quant_flag, pre_lname='conv', prefix=None,
            requant_sb_initializer='zeros', **kwargs):
        prefix_name = pre_lname if pre_lname.endswith("_") else pre_lname+"_"
        super(ReQuant, self).__init__(prefix=prefix_name, **kwargs)

        self.prefix_name = prefix_name
        self.shift_bits = self.params.get('requant_shift_bits',
                                shape=(1,),
                                init=requant_sb_initializer,
                                allow_deferred_init=True)
        self.logger = logging.getLogger("log.quant.op.requant")
        self.logger.setLevel(quant_flag.log_level)

    def hybrid_forward(self, F, x, shift_bits):
        out, _ = quant_helper(x, shift_bits=shift_bits,
                logger=self.logger, msg=self.prefix_name[:-1])
        return out

    def __repr__(self):
        s = '{name}(pre_lname={pre_lname}, shift_bits={shift_bits})'
        return s.format(name='_requant',
                        pre_lname=self.pre_lname,
                        shift_bits=self.shift_bits)

def requant_helper(graph, quant_flag):
    logger = logging.getLogger("log.quant.op.requant.helper")
    logger.setLevel(quant_flag.log_level)

    if quant_flag.calib_mode == CalibMode.NONE:
        return

    previous_lname = graph[-1].name

    # allow_flag = any(allow_layer in previous_lname for allow_layer in quant_flag.allowed_layers)
    # if len(quant_flag.allowed_layers) > 0 and (not allow_flag):
        # logger.debug("disable requant for layer [ %s ] not in allowed_layers", previous_lname)
        # return

    disable_flag = any(dis_layer in previous_lname for dis_layer in quant_flag.disabled_layers)
    if len(quant_flag.disabled_layers) > 0 and disable_flag:
        logger.debug("disable requant for layer [ %s ] in disabled_layers", previous_lname)
        return

    logger.debug("requant for layer [ %s ]"%previous_lname)
    graph.add(ReQuant(quant_flag, pre_lname=previous_lname))


class Pass(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(Pass, self).__init__(**kwargs)

        self.logger = logging.getLogger("log.quant.op.requant")
        self.logger.setLevel(quant_flag.log_level)

    def hybrid_forward(self, F, x):
        out = x
        if isinstance(x, nd.NDArray):
            self.logger.debug("quant %s with data=<%s,%s>, max=%s, min=%s",
                    "pass layer",
                    out.asnumpy().flatten()[0],
                    out.asnumpy().flatten()[0:49].max(),
                    out.max().asnumpy(),
                    out.min().asnumpy())
        return out

# def conv2d_quant(channels, kernel_size, stride, padding, use_bias, in_channels):
    # quant_conv_layer = nn.HybridSequential(prefix='')
    # quant_conv_layer.add(QuantOp())
    # quant_conv_layer.add(nn.Conv2D(channels, kernel_size, strides=stride,
                # padding=padding, use_bias=False))
    # quant_conv_layer.add(QuantOp())
    # if use_bias:
        # quant_conv_layer.add(BiasAddOp())


    # quant aware calibration
    # added_params_name, val_changed_params_name, deleted_params_name = [], [], []
    # def move_params(layer_name, previous_lout_sb):
    #     weight_name = layer_name.replace("_fwd_output", "_weight")
    #     assert weight_name not in qparams
    #     weight_quant_name = weight_name + "_quant"
    #     qparams[weight_name] = qparams[weight_quant_name]

    #     # set layer input shift_bits
    #     layer_input_sb_name = layer_name.replace("_fwd_output", "_input_shift_bits")
    #     assert previous_lout_sb != None
    #     qparams[layer_input_sb_name] = previous_lout_sb
    #     added_params_name.append(layer_input_sb_name)

    #     weight_shift_bits_name = weight_name + "_shift_bits"
    #     bias_name = layer_name.replace("_fwd_output", "_bias")
    #     bias_quant_names = [bias_name+"_quant", bias_name+"_shift_bits"]
    #     if bias_name in qparams:
    #         shift_bits = qparams[weight_shift_bits_name] + qparams[layer_input_sb_name]
    #         qparams[bias_quant_names[0]], qparams[bias_quant_names[1]] = \
    #             quant_helper(qparams[bias_name], shift_bits=shift_bits)
    #         qparams[bias_name] = qparams[bias_quant_names]
    #         added_params_name.extend(bias_quant_names)
    #         val_changed_params_name.append(bias_name)

    # layer_step_wise_prefix = "./data/params.resnet18.quant.tmp"
    # tmp_params_file = layer_step_wise_prefix

    # previous_lout_sb = None
    # for layer_name in outputs:
    #     # prepare name for quant model params
    #     move_params(layer_name,
    #             input_shift_bits if previous_lout_sb == None else previous_lout_sb)
    #     nd.save(tmp_params_file, qparams)

    #     # calculate calib output of quantize layer
    #     sym_block = gluon.SymbolBlock(layers[layer_name], [inputs])
    #     sym_block.load_parameters(tmp_params_file, ctx=ctx, ignore_extra=True)
    #     calib_res = sym_block.forward(image_data.as_in_context(ctx))

    #     # calculate requant shift_bits
    #     _, requant_shift_bits = quant_helper(calib_res)
    #     requant_sb_name = layer_name.replace("_fwd_output", "_requant_shift_bits")
    #     qparams[requant_sb_name] = requant_shift_bits

    #     # next layer input shift bits is the sum of input_shift_bits of the layer,
    #     # weight_shift_bits, and requant_shift_bits
    #     weight_sb_name = layer_name.replace("_fwd_output", "_weight") + "_shift_bits"
    #     linput_sb_name = layer_name.replace("_fwd_output", "_input_shift_bits")
    #     previous_lout_sb = qparams[weight_sb_name] + qparams[linput_sb_name]
    #     previous_lout_sb += qparams[requant_sb_name]

    # print ("[ added_params_name       ]: ", added_params_name)
    # print ("[ deleted_params_name     ]: ", deleted_params_name)

