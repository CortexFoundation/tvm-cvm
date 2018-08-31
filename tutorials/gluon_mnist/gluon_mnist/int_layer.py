# -*- encode: utf8 -*-
from mxnet import gluon
from mxnet.gluon.parameter import ParameterDict
from functional import *
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon import HybridBlock
from mxnet import symbol
from mxnet.base import numeric_types

#对输入的input进行int化操作
def weight_Quan(F, input, acti_bit, exponent, one, zero):
    input_max = F.max(F.abs(input))
    int_len = F.where(input_max == zero, zero.astype('float32'), (F.ceil(F.log2(input_max))).astype('float32'))
    frac_len = (acti_bit).__sub__(int_len)
    f = (exponent**(frac_len)).astype('float32')
    activation_quan = (F.broadcast_mul(input, f)).round()
    return activation_quan

#对两个输入的input同时进行int化操作
def weight_Quan_double(F, input1, input2, acti_bit, exponent, one, zero):
    input1_max = F.max(F.abs(input1))
    input2_max = F.max(F.abs(input2))
    int_len1 = F.where(input1_max == zero, zero.astype('float32'), (F.ceil(F.log2(input1_max))).astype('float32'))
    int_len2 = F.where(input2_max == zero, zero.astype('float32'), (F.ceil(F.log2(input2_max))).astype('float32'))

    int_len = F.max(int_len1, int_len2)
    frac_len = (acti_bit).__sub__(int_len)
    f = (exponent**(frac_len)).astype('float32')
    activation_quan1 = (F.broadcast_mul(input1, f)).round()
    activation_quan2 = (F.broadcast_mul(input2, f)).round()
    return activation_quan1, activation_quan2
#def int_two_param(F, input1, input2, acti_bit, exponent, one):

class Activation_int(nn.HybridBlock):

#	参数：acti_bit 指定的int位宽
#		  exponet： 常数2
#		  one：常数1
#		  zero： 常数0
#	用途：将激活后的值限定在8bit中间

    def __init__(self, acti_bit, valid=False, act_type='relu'):
        super(Activation_int, self).__init__()
        self.valid = valid
        # To keep a sign bit, stay consistent with the weight quantize
        self.act_type = act_type
        self.acti_bit = self.params.get_constant('acti_bit', [acti_bit])
        self.exponent = self.params.get_constant('exponent', [2])
        self.one = self.params.get_constant('one', [1])
        self.zero = self.params.get_constant('zero', [0])

        self._kwargs ={'acti_bit': acti_bit, 'acti_type': act_type}

    def hybrid_forward(self, F, input, acti_bit, exponent, one, zero):
        if self.valid:
            #将输入的input进行int化
            activation_quan = weight_Quan(F, input, acti_bit, exponent, one, zero)
	    #print(activation_quan.max())
            #print(activation_quan)
            #将int化后的input通过制定的激活层
            return F.Activation(activation_quan, self.act_type)
        else:
            return F.Activation(input, self.act_type)

    def __repr__(self):
        s = '{name}({_act_type}, acti_bit={acti_bit})'
        return s.format(name=self.__class__.__name__, _act_type = self._kwargs['acti_type'], acti_bit=self._kwargs['acti_bit'],)


class Dense(nn.HybridBlock):
#	参数：acti_bit 指定的int位宽
#		  exponet： 常数2
#		  one：常数1
#		  zero： 常数0
#		  weight：浮点数（量级在0.01）左右
#		  bias：浮点数（量级在0.01）左右
#		  int_weight:整形化处理后的数
#		  int_bias:整形化处理后的数

    def __init__(self, units, activation=None, use_bias=True, flatten=True, weight_initializer=None, bias_initializer='normal', in_units=0, valid=True, acti_bit=8, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.acti_bit = self.params.get_constant('acti_bit', [acti_bit])
        self.exponent = self.params.get_constant('exponent', [2])
        self.one = self.params.get_constant('one', [1])
        self.zero = self.params.get_constant('zero', [0])

        self._kwargs ={'acti_bit': acti_bit}
        self.valid = valid
        self._flatten = flatten

        with self.name_scope():
            self._units = units
            self._in_units = in_units
	    #创建参数
            self.weight = self.params.get('weight', shape=(units, in_units), init=weight_initializer,)
            self.int_weight = self.params.get('int_weight', shape=(units, in_units), init=weight_initializer,)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,), init=bias_initializer,)
                self.int_bias = self.params.get('int_bias', shape=(units,), init=bias_initializer,)
            else:
                self.bias = None

            if activation is not None:
                self.act = Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, int_weight, acti_bit, exponent, one, zero, bias=None, int_bias=None):
        #weight = weight_Quan(F, weight, acti_bit, exponent, one)
        quan_x = weight_Quan(F, x, acti_bit, exponent, one, zero)
        quan_weight =  weight_Quan(F, weight, acti_bit, exponent, one, zero)
        #下面这一行的目的是将整形化后的参数写入到int_weight中
        #self.int_weight.set_data(quan_weight)
        if bias is not None:
            quan_bias =  weight_Quan(F, bias, acti_bit, exponent, one, zero)
            #self.int_bias.set_data(quan_bias)
            #exit(0)
            #bias = weight_Quan(F, bias, acti_bit, exponent, one)
            out = F.FullyConnected(quan_x, quan_weight, quan_bias, no_bias=bias is None, num_hidden=self._units, flatten=self._flatten, name='fwd')
        else:
            out = F.FullyConnected(x, weight, no_bias=bias is None, num_hidden=self._units, flatten=self._flatten, name='fwd')
        return out

    def __repr__(self):
        s = '{name}({layout}, {act}, acti_bit={acti_bit})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__, act=self.act if self.act else 'linear',
                layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                acti_bit = str(self._kwargs['acti_bit']))


#原版mxnet中conv的统一父类
class _Conv(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, dilation,
            groups, layout, in_channels=0, activation=None, use_bias=True,
            weight_initializer=None, bias_initializer='ones',
            op_name='Convolution', adj=None, prefix=None, params=None):
        super(_Conv, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
        if isinstance(strides, numeric_types):
            strides = (strides,)*len(kernel_size)
        if isinstance(padding, numeric_types):
            padding = (padding,)*len(kernel_size)
        if isinstance(dilation, numeric_types):
            dilation = (dilation,)*len(kernel_size)
        self._op_name = op_name
        self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}
        if adj is not None:
            self._kwargs['adj'] = adj

        dshape = [0]*(len(kernel_size) + 2)
        dshape[layout.find('N')] = 1
        dshape[layout.find('C')] = in_channels
        #wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
        self.weight = self.params.get('weight', shape=(channels, in_channels, *kernel_size),
                init=weight_initializer, allow_deferred_init=True)

        self.int_weight = self.params.get('int_weight', shape=(channels, in_channels, *kernel_size),
                init=weight_initializer, allow_deferred_init=True)
        if use_bias:
            self.bias = self.params.get('bias', shape=(channels,), init=bias_initializer, allow_deferred_init=True)
            self.int_bias = self.params.get('int_bias', shape=(channels,), init=bias_initializer, allow_deferred_init=True)
        else:
            self.bias = None

        if activation is not None:
            self.act = Activation(activation, prefix=activation+'_')
        else:
            self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            act = getattr(F, self._op_name)(x, weight, name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, weight, bias, name='fwd', **self._kwargs)
            if self.act is not None:
                act = self.act(act)
                return act

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                **self._kwargs)

class Convolution_int(_Conv):
#	参数：acti_bit 指定的int位宽
#		  exponet： 常数2
#		  one：常数1
#		  zero： 常数0
#		  weight：浮点数（量级在0.01）左右
#		  bias：浮点数（量级在0.01）左右
#		  int_weight:整形化处理后的数
#		  int_bias:整形化处理后的数
#		这里的卷积操作暂时不支持use_bias = True

    def __init__(self, acti_bit, channels, kernel_size, strides=(1, 1), padding=(0, 0),
            dilation=(1, 1), groups=1, layout='NCHW',
            activation=None, use_bias=True, weight_initializer=None,
            bias_initializer='ones', in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(Convolution_int, self).__init__(
                channels, kernel_size, strides, padding, dilation, groups, layout,
                in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
        self.acti_bit = self.params.get_constant('acti_bit', [acti_bit])
        self.exponent = self.params.get_constant('exponent', [2])
        self.one = self.params.get_constant('one', [1])
        self.zero = self.params.get_constant('zero', [0])

    def hybrid_forward(self, F, x, weight, int_weight, acti_bit, exponent, one, zero, bias=None, int_bias=None):
        print ('F: ', F)
        quan_x = weight_Quan(F, x, acti_bit, exponent, one, zero)
        quan_weight =  weight_Quan(F, weight, acti_bit, exponent, one, zero)
        #self.int_weight.set_data(quan_weight)

        if bias is None:
            return F.Convolution(quan_x, quan_weight, name='fwd', **self._kwargs)

        else:
            #print(weight, bias)
            int_weight, int_bias = weight_Quan_double(F, weight, bias, acti_bit, exponent, one, zero)
            #self.int_weight.set_data(quan_weight)
            #self.int_bias.set_data(int_bias)
            #print(weight)
            #print(bias)
            return F.Convolution(quan_x, quan_weight, int_bias, name='fwd', **self._kwargs)
