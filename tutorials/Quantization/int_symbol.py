# -*- coding:utf-8 -*-
from layers import IntDense, IntConv
from CVMSymbol import *
# from mxboard import SummaryWriter
import mxnet as mx
import numpy as np
import logging
from mxnet import nd
from mxnet import autograd
import shutil

# shutil.rmtree('runs')
# writer = SummaryWriter(logdir='./runs')
#
logging.getLogger().setLevel(logging.DEBUG)
import argparse

parser = argparse.ArgumentParser(description='model quant')
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--is_quant_train', action='store_true')
args = parser.parse_args()

global batch
batch = 0

class IntBatch(mx.operator.CustomOp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=False, name=''):
        self.num_bit = nd.array([bit])
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def forward(self, is_train, req, in_data, out_data, aux):
        global batch
        batch += 1
        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        moving_mean = in_data[3]
        moving_var = in_data[4]
        new_gamma = in_data[5]
        new_beta = in_data[6]
        y_shift_bit = in_data[7]
        last_shift_bit = in_data[8]
        #print(batch)
        if batch % 20 == 0:
            writer.add_histogram(tag='batch1_input', values=x, bins=np.arange(-10, 10), global_step=batch)
        #if x.max() > 127 or x.min() < -128:
        #    print(x)
        y = out_data[0]
        if is_train:
            mean = nd.mean(x, axis=(0, 2, 3))
            var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

            quan_gamma = gamma / (nd.sqrt(var + self.eps))
            quan_beta = beta - mean * gamma / nd.sqrt(var + self.eps)
            # print(quan_gamma)
            quan_gamma = quan_gamma * (2 ** last_shift_bit)
            quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)

            y = nd.BatchNorm(x, gamma=nd.ones(shape=moving_var.shape), beta=nd.zeros(shape=moving_mean.shape), moving_mean=nd.zeros(shape=moving_mean.shape),
                             moving_var=nd.ones(shape=moving_var.shape), eps=1e-5,
                             momentum=self.momentum, fix_gamma=True, name=self.name)
            y, y_shift_bit = self.int_quantize(y)
            # print('train gamma', quan_gamma)
        else:
            # quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)
            y = nd.BatchNorm(x, gamma=nd.ones(shape=moving_var.shape), beta=new_beta,
                             moving_mean=nd.zeros(shape=moving_mean.shape),
                             moving_var=nd.ones(shape=moving_var.shape), eps=1e-5,
                             momentum=self.momentum, fix_gamma=True, name=self.name)
            # y, y_shift_bit = self.int_quantize(y)
            y = y * (2 ** y_shift_bit)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dgamma = in_grad[1]
        dbeta = in_grad[2]

        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        mean = in_data[3]
        var = in_data[4]
        new_gamma = in_data[5]
        new_beta = in_data[6]
        y_shift_bit = in_data[7]
        last_shift_bit = in_data[8]

        y = out_data[0]
        dy = out_grad[0]

        mean = nd.mean(x, axis=(0, 2, 3))
        var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

        quan_gamma = gamma / (nd.sqrt(var + self.eps))
        quan_beta = beta - mean * gamma / nd.sqrt(var + self.eps)

        # quan_gamma = nd.clip(nd.floor(nd.log2(quan_gamma)), a_min=-3, a_max=0)
        # quan_gamma = 2**(quan_gamma)
        quan_gamma = quan_gamma * (2 ** last_shift_bit)
        # quan_beta, beta_shift_bit = self.int_quantize(quan_beta)
        quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)
        x.attach_grad(), quan_gamma.attach_grad(), quan_beta.attach_grad()
        # print(quan_gamma)

        with autograd.record():
            y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=nd.zeros(shape=mean.shape),
                             moving_var=nd.ones(shape=var.shape), eps=self.eps,
                             momentum=self.momentum, fix_gamma=False, name=self.name)
            y, y_shift_bit = self.int_quantize(y)
        # print(quan_gamma)

        dx, dgamma, dbeta = autograd.grad(y, [x, quan_gamma, quan_beta], dy, retain_graph=True)

        self.assign(in_grad[0], req[0], dx / 2 ** y_shift_bit)
        self.assign(in_grad[1], req[0], dgamma / 2 ** (gamma_shift_bit + last_shift_bit))
        self.assign(in_grad[2], req[0], dbeta / 2 ** gamma_shift_bit)

        self.assign(in_data[5], req[0], quan_gamma)
        self.assign(in_data[6], req[0], quan_beta)
        self.assign(in_data[7], req[0], y_shift_bit)

    def quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor() * (1 / f)
            return y
        return x

    def int_quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor()
            y = nd.clip(y, a_min=-128, a_max=127)

            return y, frac_len
        return x, 0

    def int_quantize_double(self, x, w):
        max1 = nd.max(nd.abs(x))
        max2 = nd.max(nd.abs(w))
        if max1 > max2:
            max = max1
        else:
            max = max2
        if max != 0:
            int_len = (nd.ceil(nd.log2(max))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            int_x = ((x * f)).floor()
            int_w = ((w * f)).floor()
            return int_x, int_w, frac_len
        return x, w, 0


@mx.operator.register("batchnorm_int")
class IntBatchProp(mx.operator.CustomOpProp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
        super(IntBatchProp, self).__init__(True)
        self.num_bit = bit
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def list_arguments(self):
        return ['data', 'gamma', 'beta', 'moving_mean', 'moving_var', 'quan_gamma', 'quan_beta', 'y_shift_bit',
                'last_shift_bit']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        gamma_shape = [in_shapes[0][1]]
        beta_shape = [in_shapes[0][1]]
        moving_mean_shape = [in_shapes[0][1]]
        moving_var_shape = [in_shapes[0][1]]
        output_shape = in_shapes[0]
        return [data_shape, gamma_shape, beta_shape, moving_mean_shape,
                moving_var_shape, gamma_shape, beta_shape, [1], [1]], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return IntBatch(self.num_bit, self.eps, self.momentum, self.fix_gamma, self.name)


def load_checkpoint(prefix, epoch):
        """
        Load model checkpoint from file.
        :param prefix: Prefix of model name.
        :param epoch: Epoch number of model we would like to load.
        :return: (arg_params, aux_params)
        arg_params : dict of str to NDArray
                Model parameter, dict of name to NDArray of net's weights.
        aux_params : dict of str to NDArray
                Model parameter, dict of name to NDArray of net's auxiliary states.
        """
        save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
                tp, name = k.split(':', 1)
                if tp == 'arg':
                        arg_params[name] = v
                if tp == 'aux':
                        arg_params[name] = v
        return arg_params, aux_params

if __name__ == '__main__':

        batch_size = 64  # 定义每次处理数据的数量为64
        mnist = mx.test_utils.get_mnist()  # 使用内置的api读取mnist数据
        train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)  # 划分训练集和验证集，本代码仅用验证集做inference
        val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


        data = mx.symbol.Variable('data')

        #mxnet 原版网络
        if args.is_train and not args.is_quant_train:
                x= mx.sym.Convolution(data=data, kernel=(5,5), pad=(2,2), num_filter=32, name='conv1', no_bias=True)
                x= mx.sym.Activation(data=x, act_type="relu", name='relu1')
                #x = mx.sym.BatchNorm(data=x, name='batch1')
                x= mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool1')
                # second conv layer
                x= mx.sym.Convolution(data=x, kernel=(5,5), pad=(2,2), num_filter=64, name='conv2', no_bias=False)
                #x = mx.sym.BatchNorm(data=x, name='batch2')
                x= mx.sym.Activation(data=x, act_type="relu", name='relu2')
                x= mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool2')
                # first fullc layer
                x= mx.sym.Flatten(data=x, name='flatten')
                x= mx.symbol.FullyConnected(data=x, num_hidden=512, name='fc1')
                x= mx.sym.Activation(data=x, act_type="relu", name='relu3')
                # second fullc
                x= mx.sym.FullyConnected(data=x, num_hidden=10, name='fc2')
                # softmax loss
                lenet= mx.sym.SoftmaxOutput(data=x, name='softmax')
        else:
                x, last_shift_bit_1 = mx.symbol.Custom(data , name='conv1', op_type='intconv',
                                                        bit=7, num_filter=32, kernel=(5, 5),
                                                        pad=(2, 2), stride=(1, 1), no_bias=True, )
                x = mx.symbol.Activation(data=x, name='relu1', act_type="relu")
                #x = mx.symbol.Custom(data=x, last_shift_bit= last_shift_bit_1, name='batch1', op_type='batchnorm_int', bit=7)
                x = mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool1')
                x, last_shift_bit_2= mx.symbol.Custom(x, name='conv2', op_type='intconv', bit=7,
                                                        num_filter=64, kernel=(5, 5), stride=(1, 1),pad=(2, 2), no_bias=True,)
                #x = mx.symbol.Custom(data=x, last_shift_bit= last_shift_bit_2, name='batch2', op_type='batchnorm_int', bit=7)
                x= mx.sym.Activation(data=x, act_type="relu", name='relu2')
                x= mx.sym.Pooling(data=x, pool_type="max", kernel=(2,2), stride=(2,2), name='pool2')
                x= mx.sym.Flatten(data=x, name='flatten')
                x = mx.symbol.Custom(data=x, op_type='intdense', bit=7, num_hidden=512, name='fc1',)
                x= mx.sym.Activation(data=x, act_type="relu", name='relu3')
                x = mx.symbol.Custom(data=x, op_type='intdense', bit=7, num_hidden=10, name='fc2',)
                lenet = mx.sym.SoftmaxOutput(data=x, name='softmax')

        if args.is_train:
            mod = mx.mod.Module(lenet, context=mx.gpu(1))
            # mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
            mod.fit(train_data=train_iter,
                    eval_data=val_iter,
                    optimizer='adam',
                    optimizer_params={'learning_rate':1e-3 , },
                    num_epoch=1,
                    eval_metric='acc',
                    batch_end_callback=mx.callback.Speedometer(1, 10))
            mod.save_checkpoint('int_dense_conv_mnist_7bit', 20)
        elif args.is_quant_train:
            # print (lenet.tojson())
            mod = mx.mod.Module(lenet, data_names = ['data'], label_names = ['softmax_label'], context=mx.gpu(1))
            mod.fit(train_data=train_iter,
                    eval_data=val_iter,
                    optimizer='adam',
                    optimizer_params={'learning_rate':1e-3 , },
                    num_epoch=5,
                    eval_metric='acc',
                    initializer=mx.init.Mixed(['.*shift_bit', '.*'], [mx.init.Constant(7), mx.init.Uniform(0.1)]),
                    batch_end_callback=mx.callback.Speedometer(64, 100))
            mod.save_checkpoint('int_dense_conv_mnist_7bit_test', 1)
        else:
            # arg_params, aux_params = load_checkpoint('mnist', 20)
            arg_params, aux_params = load_checkpoint('int_dense_conv_mnist_7bit_test', 1)
            mod = mx.mod.Module(lenet, context=mx.gpu(1))
            mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
            mod.set_params(arg_params, aux_params, allow_missing=True, force_init=True, allow_extra=True)
            print (lenet.tojson())
        score = mod.score(val_iter, ['acc', 'loss'])  # 进行inference
        print("Accuracy score is ", (score[0]))  # 输出预测精度

