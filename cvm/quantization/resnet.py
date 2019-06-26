import mxnet as mx
from mxnet.gluon import HybridBlock, nn

import logging
from quant_utils import *
from quant_op import *


def _conv3x3(channels, stride, in_channels, use_bias=False):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=use_bias, in_channels=in_channels)


class BasicBlockV1Q(HybridBlock):
    def __init__(self, channels, stride, quant_flag, downsample=False,
            residual_sb_initializer='zeros', in_channels=0, prefix='', **kwargs):
        if quant_flag.calib_mode == CalibMode.NONE:
            super(BasicBlockV1Q, self).__init__(prefix=prefix, **kwargs)
        else:
            super(BasicBlockV1Q, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)

        self.body.add(_conv3x3(channels, 1, channels, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                        use_bias=quant_flag.is_fuse_bn, in_channels=in_channels))
            requant_helper(self.downsample, quant_flag)

            if not quant_flag.is_fuse_bn:
                self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.quant_flag = quant_flag
        self.logger = logging.getLogger("log.quant.op.residual.block")
        self.logger.setLevel(quant_flag.log_level)

        if self.quant_flag.calib_mode != CalibMode.NONE:
            self.shift_bits = self.params.get('requant_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.first_sb = self.params.get('first_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.second_sb = self.params.get('second_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

    def _alias(self):
        return '_plus'

    def hybrid_forward(self, F, x, shift_bits=None, first_sb=None, second_sb=None):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)


        if self.quant_flag.calib_mode != CalibMode.NONE:
            # self.logger.info("sb = %s, %s", first_sb.asnumpy(), second_sb.asnumpy())
            residual, _ = quant_helper(residual, shift_bits=first_sb, F=F,
                    logger=self.logger, msg="residual first param")
            x, _ = quant_helper(x, shift_bits=second_sb, F=F,
                    logger=self.logger, msg="residual second param")

        out = residual + x

        if self.quant_flag.calib_mode != CalibMode.NONE:
            # self.logger.info("sb = %s", shift_bits.asnumpy())
            out, _ = quant_helper(out, shift_bits=shift_bits, F=F,
                    logger=self.logger, msg=self.name)

        return F.Activation(out, act_type='relu')


class BottleneckV1Q(HybridBlock):
    def __init__(self, channels, stride, quant_flag, downsample=False,
            residual_sb_initializer='zeros', in_channels=0, prefix='', **kwargs):
        if quant_flag.calib_mode == CalibMode.NONE:
            super(BottleneckV1Q, self).__init__(prefix=prefix, **kwargs)
        else:
            super(BottleneckV1Q, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride,
                    use_bias=True))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)
        self.body.add(_conv3x3(channels//4, 1, channels//4, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                    use_bias=True))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=quant_flag.is_fuse_bn, in_channels=in_channels))
            requant_helper(self.downsample, quant_flag)

            if not quant_flag.is_fuse_bn:
                self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.quant_flag = quant_flag
        self.logger = logging.getLogger("log.quant.op.residual.block")
        self.logger.setLevel(quant_flag.log_level)

        if self.quant_flag.calib_mode != CalibMode.NONE:
            self.shift_bits = self.params.get('requant_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.first_sb = self.params.get('first_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.second_sb = self.params.get('second_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

    def _alias(self):
        return '_plus'

    def hybrid_forward(self, F, x, shift_bits=None, first_sb=None, second_sb=None):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)


        if self.quant_flag.calib_mode != CalibMode.NONE:
            residual, _ = quant_helper(residual, shift_bits=first_sb, F=F,
                    logger=self.logger, msg="residual first param")
            x, _ = quant_helper(x, shift_bits=second_sb, F=F,
                    logger=self.logger, msg="residual second param")

        out = residual + x

        if self.quant_flag.calib_mode != CalibMode.NONE:
            out, _ = quant_helper(out, shift_bits=shift_bits, F=F,
                    logger=self.logger, msg=self.name)

        return F.Activation(out, act_type='relu')

class ResNetV1Q(HybridBlock):
    def __init__(self, block, layers, channels, quant_flag, classes=1000, **kwargs):
        super(ResNetV1Q, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.quant_flag = quant_flag

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3,
                        use_bias=self.quant_flag.is_fuse_bn))
            requant_helper(self.features, quant_flag)

            if not self.quant_flag.is_fuse_bn:
                self.features.add(nn.BatchNorm())

            self.features.add(nn.Activation('relu'))
            requant_helper(self.features, quant_flag)

            self.features.add(nn.MaxPool2D(3, 2, 1))
            requant_helper(self.features, quant_flag)

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))

            # self.features.add(nn.GlobalAvgPool2D())
            self.features.add(GlobalAvgPool2D(quant_flag))
            requant_helper(self.features, quant_flag)

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Dense(classes, in_units=channels[-1]))
            requant_helper(self.output, quant_flag)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, self.quant_flag, channels != in_channels,
                        in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, self.quant_flag, False,
                            in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x

# def gluon_quant_resnet(quant_flag, batch_size=10,
#         iter_num=10, need_requant=False):
#     logger = logging.getLogger("log.quant.main.gluon")
#     logger.info("=== Model Quantazation ===")
# 
#     pass_name = "gluon.quant"
#     quant_symbol_file, quant_params_file = get_dump_fname(pass_name)
# 
#     if not os.path.exists(resnet.SYMBOL_FILE):
#         logger.info("save resnet symbol&params")
#         resnet.save_graph(mx.gpu())
# 
#     inputs_ext = {
#         'data': {
#             'shape': (batch_size, 3, 224, 244),
#         }
#     }
#     inputs = mx.sym.var('data')
#     ctx = mx.cpu(0)
# 
#     logger.info("load dataset")
#     data_iter = load_dataset(batch_size)
#     calib_data = data_iter.next()
# 
#     logger.info("quantization model")
#     tmp_params_file = quant_params_file + ".tmp"
#     if (not need_requant) and os.path.exists(tmp_params_file):
#         logger.debug("load quant params")
#         qparams = nd.load(tmp_params_file)
#     else:
#         qparams = qpass.fuse_bn_parameters(nd.load(resnet.PARAMS_FILE), quant_flag)
#         name_scope = "calib_"
#         scope_graph = nn.HybridSequential(prefix=name_scope)
#         with scope_graph.name_scope():
#             graph = resnet.load_quant_graph(QuantFlag(is_fuse_bn=True,
#                         calib_mode=CalibMode.NONE))
#             scope_graph.add(graph)
#         qparams = qpass.calibrate_parameters(scope_graph, qparams, ctx,
#                 calib_data.data[0], quant_flag, name_scope=name_scope)
#         nd.save(tmp_params_file, qparams)
# 
#     graph = resnet.load_quant_graph(quant_flag)
#     sym, qparams = graph(inputs), load_parameters(graph, qparams, ctx=ctx)
# 
#     sym, qparams = fold_cond_op(sym, qparams, {}, quant_flag)
#     # sym, qparams = mx_sym_rewrite(sym, qparams, quant_flag, inputs_ext)
#     # exit()
# 
#     nd.save(quant_params_file, qparams)
#     with open(quant_symbol_file, 'w') as fout:
#         fout.write(sym.tojson())
# 
#     logger.info("load quant/original model")
#     qsym_block = nn.SymbolBlock(sym, [inputs])
#     qsym_block.load_parameters(quant_params_file, ctx=ctx, ignore_extra=True)
# 
#     sym_block = resnet.load_graph(ctx)
# 
#     logger.info("calculate model accuracy")
#     qacc, acc, diff, total = 0, 0, 0, 0
#     for i in range(iter_num):
#         image_data = calib_data.data[0]
#         qimage_data, _ = quant_helper(image_data)
# 
#         res = sym_block.forward(image_data.as_in_context(ctx))
# 
#         if quant_flag.calib_mode == CalibMode.NONE:
#             qimage_data = image_data
#         qres = qsym_block.forward(qimage_data.as_in_context(ctx))
# 
#         assert res.shape == qres.shape
#         for idx in range(res.shape[0]):
#             res_label = res[idx].asnumpy().argmax()
#             qres_label = qres[idx].asnumpy().argmax()
#             image_label = calib_data.label[0][idx].asnumpy()
# 
#             diff += 0 if res_label == qres_label else 1
#             acc += 1 if res_label == image_label else 0
#             qacc += 1 if qres_label == image_label else 0
#             total += 1
# 
#         try:
#             calib_data = data_iter.next()
#         except:
#             exit()
# 
#         logger.info("Iteration: %5d | Accuracy: %.2f%% | Quant Acc: %.2f%%" +
#                 " | Difference: %.2f%% | Total Sample: %5d",
#                 i, 100.*acc/total, 100.*qacc/total, 100.*diff/total, total)
# def test_quant_model(batch_size=10, iter_num=10):
#     logger = logging.getLogger("log.test.mxnet")
#     logger.info("=== Log Test Mxnet ===")
# 
#     load_symbol_file, load_params_file = get_dump_fname("post.quant")
# 
#     ctx = mx.gpu(1)
#     inputs = mx.sym.var("data")
# 
#     sym = mx.sym.load(load_symbol_file)
# 
#     data_iter = load_dataset(batch_size)
#     calib_data = data_iter.next()
# 
#     graph = nn.SymbolBlock(sym, [inputs])
#     # print ('graph params:', sorted(list(graph.collect_params().keys())))
#     # print ('params:', sorted(list(params.keys())))
#     # params_dict = load_parameters(graph, params, ctx=ctx)
# 
#     graph.load_parameters(load_params_file, ctx=ctx)
# 
#     qacc, total = 0, 0
#     for i in range(iter_num):
#         qimage_data, _ = quant_helper(calib_data.data[0])
# 
#         # params['data'] = qimage_data
#         # graph = sym.bind(ctx, params)
#         qres = graph.forward(qimage_data.as_in_context(ctx))
# 
#         for idx in range(qres.shape[0]):
#             qres_label = qres[idx].asnumpy().argmax()
#             image_label = calib_data.label[0][idx].asnumpy()
# 
#             qacc += 1 if qres_label == image_label else 0
#             total += 1
# 
#         try:
#             calib_data = data_iter.next()
#         except:
#             exit()
# 
#         logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
#                 i, 100.*qacc/total, total)
# def test_nnvm_load(batch_size=10, iter_num=10):
#     logger = logging.getLogger("log.test.nnvm")
#     logger.info("=== Log Test NNVM ===")
# 
#     target = "llvm -mcpu=core-avx2 -libs=cvm"
#     ctx = tvm.context(target, 0)
# 
#     load_symbol_fname, load_params_fname = get_dump_fname("gluon.quant")
# 
#     in_shape = (batch_size, 3, 224, 224)
#     data_iter = load_dataset(batch_size)
#     calib_data = data_iter.next()
# 
#     params = nd.load(load_params_fname)
# 
#     sym = mx.sym.load(load_symbol_fname)
#     nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
# 
#     nnvm_sym, params = quant_realize(nnvm_sym, params, {}, quant_flag)
#     # , ctx=tvm.context("opencl", 0))
# 
#     nnvm_graph = nnvm.graph.create(nnvm_sym)
#     save_symbol_file, _ = get_dump_fname("nnvm.realize")
#     with open(save_symbol_file, "w") as fout:
#        fout.write(nnvm_graph.ir())
#     use_dtype = "int32"
#     for key, value in list(params.items()):
#         params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), ctx)
#     with nnvm.compiler.build_config(opt_level=0):
# #, add_pass=["PrecomputePrune"]):
#         deploy_graph, lib, params = nnvm.compiler.build(
#             nnvm_sym, target=target,
#             shape={"data": in_shape},
#             params=params, dtype=use_dtype)
#         ret = deploy_graph.apply('SaveJSON')
#         graph_str = ret.json_attr('json')
# 
#         with open("graph_str.log", "w") as fout:
#             fout.write(graph_str)
#         with open("deploy.log", "w") as fout:
#             fout.write(deploy_graph.ir())
# 
#     module = graph_runtime.create(deploy_graph, lib, ctx)
#     param_bytes = nnvm.compiler.save_param_dict(params)
#     module.set_input(**params)
# 
#     qacc, total = 0, 0
#     for i in range(iter_num):
#         qimage_data, _ = quant_helper(calib_data.data[0])
#         qimage_data = tvm.nd.array(qimage_data.asnumpy(), ctx)
# 
#         module.run(data=qimage_data.asnumpy())
#         qres = module.get_output(0).asnumpy()
# 
#         for idx in range(qres.shape[0]):
#             qres_label = qres[idx].argmax()
#             image_label = calib_data.label[0][idx].asnumpy()
# 
#             qacc += 1 if qres_label == image_label else 0
#             total += 1
# 
#         try:
#             calib_data = data_iter.next()
#         except:
#             exit()
# 
#         logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
#                 i, 100.*qacc/total, total)

