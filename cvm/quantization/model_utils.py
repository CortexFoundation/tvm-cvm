import mxnet as mx
from tfm_pass import infer_shape

def load_model(model_name, sym_path, prm_path, ctx, inputs_qext=None):
    inputs = [mx.sym.var('data')]
    sym, params = mx.sym.load(sym_path), nd.load(prm_path)
    net = gluon.nn.SymbolBlock(sym, inputs)
    nparams = params if inputs_qext else \
            convert_params_dtype(params, src_dtypes="float64",
                                 dest_dtype="float32")
    utils.load_parameters(net, nparams, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def model_func(data, label):
        data = sim.load_real_data(data.astype("float64"), 'data', inputs_qext) \
               if inputs_qext else data
        data = gluon.utils.split_and_load(data, ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        res = [net.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)
    return model_func

def validate_model(sym_path, prm_path, ctx, num_channel=3,
                   input_size=224, batch_size=16, iter_num=10,
                   ds_name='imagenet', from_scratch=0, lambd=None,
                   dump_model=False):
    from gluon_zoo import save_model

    flag = [False]*from_scratch + [True]*(2-from_scratch)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)
    input_shape = (batch_size, num_channel, input_size, input_size)
    logger = logging.getLogger("log.validate.%s"%model_name)

    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name)
    sym, params = mx.sym.load(sym_path), mx.nd.load(prm_path)

    print(collect_op_names(sym, params))

    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, _ = data_iter_func()

    # prepare
    mrt = MRT(sym, params, input_shape)
    mrt.set_data(data)

    # calibrate
    prefix = path.join(model_dir, model_name+'.mrt.dict')
    _, _, dump_ext = utils.extend_fname(prefix, True)
    if flag[0]:
        th_dict = mrt.calibrate(lambd=lambd)
        sim.save_ext(dump_ext, th_dict)
    else:
        (th_dict,) = sim.load_ext(dump_ext)
        mrt.set_th_dict(th_dict)

    mrt.set_input_prec(8)
    mrt.set_output_prec(8)

    # quantize, get: qsym, qprm, inputs_qext
    qsym, qprm, inputs_qext = None, None, None
    prefix = path.join(model_dir, model_name+'.mrt.quantize')
    qsym_path, qprm_path, qext_path = utils.extend_fname(prefix, True)
    if flag[1]:
        qsym, qprm, inputs_qext = mrt.quantize()
        open(path.expanduser(qsym_path), 'w').write(qsym.tojson())
        nd.save(qprm_path, qprm)
        sim.save_ext(qext_path, inputs_qext)
    else:
        qsym, qprm = mx.sym.load(qsym_path), nd.load(qprm_path)
        (inputs_qext, ) = sim.load_ext(qext_path)

    # dump model
    if dump_model:
        datadir = "/data/ryt"
        model_name = model_name + "_tfm"
        dump_shape = (1, num_channel, input_size, input_size)
        compile_to_cvm(qsym, qprm, model_name, datadir=datadir,
                       input_shape=dump_shape)
        data = data[0].reshape(dump_shape)
        data = sim.load_real_data(data.astype("float64"), 'data', inputs_qext)
        np.save(datadir+"/"+model_name+"/data.npy", data.astype('int8').asnumpy())
        sys.exit(0)

    # validate
    org_model = load_model(model_name, sym_path, prm_path, ctx)
    cvm_quantize = load_model(model_name, qsym_path, qprm_path, ctx, \
            inputs_qext=inputs_qext)

    utils.multi_validate(org_model, data_iter_func, cvm_quantize,
                         iter_num=iter_num,
                         logger=logging.getLogger('mrt.validate'))
    logger.info("test %s finished.", model_name)
