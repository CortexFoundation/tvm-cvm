import sys
from os import path
import configparser
import logging
import numpy as np

import mxnet as mx

from transformer import Model
from gluon_zoo import save_model
import dataset as ds
import sim_quant_helper as sim
import utils
import sym_utils as sutils

def set_batch(input_shape, batch):
    return [batch if s == -1 else s for s in input_shape]

def batch_axis(input_shape):
    idx = [i for i, s in enumerate(input_shape) if s == -1]
    assert len(idx) == 1
    return idx[0]

def _check(expression, section, option, message='Not a valid value'):
    assert expression, message + '.\noption `%s` in section `%s`' \
        % (option, section)

NoneType = object()

def _get_path(config, section, option, is_dir=False, dpath=NoneType):
    pth_ = _get_val(config, section, option, dval=dpath)
    pth = path.abspath(path.expanduser(pth_))
    if is_dir:
        _check(path.isdir(pth), section, option,
               message='Not a valid dir `%s`' % pth_)
        if not path.exists(pth):
            path.makedirs(pth)
    else:
        _check(path.exists(pth), section, option,
               message='File `%s` not found' % pth_)
    return pth

def _get_ctx(config, section, dctx=mx.cpu()):
    contex = dctx
    device_type = _get_val(config, section, 'Device_type', dval='cpu')
    _check(device_type in ['', 'gpu', 'cpu'], section, 'Device_type',
           message='Only support `gpu`, `cpu` and null value')
    if device_type == 'gpu':
        device_ids = _get_val(
            config, section, 'Device_ids', dtype='[int]')
        contex = mx.gpu(device_ids[0]) if len(device_ids) == 1 \
              else [mx.gpu(i) for i in device_ids]
        if section == 'CALIBRATION':
            _check(type(contex).__name__ != 'list', section, 'Device_ids',
                   message='`Device_ids` should be an integer in Calibration')
    else:
        device_ids = _get_val(config, section, 'Device_ids', dval='')
        _check(device_ids == '', section, 'Device_ids',
               message='`Device_ids` should be null given `cpu` device type')
    return contex

def _get_val(config, section, option, dtype='str', dval=NoneType):
    val_ = config[section][option]
    if val_ == '':
        _check(dval != NoneType, section, option,
               message="Please specify the value")
        val = dval
    elif dtype in ['str', 'int', 'tuple', 'float']:
        val = _cast_val(section, option, val_, dtype=dtype)
    elif dtype.startswith('['):
        etype=dtype.replace('[', '').replace(']', '')
        val = [_cast_val(section, option, x.strip(), dtype=etype) \
               for x in val_.split(',')]
    elif dtype.startswith('{'):
        etypes = dtype.replace('{', '').replace('}', '').split(':')
        items = val_.split(',')
        val = {}
        for item in items:
            entries_ = [it.strip() for it in item.split(':')]
            _check(len(entries_) == len(etypes), section, option,
                   message="Dict level not consistent")
            entries = [_cast_val(section, option, entry_, dtype=etypes[level]) \
                for level, entry_ in enumerate(entries_)]
            cur = val
            for level, entry in enumerate(entries[:-2]):
                cur[entry] = {} if entry not in cur else cur[entry]
                cur = cur[entry]
            _check(entries[-2] not in cur, section, option,
                   message="Duplicate key `%s`" % entries[:-1])
            cur[entries[-2]] = entries[-1]
    return val

def _cast_val(section, option, val_, dtype='str'):
    if dtype == 'str':
        val = val_
    elif dtype in ['int', 'tuple', 'float']:
        try:
            val = float(eval(val_)) if dtype == 'float' else eval(val_)
        except SyntaxError:
            print("Not a valid value, " + \
                  "option `%s` in section `%s`" % (option, section))
            sys.exit(0)
        if dtype == 'int':
            _check(type(val).__name__ == dtype, section, option,
                   message="Only support integer value")
    return val

def _load_fname(directory, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    return utils.extend_fname(directory+suffix, with_ext)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please enter 2 python arguments."
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    fileName = path.basename(cfgPath)
    absCfgPath = path.join(baseDir, fileName)

    cfg = configparser.ConfigParser()
    cfg.read(absCfgPath)

    # default
    sec = 'DEFAULT'
    verbosity = _get_val(cfg, sec, 'Verbosity',
                         dtype='int', dval=logging.NOTSET)
    utils.log_init(level=verbosity)
    logger = logging.getLogger("log.main")
    default_dir = path.expanduser("~/tvm-cvm/data")
    assert path.exists(default_dir), \
        "Please create the folder `data` in the working directory first"
    model_dir = _get_val(cfg, sec, 'Model_dir', dval=default_dir)
    model_name = _get_val(cfg, sec, 'Model_name')
    sym_path, prm_path = _load_fname(path.join(model_dir, model_name))
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name, sym_path=sym_path, prm_path=prm_path)
    model_ctx = _get_ctx(cfg, sec)
    input_shape = _get_val(cfg, sec, 'Input_shape', dtype='tuple')

    # prepare
    sec = 'PREPARE'
    model = Model.load(sym_path, prm_path)
    batch = _get_val(cfg, sec, 'Batch', dtype='int', dval=1)
    model.prepare(set_batch(input_shape, 16))
    # dump_dir = _get_path(
        # cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    # sym_file, prm_file = _load_fname(dump_dir, suffix='prepare')
    # model.save(sym_file, prm_file)
    logger.info("Prepare finihed")

    # split model
    sec = 'SPLIT_MODEL'
    keys = _get_val(cfg, sec, 'Keys', dtype='[str]', dval='')
    dump_dir = _get_path(
        cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    if keys == '':
        mrt = model.get_mrt()
    else:
        base, top = model.split(keys)
        mrt = base.get_mrt()
        sym_file, prm_file = _load_fname(dump_dir, suffix='top')
        top.save(sym_file, prm_file)
        sym_file, prm_file = _load_fname(dump_dir, suffix='base')
        base.save(sym_file, prm_file)
        logger.info("Split model finihed")

    # calibration
    sec = 'CALIBRATION'
    calibrate_num = _get_val(cfg, sec, 'Calibrate_num', dtype='int')
    lambd = _get_val(cfg, sec, 'Lambda', dtype='float', dval=None)
    ds_name = _get_val(cfg, sec, 'dataset')
    batch = _get_val(cfg, sec, 'Batch', dtype='int', dval=batch)
    shp = set_batch(input_shape, batch)
    dataset = ds.DS_REG[ds_name](shp)
    data_iter_func = dataset.iter_func()
    '''
    if dataset in ['voc', 'imagenet', 'mnist', \
                   'quickdraw', 'cifar10']:
        _check(input_shape[2] == input_shape[3], 'PREPARE', 'Input_shape',
               message='Inconsistent input size')
        batch_size, num_channel, input_size, _ = input_shape
        data_iter_func = ds.data_iter(
            dataset, batch_size, input_size=input_size)
    elif dataset in ['trec']:
        _check(input_shape[0] == 38, 'PREPARE', 'Input_shape',
               message='Invalid input shape for Trec')
        batch_size = input_shape[1]
        data_iter_func = ds.load_trec(batch_size)
    else:
        _check(False, sec, 'Dataset')
    '''
    ctx = _get_ctx(cfg, sec, dctx=model_ctx)
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        th_dict = mrt.calibrate(lambd=lambd, ctx=ctx)
    dump_dir = _get_path(
        cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    _, _, ext_file = _load_fname(
        dump_dir, suffix='th_dict', with_ext=True)
    # sim.save_ext(ext_file, th_dict)
    logger.info("Calibration finihed")

    # quantization
    sec = 'QUANTIZATION'
    input_precision = _get_val(
        cfg, sec, 'Input_precision', dtype='int', dval=None)
    if input_precision is not None:
        mrt.set_input_prec(input_precision)
    output_precision = _get_val(
        cfg, sec, 'Output_precision', dtype='int', dval=None)
    if output_precision is not None:
        mrt.set_output_prec(output_precision)
    ctx = _get_ctx(cfg, sec, dctx=model_ctx)
    softmax_lambd = _get_val(
        cfg, sec, 'softmax_lambd', dtype='float', dval=None)
    if softmax_lambd is not None:
        mrt.set_softmax_lambd(softmax_lambd)
    shift_bits = _get_val(
        cfg, sec, 'shift_bits', dtype='int', dval=None)
    if shift_bits is not None:
        mrt.set_shift_bits(shift_bits)
    thresholds = _get_val(
        cfg, sec, 'Thresholds', dtype='{str:float}', dval=None)
    if thresholds is not None:
        for name, threshold in thresholds.items():
            mrt.set_threshold(name, threshold)
    mrt.quantize()
    qmodel = mrt.current_model
    dump_dir = _get_path(
        cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    mrt.save(model_name+'base.quantize', datadir=dump_dir)
    logger.info("Quantization finihed")

    # merge_model
    sec = 'MERGE_MODEL'
    dump_dir = _get_path(
        cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    if keys != '':
        model_merger = Model.merger(qmodel, top, mrt.get_maps())
        base_oscales = mrt.get_output_scales()
        attribute_deps = _get_val(
            cfg, sec, 'Attribute_deps', dtype='{str:str:int}')

        def mergefunc(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
            childs, attr = sutils.sym_iter(
                node.get_children()), node.list_attr()
            if op_name in attribute_deps:
                attr_deps = attribute_deps[op_name]
                for attr_name, entry in attr_deps.items():
                    val = sutils.get_attr(attr, attr_name, 0)
                    attr[attr_name] = int(val*base_oscales[entry])
                node = sutils.get_mxnet_op(op_name)(
                    *childs, **attr, name=name)
            return node

        qmodel = model_merger.merge(callback=mergefunc)
        oscale_maps = _get_val(cfg, sec, 'Oscale_maps', dtype='{str:str}')
        oscales = model_merger.get_output_scales(base_oscales, oscale_maps)
        sym_file, prm_file = _load_fname(dump_dir, suffix='all.quantize')
        qmodel.save(sym_file, prm_file)
        logger.info("Merge model finihed")

    # evaluation
    sec = 'EVALUATION'
    iter_num = _get_val(cfg, sec, 'Iter_num', dtype='int', dval=0)
    batch = _get_val(cfg, sec, 'Batch', dtype='int', dval=batch)
    ctx = _get_ctx(cfg, sec, dctx=model_ctx)
    org_model = Model.load(sym_path, prm_path)
    graph = org_model.to_graph(ctx=ctx)
    dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
    data_iter_func = dataset.iter_func()
    metric = dataset.metrics()

    def evalfunc(data, label):
        outs = graph(data.as_in_context(ctx))
        acc = dataset.validate(metric, outs, label)
        return acc

    qgraph = qmodel.to_graph(ctx=ctx)
    qmetric = dataset.metrics()

    def quantize(data, label):
        data = sim.load_real_data(data, 'data', mrt.get_inputs_ext())
        outs = qgraph(data.as_in_context(ctx))
        outs = [(t / oscales[i]) for i, t in enumerate(outs)]
        acc = dataset.validate(qmetric, outs, label)
        return acc

    if iter_num > 0:
        utils.multi_validate(evalfunc, data_iter_func, quantize,
                             iter_num=iter_num,
                             logger=logging.getLogger('mrt.validate'))


    # compilation
    sec = 'COMPILATION'
    dump_dir = _get_path(
        cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    batch = _get_val(cfg, sec, 'Batch', dtype='int', dval=batch)
    model_name_tfm = model_name + "_tfm"
    qmodel.to_cvm(model_name_tfm, datadir=dump_dir,
                  input_shape=set_batch(input_shape, batch))

    dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
    dump_data, _ = dataset.iter_func()()
    dump_data = sim.load_real_data(
        dump_data.astype("float64"), 'data', mrt.get_inputs_ext())
    np.save(path.join(dump_dir, model_name_tfm, "data.npy"),
            dump_data.astype('int8').asnumpy())
    logger.info("Compilation finihed")

