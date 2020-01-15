import sys
from os import path
import configparser
import logging

import mxnet as mx

from transformer import Model, MRT
import dataset as ds
import sim_quant_helper as sim
import utils

def _check(exp, sec, opt, message='Not a valid value'):
    assert exp, message + '.\noption `%s` in section `%s`' % (opt, sec)

NoneType = object()

def _get_path(config, sec, opt, is_dir=False, dpath=NoneType):
    pth_ = _get_val(config, sec, opt, dval=dpath)
    pth = path.abspath(path.expanduser(pth_))
    if is_dir:
        _check(path.isdir(pth), sec, opt,
               message='Not a valid dir `%s`' % pth_)
        if not path.exists(pth):
            path.makedirs(pth)
    else:
        _check(path.exists(pth), sec, opt,
               message='File `%s` not found' % pth_)
    return pth

def _get_ctx(config, sec, dctx=mx.cpu()):
    ctx = dctx
    device_type = _get_val(config, sec, 'Device_type', dval='cpu')
    _check(device_type in ['', 'gpu', 'cpu'], sec, 'Device_type',
           message='Only support `gpu`, `cpu` and null value')
    if device_type == 'gpu':
        device_ids = _get_val(config, sec, 'Device_ids',
                              dtype='list', dtype1='int')
        ctx = mx.gpu(device_ids[0]) if len(device_ids) == 1 \
              else [mx.gpu(i) for i in device_ids]
        if sec == 'CALIBRATION':
            _check(type(ctx).__name__ != 'list', sec, 'Device_ids',
                   message='`Device_ids` should be an integer in Calibration')
    else:
        device_ids = _get_val(config, sec, 'Device_ids', dval='')
        print(device_ids)
        _check(device_ids == '', sec, 'Device_ids',
               message='`Device_ids` should be null given `cpu` device type')
    return ctx

def _get_val(config, sec, opt, dtype='str', dval=NoneType,
             dtype1='str', dtype2='str'):
    val_ = config[sec][opt]
    if val_ == '':
        _check(dval != NoneType, sec, opt,
               message="Please specify the value")
        val = dval
    elif dtype in ['str', 'int', 'tuple', 'float']:
        val = _cast_val(sec, opt, val_, dtype=dtype)
    elif dtype == 'list':
        val = [_cast_val(sec, opt, x.strip(), dtype=dtype1) \
               for x in val_.split(',')]
    elif dtype == 'dict':
        val = {}
        for x in val_.split(','):
            k_, v_ = x.split(':')
            k = _cast_val(sec, opt, k_.strip(), dtype=dtype1)
            _check(k not in val, sec, opt,
                   message="Duplicate key `%s`" % k_.strip())
            val[k] = _cast_val(sec, opt, v_.strip(), dtype=dtype2)
    # else:
    #     _check(False, sec, opt, message="Unknown data type")
    return val

def _cast_val(sec, opt, val_, dtype='str'):
    if dtype == 'str':
        val = val_
    elif dtype in ['int', 'tuple', 'float']:
        try:
            val = float(eval(val_)) if dtype == 'float' else eval(val_)
        except SyntaxError:
            print("Not a valid value, " + \
                  "option `%s` in section `%s`" % (opt, sec))
            sys.exit(0)
        if dtype == 'int':
            _check(type(val).__name__ == dtype, sec, opt,
                   message="Only support integer value")
    return val

def _load_fname(directory, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    return utils.extend_fname(directory+suffix, with_ext)

if __name__ == "__main__":
    utils.log_init()
    logger = logging.getLogger("log.main")

    assert len(sys.argv) == 2, "Please enter 2 python arguments."
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    fileName = path.basename(cfgPath)
    absCfgPath = path.join(baseDir, fileName)

    config = configparser.ConfigParser()
    config.read(absCfgPath)
    print([(k, v) for k, v in config['TEST'].items()])
    exit()

    # default
    sec = 'DEFAULT'
    sym_path = _get_path(config, sec, 'Symbol')
    prm_path = _get_path(config, sec, 'Params')
    model_dir = path.dirname(sym_path)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_ctx = _get_ctx(config, sec)
    org_model = Model.load(sym_path, prm_path)

    # prepare
    sec = 'PREPARE'
    model = Model.load(sym_path, prm_path)
    input_shape = _get_val(config, sec, 'Input_shape', dtype='tuple')
    model.prepare(input_shape)
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    sym_file, prm_file = _load_fname(dump_dir, suffix='prepare')
    model.save(sym_file, prm_file)
    logger.info("Prepare finihed")

    # split model
    sec = 'SPLIT_MODEL'
    keys = _get_val(config, sec, 'Keys', dtype='list', dval='')
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
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
    calibrate_num = _get_val(config, sec, 'Calibrate_num', dtype='int')
    lambd = _get_val(config, sec, 'Lambda', dtype='float', dval=None)
    dataset = _get_val(config, sec, 'dataset')
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
    ctx = _get_ctx(config, sec, dctx=model_ctx)
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        th_dict = mrt.calibrate(lambd=lambd, ctx=ctx)
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    _, _, ext_file = _load_fname(
        dump_dir, suffix='th_dict', with_ext=True)
    sim.save_ext(ext_file, th_dict)
    logger.info("Calibration finihed")

    # quantization
    sec = 'QUANTIZATION'
    input_precision = _get_val(
        config, sec, 'Input_precision', dtype='int', dval=None)
    if input_precision is not None:
        mrt.set_input_prec(input_precision)
    output_precision = _get_val(
        config, sec, 'Output_precision', dtype='int', dval=None)
    if output_precision is not None:
        mrt.set_output_prec(output_precision)
    ctx = _get_ctx(config, sec, dctx=model_ctx)
    softmax_lambd = _get_val(
        config, sec, 'softmax_lambd', dtype='float', dval=None)
    if softmax_lambd is not None:
        mrt.set_softmax_lambd(softmax_lambd)
    shift_bits = _get_val(
        config, sec, 'shift_bits', dtype='int', dval=None)
    if shift_bits is not None:
        mrt.set_shift_bits(shift_bits)
    thresholds = _get_val(
        config, sec, 'Thresholds', dtype='dict', dval=None, dtype2='float')
    if thresholds is not None:
        for name, threshold in thresholds.items():
            mrt.set_threshold(name, threshold)
    mrt.quantize()
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    mrt.save(model_name+'base.quantize', datadir=dump_dir)
    oscales = mrt.get_output_scales()
    maps = mrt.get_maps()
    logger.info("Quantization finihed")

    # merge_model
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    if keys != '':
        model_merger = Model.merger(base, top, maps)
        model = model_merger.merge_model(callback=None)
        sym_file, prm_file = _load_fname(dump_dir, suffix='all.quantize')
        model.save(sym_file, prm_file)
        logger.info("Merge model finihed")

    # evaluation
    sec = 'EVALUATION'
    iter_num = _get_val(config, sec, 'Iter_num', dtype='int', dval=10)

    # compilation

