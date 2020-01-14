import sys
from os import path
import configparser

import mxnet as mx

from transformer import Model
import dataset as ds
import sim_quant_helper as sim
import utils

def _check_valid(exp, sec, opt, val, message='Not valid'):
    assert exp, message + '.    option `%s`: `%s` in ' + \
        'section `%s`' % (opt, val, sec)

def _get_path(config, sec, opt, is_dir=False, dpath=NoneType):
    pth_ = _get_val(config, sec, opt, '' if is_dir else NoneType)
    pth = path.abspath(path.expanduser(pth_))
    if is_dir:
        _check_valid(
            path.isdir(pth), sec, opt, pth_, message='Not a valid dir')
        if not path.exists(pth):
            path.makedirs(pth)
    else:
        _check_valid(
            path.exists(pth), sec, opt, pth_, message='File not found')
    return pth

def _get_ctx(config, sec, dctx=mx.cpu()):
    ctx = dctx
    device_type = _get_val(config, sec, 'Device_type', dval='cpu')
    _check_valid(
        device_type in ['', 'gpu', 'cpu'], sec, 'Device_type', device_type,
        message='Only support `gpu`, `cpu` and null value')
    device_ids_ = config[sec]['Device_ids']
    if device_type == 'gpu':
        device_ids = eval(device_ids_)
        ctx = mx.gpu(device_ids) if isinstance(device_ids, int) \
              else [mx.gpu(i) for i in device_ids]
    else:
        _check_valid(
            device_ids == '', sec, 'Device_ids', device_ids_,
            message='`Device_ids` should be null given `cpu` device type')
    return ctx

NoneType = object()

def _get_val(config, sec, opt, dtype='str', dval=NoneType):
    val_ = config[sec][opt]
    if val_ == '':
        _check_valid(
            dval != NoneType, sec, opt, val_,
            message="Please specify the default value")
        val = dval
    elif dytpe == 'str':
        val = val_
    elif dtype in ['int', 'tuple', 'float']:
        val = eval(val_)
        vtype = type(val).__name__
        _check_valid(
            vtype == dtype, sec, opt, val_,
            message="Only support data type: %s" % dtype)
    elif dtype == 'comma_list':
        val = [s.strip() for x in val_.split(',')]
    else:
        _check_valid(
            False, sec, opt, val_, message="Unknown data type")
    return val

def _load_fname(directory, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    return utils.extend_fname(directory+suffix, with_ext)

nonempty_opts = {
    'DEFAULT': {'Symbol', 'Params'},
    'PREPARE': {'Input_shape'},
    'SPLIT_MODEL': {},
    'CALIBRATION': {'Calibrate_num', 'Dataset'},
    'QUANTIZATION': {},
    'MERGE_MODEL': {},
    'EVALUATION': {},
    'COMPILATION': {},
    'DUMP': {},
}

if __name__ == "__main__":
    assert len(sys.argv) == 2
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    fileName = path.basename(cfgPath)
    absCfgPath = path.join(baseDir, fileName)

    config = configparser.ConfigParser()
    config.read(absCfgPath)

    # default
    sec = 'DEFAULT'
    sym_path = _get_path(config, sec, 'Symbol')
    prm_path = _get_path(config, sec, 'Params')
    model_dir = path.dir(sym_path)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_ctx = _get_ctx(config, sec)

    # prepare
    sec = 'PREPARE'
    model = Model.load(sym_path, prm_path)
    input_shape = _get_val(config, sec, 'Input_shape', dtype='tuple')
    model.prepare(input_shape)
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    sym_file, prm_file = _load_fname(dump_dir, suffix='prepare')
    model.save(sym_file, prm_file)

    # split model
    sec = 'SPLIT_MODEL'
    keys = _get_val(config, sec, 'Keys', dval='')
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

    # calibration
    sec = 'CALIBRATION'
    calibrate_num = _get_val(config, sec, 'Calibrate_num', dtype='int')
    lambd = _get_val(config, sec, 'Lambda', dtype='float', dval=None)
    dataset = _get_val(config, sec, 'dataset')
    if dataset in ['voc', 'imagenet', 'mnist', \
                   'quickdraw', 'cifar10']:
        _check_valid(
            input_shape[2] == input_shape[3], 'PREPARE', 'Input_shape',
            input_shape, message='Inconsistent input size')
        batch_size, num_channel, input_size, _ = input_shape
        data_iter_func = ds.data_iter(
            dataset, batch_size, input_size=input_size)
    elif dataset in ['trec']:
        _check_valid(
            input_shape[0] == 38, 'PREPARE', 'Input_shape', input_shape,
            message='Invalid input shape for Trec')
        batch_size = input_shape[1]
        data_iter_func = ds.load_trec(batch_size)
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
    dump_dir = _get_path(
        config, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
    mrt.quantize()

    # merge_model

    # evaluation

    # compilation

