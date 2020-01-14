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

def _get_path(config, sec, opt, is_dir=False, default_dir=None):
    pth_ = config[sec][opt]
    pth = path.abspath(path.expanduser(pth_))
    if is_dir:
        if pth_ == '':
            return default_dir
        _check_valid(path.isdir(pth), sec, opt, pth_,
                     message='Not a valid dir')
        if not path.exists(pth):
            path.makedirs(pth)
    else:
        _check_valid(path.exists(pth), sec, opt, pth_,
                    message='File not found')
    return pth

def _get_ctx(config, sec, default=mx.cpu()):
    ctx = default
    device_type = config[sec]['Device_type']
    _check_valid(device_type in ['', 'gpu', 'cpu'],
                 sec, 'Device_type', device_type)
    device_ids_ = config[sec]['Device_ids']
    if device_type == 'gpu':
        device_ids = eval(device_ids_)
        ctx = mx.gpu(device_ids) if isinstance(device_ids, int) \
              else [mx.gpu(i) for i in device_ids]
    else:
        _check_valid(device_ids == '', sec, 'Device_ids', device_ids_,
                     message='`Device_ids` should be null')
    return ctx

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

    # check nonemtpy options
    for sec in config.sections():
        for opt in nonempty_opts[sec]:
            assert config[sec][opt] != '', \
                'Please specify the value for option' + \
                '`%s` in section `%s`' % (opt, sec)

    # default
    sec = 'DEFAULT'
    sym_path = _get_path(config, sec, 'Symbol')
    prm_path = _get_path(config, sec, 'Params')
    model_dir = patn.dir(sym_path)
    model_ctx = _get_ctx(config, sec)

    # prepare
    model = Model.load(sym_path, prm_path)
    input_shape_ = config['PREPARE']['Input_shape']
    input_shape = eval(input_shape_)
    model.prepare(input_shape)
    dump_dir = _get_path(config, sec, 'Dump_dir', is_dir=True,
                         default_dir=model_dir)
    sym_file, prm_file = _load_fname(dump_dir, suffix='prepare')
    model.save(sym_file, prm_file)

    # split model
    sec = 'SPLIT_MODEL'
    keys_ = config[sec]['Keys']
    dump_dir = _get_path(config, sec, 'Dump_dir', is_dir=True,
                         default_dir=model_dir)
    if keys_:
        keys = [x.strip() for x in keys_.split(",")]
        base, top = model.split(keys)
        mrt = base.get_mrt()
        sym_file, prm_file = _load_fname(dump_dir, suffix='top')
        top.save(sym_file, prm_file)
        sym_file, prm_file = _load_fname(dump_dir, suffix='base')
        base.save(sym_file, prm_file)
    else:
        mrt = model.get_mrt()

    # calibration
    sec = 'CALIBRATION'
    calibrate_num = config.getint(sec, 'Calibrate_num')
    lambd = config.getfloat(sec, 'Lambda') \
        if config[sec]['Lambda'] else None
    dataset = config[sec]['Dataset']
    if dataset in ['voc', 'imagenet', 'mnist', \
                   'quickdraw', 'cifar10']:
        _check_valid(input_shape[2] == input_shape[3],
                     'PREPARE', 'Input_shape', input_shape,
                     message='Inconsistent input size')
        batch_size, num_channel, input_size, _ = input_shape
        data_iter_func = ds.data_iter(dataset, batch_size,
                                      input_size=input_size)
    elif dataset in ['trec']:
        _check_valid(input_shape[0] == 38, 'PREPARE',
                     'Input_shape', input_shape,
                     message='Invalid input shape for Trec')
        batch_size = input_shape[1]
        data_iter_func = ds.load_trec(batch_size)
    ctx = _get_ctx(config, sec, default=model_ctx)
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        th_dict = mrt.calibrate(lambd=lambd, ctx=ctx)
    dump_dir = _get_path(config, sec, 'Dump_dir', is_dir=True,
                         default_dir=model_dir)
    _, _, ext_file = _load_fname(dump_dir, suffix='th_dict',
                                 with_ext=True)
    sim.save_ext(ext_file, th_dict)

    # quantization

    # merge_model

    # evaluation

    # compilation

