from tvm.contrib import util
import nnvm.compiler
import tvm
import os

def save(sym, params, target, shape_dict):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
    save_module(graph, lib, params)

def save_module(graph, lib, params):
    if not os.path.isdir('module'):
        os.mkdir('module')

    tmp = os.getcwd()
    path_lib = os.path.join(tmp, 'module', 'deploy_lib.so')
    lib.export_library(path_lib)
    graph_path = os.path.join(tmp, 'module', 'deploy_graph.json')
    with open(graph_path, 'w') as fo:
        fo.write(graph.json())
    param_path = os.path.join(tmp, 'module', 'deploy_param.params')
    with open(param_path, 'wb') as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

def load_module(graph_path,lib_path, param_path):
    loaded_json = open(graph_path).read()
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(param_path, 'rb').read())

    module = graph_rumtime.create(loaded_json, loaded_lib, tvm.gpu(0))
    module.load_params(loaded_params)
    return module

def load():
    return load_module('module/deploy_graph.json', 'module/deploy_lib.so', 'module/deploy_param.params')
