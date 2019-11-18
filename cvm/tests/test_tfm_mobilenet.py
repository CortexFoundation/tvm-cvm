import os
import dataset as ds
import sim_quant_helper as sim

import utils
import logging
import mxnet as mx
from transformer import collect_op_names, pass_info, MRT, load_model

# set python-logging color format
utils.log_init()

'''
sym_path = "/tmp/densenet/densenet161.json"
prm_path = "/tmp/densenet/densenet161.params"
'''

# mobilenet1_0
sym_path = "./data/mobilenet1_0.json"
prm_path = "./data/mobilenet1_0.params"
if not os.path.exists(sym_path) or not os.path.exists(prm_path):
    save_mobilenet1_0()
batch_size = 16
input_size = 224
ds_name = 'imagenet'

'''
# inception_v3
sym_path = "./data/tf_inceptionv3.json"
prm_path = "./data/tf_inceptionv3.params"
batch_size = 16
input_size = 299
ds_name = 'imagenet'
'''

sym, params = mx.sym.load(sym_path), mx.nd.load(prm_path)
print (collect_op_names(sym, params))
print ("Registered Graph Pass")
for k, v in pass_info().items():
    print ("%20s" % k, v)

data_iter_func = ds.data_iter(ds_name,
        batch_size, input_size=input_size)
data, _ = data_iter_func()
mrt = MRT(sym, params,
        input_shape=(batch_size, 3, input_size, input_size))
mrt.set_data(data)
mrt.calibrate()
mrt.set_input_prec(8)
mrt.set_output_prec(8)
mrt.quantize()
qsym_path, qprm_path, qext_path = mrt.dump()

ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
inputs = [mx.sym.var('data')]
inputs_ext = { 'data': {
    'shape': (batch_size, 3, input_size, input_size), } }

# load original model
org_model = load_model(sym_path, prm_path, ctx)

# load quantized model
(inputs_ext, ) = sim.load_ext(qext_path)
cvm_quantize = load_model(qsym_path, qprm_path, ctx, inputs_ext=inputs_ext)

utils.multi_validate(org_model, data_iter_func, cvm_quantize,
        iter_num=10, logger=logging.getLogger('mrt.validate'))

