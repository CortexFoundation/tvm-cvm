import tensorflow as tf
import numpy as np

import dataset as ds

import sys
import os
from os import path

def load_data_1(input_size=224, batch_size=1, layout='NHWC'):
    ds_name = 'imagenet'
    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, label = data_iter_func()
    data = data.asnumpy()
    if layout == 'NHWC':
        data = np.transpose(data, axes=[0,2,3,1])
    print('data loaded with shape: ', data.shape)
    return data, label

def run_lite(modelname):
    lite = lite_path[modelname]
    interpreter = tf.lite.Interpreter(model_path=lite)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_details[0]['shape'] = np.array([160, 299, 299, 3])
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    _, input_size, _, _ = input_shape
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data, label = load_data_1(input_size=input_size, batch_size=160, layout="NHWC")
    # input_data, label = load_data_1(input_size=input_size, batch_size=1, layout="NHWC")
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)


lite_path = {
    'inception_v3': "/data/tfmodels/lite/Inception_V3/inception_v3.tflite",
}

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Please enter at least 2 python arguments."
    modelname = sys.argv[1]
    run_lite(modelname)

