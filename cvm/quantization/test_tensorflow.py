import tensorflow as tf

import utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# from tensorflow.python.keras import backend as K

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

import os
from tvm.relay.frontend import tensorflow_parser as tp
from official.vision.image_classification import resnet_model as resm
from tensorflow.python.framework.graph_util import convert_variables_to_constants
def dump_resnet50_v1():
    root = "/tmp/tfmodels"
    model = resm.resnet50(1000)
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #     loss=tf.keras.losses.sparse_categorical_crossentropy,
    #     metrics=['accuracy'])
    os.makedirs(root, exist_ok=True)
    model.save(os.path.join(root, "keras.h5"))

    # tf.contrib.saved_model.save_keras_model(model, root,
            # serving_only=False)
    K.set_learning_phase(0)
    model = load_model(os.path.join(root, "keras.h5"))
    print("DDDDDDD", model.outputs)
    frozen_graph = freeze_session(K.get_session(),
            output_names=[out.op.name for out in model.outputs])
    # with K.get_session() as sess:
    #     frozen_graph = convert_variables_to_constants(sess, sess.graph_def,
    #                              [x.op.name for x in model.outputs])
    tf.train.write_graph(frozen_graph, root, "resnet50_v1.pb", as_text=False)

def dump_pretrain_resnet50_v1():
    root = "/tmp/tf/resnet50_v1"
    os.makedirs(root, exist_ok=True)

    # resnet50_v1 = tf.keras.applications.ResNet50(weights=None)
    resnet50_v1 = tf.keras.applications.ResNet50(weights='imagenet')
    resnet50_v1.save(os.path.join(root, "model.h5"))

    K.set_learning_phase(0)
    model = load_model(os.path.join(root, "model.h5"))
    print ("ResNet50_V1 Model: ", model.outputs)
    frozen_graph = freeze_session(K.get_session(),
            output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, root, "model.pb", as_text=False)

import keras_applications as kapp
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    # image /= 255.0  # normalize to [0,1] range
    # image = kapp.resnet.preprocess_input(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    print (image.numpy().min(), image.numpy().max())

    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def load_imagenet():
    data_root = os.path.expanduser("~/.mxnet/datasets/imagenet/val")
    print (data_root)
    import random
    import pathlib
    data_root = pathlib.Path(data_root)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    image_count = len(all_image_paths)
    print (image_count)
    print (all_image_paths[:10])

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print (label_names[-10:])
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    print (all_image_labels[:10])

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    print (image_label_ds)

    ds = image_label_ds.apply(
              tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(16)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    print (ds)


if __name__ == '__main__':
    utils.log_init()

    resnet50_v1 = tf.keras.applications.ResNet50(weights='imagenet')
    load_imagenet()
    tf.losses.softmax_cross_entropy
    tf.metrics.accuracy
    tf.keras.optimizers.SGD
