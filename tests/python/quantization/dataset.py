import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet.gluon.data.vision import transforms
import numpy as np

import os
import math

def load_voc(batch_size, input_size=416):
    width, height = input_size, input_size
    val_dataset = gdata.VOCDetection(splits=[('2007', 'test')])
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)
    return val_loader

def load_voc_metric():
    return VOC07MApMetric(iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)


def load_imagenet(batch_size):
    val_dataset = ImageNet(train=False)
    val_loader = gluon.data.DataLoader(
        val_dataset,
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)

def load_imagenet_rec(batch_size, input_size=224):
    rec_val = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.rec")
    rec_val_idx = os.path.expanduser("~/.mxnet/datasets/imagenet/rec/val.idx")
    crop_ratio = 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    val_data = mx.io.ImageRecordIter(
	path_imgrec         = rec_val,
	path_imgidx         = rec_val_idx,
	preprocess_threads  = 24,
	shuffle             = False,
	batch_size          = batch_size,

	resize              = resize,
	data_shape          = (3, input_size, input_size),
	mean_r              = mean_rgb[0],
	mean_g              = mean_rgb[1],
	mean_b              = mean_rgb[2],
	std_r               = std_rgb[0],
	std_g               = std_rgb[1],
	std_b               = std_rgb[2],
    )
    return val_data

def load_cifar10(batch_size, input_size=224, num_workers=4):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            yield data[0], label[0]
    return data_iter()

def load_quickdraw10(batch_size, num_workers=4):
    X = nd.array(np.load('/home/tian/cortex_ml_data/quickdraw_X_test.npy'))
    y = nd.array(np.load('/home/tian/cortex_ml_data/quickdraw_y_test.npy'))
    val_data = gluon.data.DataLoader(
             mx.gluon.data.dataset.ArrayDataset(X, y),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            data, label = data[0], label[0]
            yield data, label
    return data_iter()

def load_sa_trec(batch_size, num_workers=4):
    import re
    import time

    import gluonnlp as nlp
    from mxnet import nd, gluon


    def _load_file(data_name):
        if data_name == 'MR':
            train_dataset = nlp.data.MR(root='data/mr')
            output_size = 2
            return train_dataset, output_size
        elif data_name == 'SST-1':
            train_dataset, test_dataset = [nlp.data.SST_1(root='data/sst-1', segment=segment)
                                           for segment in ('train', 'test')]
            output_size = 5
            return train_dataset, test_dataset, output_size
        elif data_name == 'SST-2':
            train_dataset, test_dataset = [nlp.data.SST_2(root='data/sst-2', segment=segment)
                                           for segment in ('train', 'test')]
            output_size = 2
            return train_dataset, test_dataset, output_size
        elif data_name == 'Subj':
            train_dataset = nlp.data.SUBJ(root='data/Subj')
            output_size = 2
            return train_dataset, output_size
        else:
            train_dataset, test_dataset = [nlp.data.TREC(root='data/trec', segment=segment)
                                           for segment in ('train', 'test')]
            output_size = 6
            return train_dataset, test_dataset, output_size


    def _clean_str(string, data_name):
        if data_name == 'SST-1' or data_name == 'SST-2':
            string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
            string = re.sub(r'\s{2,}', ' ', string)
            return string.strip().lower()
        else:
            string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
            string = re.sub(r'\'s', ' \'s', string)
            string = re.sub(r'\'ve', ' \'ve', string)
            string = re.sub(r'n\'t', ' n\'t', string)
            string = re.sub(r'\'re', ' \'re', string)
            string = re.sub(r'\'d', ' \'d', string)
            string = re.sub(r'\'ll', ' \'ll', string)
            string = re.sub(r',', ' , ', string)
            string = re.sub(r'!', ' ! ', string)
            string = re.sub(r'\(', ' ( ', string)
            string = re.sub(r'\)', ' ) ', string)
            string = re.sub(r'\?', ' ? ', string)
            string = re.sub(r'\s{2,}', ' ', string)
            return string.strip() if data_name == 'TREC' else string.strip().lower()


    def _build_vocab(data_name, train_dataset, test_dataset):
        all_token = []
        max_len = 0
        for i, line in enumerate(train_dataset):
            train_dataset[i][0] = _clean_str(line[0], data_name)
            line = train_dataset[i][0].split()
            max_len = max_len if max_len > len(line) else len(line)
            all_token.extend(line)
        for i, line in enumerate(test_dataset):
            test_dataset[i][0] = _clean_str(line[0], data_name)
            line = test_dataset[i][0].split()
            max_len = max_len if max_len > len(line) else len(line)
            all_token.extend(line)
        vocab = nlp.Vocab(nlp.data.count_tokens(all_token))
        # embedding = nlp.embedding.create('fasttext', source='wiki.simple')
        emdedding = nlp.embedding.create('Word2Vec', source='GoogleNews-vectors-negative300')
        vocab.set_embedding()
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == nd.zeros(300)).sum() == 300:
                vocab.embedding[word] = nd.random.normal(-1.0, 1.0, 300)
        vocab.embedding['<unk>'] = nd.zeros(300)
        vocab.embedding['<pad>'] = nd.zeros(300)
        vocab.embedding['<bos>'] = nd.zeros(300)
        vocab.embedding['<eos>'] = nd.zeros(300)
        print('maximum length (in tokens): ', max_len)
        return vocab, max_len


    # Dataset preprocessing.
    def _preprocess(x, vocab, max_len):
        data, label = x
        data = vocab[data.split()]
        data = data[:max_len] + [0] * (max_len - len(data[:max_len]))
        return data, label


    def _preprocess_dataset(dataset, vocab, max_len):
        start = time.time()
        dataset = [_preprocess(d, vocab=vocab, max_len=max_len) for d in dataset]
        lengths = gluon.data.SimpleDataset([len(d[0]) for d in dataset])
        end = time.time()
        print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
        return dataset, lengths


    def load_dataset(data_name):
        """Load sentiment dataset."""
        if data_name == 'MR' or data_name == 'Subj':
            train_dataset, output_size = _load_file(data_name)
            vocab, max_len = _build_vocab(data_name, train_dataset, [])
            train_dataset, train_data_lengths = _preprocess_dataset(train_dataset, vocab, max_len)
            return vocab, max_len, output_size, train_dataset, train_data_lengths
        else:
            train_dataset, test_dataset, output_size = _load_file(data_name)
            vocab, max_len = _build_vocab(data_name, train_dataset, test_dataset)
            train_dataset, train_data_lengths = _preprocess_dataset(train_dataset, vocab, max_len)
            test_dataset, test_data_lengths = _preprocess_dataset(test_dataset, vocab, max_len)
            return vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, \
                   test_data_lengths
    vocab, max_len, output_size, train_dataset, train_data_lengths, \
    test_dataset, test_data_lengths = load_dataset('TREC')
    val_data = gluon.data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    def data_iter():
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[mx.cpu()], batch_axis=0)
            print (data[0].shape)
            yield data[0], label[0]
    return data_iter()

