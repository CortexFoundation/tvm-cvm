# -*- coding:utf-8 -*-
import os
import mxnet as mx
import numpy as np
import logging
from numpy import *
from mxnet import nd
from mxnet import autograd
from math import floor
from mxnet.test_utils import get_mnist_iterator

class Batch_QuanBlock(mx.gluon.Block):
    def __init__(self, bit, eps=0.001, momentum=0.9, name='', **kwargs):
        super(Batch_QuanBlock, self).__init__(**kwargs)
        self._num_

    def forward(self, x):
        ctx = x.context
        return mx.symbol.Custom(x, self

