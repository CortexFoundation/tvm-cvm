import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

# Download the MNIST dataset, then create the training and test sets
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                              batch_size=32, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                             batch_size=32, shuffle=False)
# Initialize the model, create a instance
net = gluon.nn.Sequential()

with net.name_scope():
    # add a full-connect layer, output 10 nums
    net.add(gluon.nn.Dense(10))

ctx = mx.cpu()
#dtype = 'int32'
dtype = 'int8'
# init model parameters, initial Normal distribution with var=0.05, mean=0
net.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=ctx)
# define loss function, here we use soft cross entropy
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# define learn/optimaize function, use small batch
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

epochs = 1
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record(): # start recording the derivatives
            output = net(data) # the forward iteration
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
        curr_loss = ndarray.mean(loss).asscalar()
        break

net.hybridize()
net(mx.nd.zeros((1, 784)))

params = {'data': mx.nd.zeros((1, 784)).astype(dtype)}
params.update({x:y.data().astype(dtype) for x, y in net.collect_params().items()})
symbol = net(mx.symbol.Variable('data', dtype=dtype))
e = symbol.bind(ctx=ctx, args=params)
e.forward()

import tvm
import nnvm
from tvm.contrib import util, graph_runtime as runtime
from tvm import rpc

target = 'llvm'
sym, params = nnvm.frontend.from_mxnet(net(mx.symbol.Variable('data', dtype=dtype)))
graph, lib, params = nnvm.compiler.build(sym, target, shape={'data': (1, 784)}, params=params)

remote = rpc.LocalSession()
#remote_ctx = remote.cpu(0)
remote_ctx = remote.gpu(0)
module = runtime.create(graph, lib, ctx=remote_ctx)

params = {}
params.update({x:tvm.nd.array((y.data() * 256).astype(dtype).asnumpy()) for x, y in net.collect_params().items()})

data = (np.random.uniform(size=(784,)) * 256).astype(dtype)
module.set_input('data', data)
module.set_input(**params)
module.run()

output = module.get_output(0, tvm.nd.empty((10,), dtype=dtype))
print (output)

