from mxnet import autograd, nd

x = nd.arange(4).reshape((4, 1))
print (x)

# request for memery to save gradient
x.attach_grad()
# invoke record() to record mxnet and gredient calculation
with autograd.record():
    y = 2 * nd.dot(x.T, x)
# call backward() for auto gredient
y.backward()
# corectness
assert (x.grad - 4 * x).norm().asscalar() == 0
print (x.grad)
