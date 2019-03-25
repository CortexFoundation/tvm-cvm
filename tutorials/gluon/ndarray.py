from mxnet import nd

x = nd.ones(shape=(3, 4))
y = nd.ones(shape=(3, 4))
nor = nd.normal(0, 1, shape=(3,4))

print (x + y)
print (x * y)
print (nd.dot(x, y.T))

x = nd.arange(3).reshape((1, 3))
y = nd.arange(2).reshape((2, 1))
print (x.shape)
print (y.shape)
print (x + y)

print (x)
print (x.asnumpy())
