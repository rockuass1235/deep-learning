from mxnet import nd, autograd
def get_data(dims = 2, num = 1000):

    true_w = nd.array([2, -3.4])
    true_b = 4.2

    x = nd.random.normal(shape = (num, dims))
    y = (x * true_w).sum(axis = 1) + true_b
    y = y.reshape((-1,1))

    return x, y + nd.random.normal(scale=0.01, shape=y.shape)


