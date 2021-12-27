import numpy

def softmax(x):
    exp_x = x.detach().numpy()
    sum_exp_x = numpy.sum(exp_x)
    y = exp_x/sum_exp_x

    return y

