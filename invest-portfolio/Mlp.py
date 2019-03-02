import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
注释：
这是定义隐藏层的类，首先明确：隐藏层的输入即input，输出即隐藏层的神经元个数。输入层与隐藏层是全连接的。
假设输入是n_in维的向量（也可以说时n_in个神经元），隐藏层有n_out个神经元，则因为是全连接，
一共有n_in*n_out个权重，故W大小时(n_in,n_out),n_in行n_out列，每一列对应隐藏层的每一个神经元的连接权重。
b是偏置，隐藏层有n_out个神经元，故b时n_out维向量。
rng即随机数生成器，numpy.random.RandomState，用于初始化W。
input训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，而这里的参数input大小是（n_example,n_in）,每一行一个样本，即每一行作为MLP的输入层。
activation:激活函数,这里定义为函数tanh
        """

        self.input = input  # 类HiddenLayer的input即所传递进来的input


"""
注释：
代码要兼容GPU，则W、b必须使用 dtype=theano.config.floatX,并且定义为theano.shared
另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。
"""
# 如果W未初始化，则根据上述方法初始化。
# 加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W，见我的上一篇文章。
if W is None:
    W_values = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    if activation == theano.tensor.nnet.sigmoid:
        W_values *= 4
    W = theano.shared(value=W_values, name='W', borrow=True)

if b is None:
    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)

# 用上面定义的W、b来初始化类HiddenLayer的W、b
self.W = W
self.b = b

# 隐含层的输出
lin_output = T.dot(input, self.W) + self.b
self.output = (
    lin_output if activation is None
    else activation(lin_output)
)

# 隐含层的参数
