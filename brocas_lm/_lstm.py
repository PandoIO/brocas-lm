__author__ = 'Christoph Jansen'

import numpy as np
import theano
import theano.tensor as T
from theano import dot
from theano.tensor.nnet import sigmoid as sigm
from theano.tensor import tanh
from theano.tensor.nnet import softmax
from theano.tensor.nnet import categorical_crossentropy

class _LSTM:
    def __init__(self, i_size, h_size, o_size, weights=None):
        if not weights:
            self.W_xi = _init_weights((i_size, h_size))
            self.W_hi = _init_weights((h_size, h_size))
            self.W_ci = _init_weights((h_size, h_size))
            self.b_i = _init_zero_vec(h_size)

            self.W_xf = _init_weights((i_size, h_size))
            self.W_hf = _init_weights((h_size, h_size))
            self.W_cf = _init_weights((h_size, h_size))
            self.b_f = _init_zero_vec(h_size)

            self.W_xc = _init_weights((i_size, h_size))
            self.W_hc = _init_weights((h_size, h_size))
            self.b_c = _init_zero_vec(h_size)

            self.W_xo = _init_weights((i_size, h_size))
            self.W_ho = _init_weights((h_size, h_size))
            self.W_co = _init_weights((h_size, h_size))
            self.b_o = _init_zero_vec(h_size)

            self.W_hy = _init_weights((h_size, o_size))
            self.b_y = _init_zero_vec(o_size)
        else:
            self.W_xi = weights['W_xi']
            self.W_hi = weights['W_hi']
            self.W_ci = weights['W_ci']
            self.b_i = weights['b_i']

            self.W_xf = weights['W_xf']
            self.W_hf = weights['W_hf']
            self.W_cf = weights['W_cf']
            self.b_f = weights['b_f']

            self.W_xc = weights['W_xc']
            self.W_hc = weights['W_hc']
            self.b_c = weights['b_c']

            self.W_xo = weights['W_xo']
            self.W_ho = weights['W_ho']
            self.W_co = weights['W_co']
            self.b_o = weights['b_o']

            self.W_hy = weights['W_hy']
            self.b_y = weights['b_y']

        S_h = _init_zero_vec(h_size) # init values for hidden units
        S_c = _init_zero_vec(h_size) # init values for cell units

        S_x = T.matrix() # inputs
        Y = T.matrix() # targets

        (S_h_r, S_c_r, S_y_r ), _ = theano.scan(fn = _step,
                                                sequences = S_x,
                                                outputs_info = [S_h, S_c, None],
                                                non_sequences = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                                                                 self.W_xf, self.W_hf, self.W_cf, self.b_f,
                                                                 self.W_xc, self.W_hc, self.b_c,
                                                                 self.W_xo, self.W_ho, self.W_co, self.b_o,
                                                                 self.W_hy, self.b_y])

        cost = T.mean(categorical_crossentropy(softmax(S_y_r), Y))

        updates = _gradient_descent(cost,
                                    [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                                     self.W_xf, self.W_hf, self.W_cf, self.b_f,
                                     self.W_xc, self.W_hc, self.b_c,
                                     self.W_xo, self.W_ho, self.W_co, self.b_o,
                                     self.W_hy, self.b_y])

        self.train = theano.function(inputs=[S_x, Y],
                                     outputs=cost,
                                     updates=updates,
                                     allow_input_downcast=True)

        self.predict = theano.function(inputs=[S_x],
                                       outputs=S_y_r,
                                       allow_input_downcast=True)

        S_h_v = T.vector()
        S_c_v = T.vector()

        S_h_s, S_c_s, S_y_s = _step(S_x, S_h_v, S_c_v,
                                    self.W_xi, self.W_hi, self.W_ci, self.b_i,
                                    self.W_xf, self.W_hf, self.W_cf, self.b_f,
                                    self.W_xc, self.W_hc, self.b_c,
                                    self.W_xo, self.W_ho, self.W_co, self.b_o,
                                    self.W_hy, self.b_y)

        self.sampling = theano.function(inputs = [S_x, S_h_v, S_c_v],
                                        outputs = [S_h_s, S_c_s, S_y_s],
                                        allow_input_downcast=True)


def _gradient_descent(cost, weights, lr=0.05):
    grads = T.grad(cost=cost, wrt=weights)
    updates = []
    for w, g in zip(weights, grads):
        updates.append([w, w - lr * g])
    return updates

def _step(S_x, S_h, S_c,
          W_xi, W_hi, W_ci, b_i,
          W_xf, W_hf, W_cf, b_f,
          W_xc, W_hc, b_c,
          W_xo, W_ho, W_co, b_o,
          W_hy, b_y):

    S_i = sigm(dot(S_x, W_xi) + dot(S_h, W_hi) + dot(S_c, W_ci) + b_i)
    S_f = sigm(dot(S_x, W_xf) + dot(S_h, W_hf) + dot(S_c, W_cf) + b_f)
    S_c = S_f * S_c + S_i * tanh(dot(S_x, W_xc) + dot(S_h, W_hc) + b_c)
    S_o = sigm(dot(S_x, W_xo) + dot(S_h, W_ho) + dot(S_c, W_co) + b_o)
    S_h = S_o * tanh(S_c)
    S_y = dot(S_h, W_hy) + b_y

    return [S_h, S_c, S_y]

def _init_weights(shape, factor=0.01):
    return theano.shared(np.asarray(np.random.randn(shape[0], shape[1]) * factor, dtype=theano.config.floatX))

def _init_zero_vec(size):
    vec = np.zeros(size, dtype=theano.config.floatX)
    return theano.shared(vec)