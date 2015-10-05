__author__ = 'Christoph Jansen'

import os
import pickle
import numpy as np
import theano
from collections import Counter
from datetime import datetime
from brocas_lm._lstm import _LSTM
from gensim.models import Word2Vec

class LanguageModel:
    def __init__(self,
                 verbose=True,
                 lm_file=None,
                 tokenized_sentences=None,
                 input_layer_size=128,
                 hidden_layer_size=512):

        if verbose:
            self.print = _echo
        else:
            self.print = _silence

        if lm_file:
            self._initialize_from_file(lm_file)
        elif tokenized_sentences:
            self._input_layer_size = input_layer_size
            self._hidden_layer_size = hidden_layer_size
            self._initialize(tokenized_sentences)
        else:
            raise Exception('ERROR: either set tokenized_sentences OR lm_file parameter')

    def _initialize_from_file(self, lm_file):
        weights = {}
        with open(lm_file, 'rb') as f:
            self._input_layer_size = pickle.load(f)
            self._hidden_layer_size = pickle.load(f)
            self._output_layer_size = pickle.load(f)
            self.sparse_embeddings = pickle.load(f)
            self.w2v_embeddings = pickle.load(f)
            weights['W_xi'] = pickle.load(f)
            weights['W_hi'] = pickle.load(f)
            weights['W_ci'] = pickle.load(f)
            weights['b_i'] = pickle.load(f)
            weights['W_xf'] = pickle.load(f)
            weights['W_hf'] = pickle.load(f)
            weights['W_cf'] = pickle.load(f)
            weights['b_f'] = pickle.load(f)
            weights['W_xc'] = pickle.load(f)
            weights['W_hc'] = pickle.load(f)
            weights['b_c'] = pickle.load(f)
            weights['W_xo'] = pickle.load(f)
            weights['W_ho'] = pickle.load(f)
            weights['W_co'] = pickle.load(f)
            weights['b_o'] = pickle.load(f)
            weights['W_hy'] = pickle.load(f)
            weights['b_y'] = pickle.load(f)
        self._lstm = _LSTM(self._input_layer_size, self._hidden_layer_size, self._output_layer_size, weights=weights)
        self.print('initialized model from file: %s' % lm_file)

    def _initialize(self, tokenized_sentences):
        self.w2v_embeddings = Word2Vec(tokenized_sentences, size=self._input_layer_size, min_count=1)
        vocab = set()
        for s in tokenized_sentences:
            vocab.update(s)
        self.sparse_embeddings = {key: i for i, key in enumerate(vocab)}
        self._output_layer_size = len(self.sparse_embeddings)
        self._lstm = _LSTM(self._input_layer_size, self._hidden_layer_size, self._output_layer_size)
        self.print('initialized new model')

    def train(self, tokenized_sentences, epochs=10, backup_directory=None, return_cost=True, log_interval=1000):
        if backup_directory and not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cost_logs = []

        self.print('start training...')

        for e in range(epochs):
            cost_log = 0
            for i, (S_x, Y) in enumerate(self._sequences(tokenized_sentences)):
                cost = self._lstm.train(S_x, Y)
                cost_log += cost

                if (i+1) % log_interval == 0:
                    cost_log /= log_interval
                    self.print('epoch: %d\tcount: %d\tcost: %f' % ((e+1), (i+1), cost_log))
                    cost_log = 0
                    if return_cost:
                        cost_logs.append(cost_log)

            if backup_directory:
                backup_file = '%s_epoch_%04d_model.bin' %(timestamp, (e+1))
                self.save(os.path.join(backup_directory, backup_file))

        self.print('end training')
        if return_cost:
            return cost_logs

    def predict(self, tokenized_sentence):
        S_x, Y = self._sequence(tokenized_sentence)
        return self._lstm.predict(S_x)

    def token_probabilities(self, tokenized_sentence, temperature=1.0):
        S_y = self.predict(tokenized_sentence)
        result = []
        for prediction, token in zip(S_y, tokenized_sentence[1:]):
            probabilities = tempered_softmax(prediction, temperature=temperature)
            idx = self.sparse_embeddings[token]
            probability = probabilities[idx]
            result.append(probability)
        return result

    def sentence_log_probability(self, tokenized_sentence, temperature=1.0):
        return sum(map(np.log, self.token_probabilities(tokenized_sentence, temperature=temperature)))

    def sample(self, start_tokens, end_tag=None, max_tokens=100, temperature=1.0):
        tokens = [''] * len(self.sparse_embeddings)
        for token, idx in self.sparse_embeddings.items():
            tokens[idx] = token

        if not start_tokens or len(start_tokens) < 1:
            raise Exception('ERROR: at least one start token in list start_tokens must be given')

        S_h = np.zeros(self._hidden_layer_size, dtype=theano.config.floatX)
        S_c = np.zeros(self._hidden_layer_size, dtype=theano.config.floatX)

        sequence = [start_tokens[0]]

        # warm up LSTM with start_tokens
        for token in start_tokens[1:]:
            if len(sequence) >= max_tokens:
                return sequence

            if end_tag and sequence[-1] == end_tag:
                return sequence

            S_x = np.asarray(self.w2v_embeddings[sequence[-1]])
            S_x = np.reshape(S_x, (1, -1))
            S_h, S_c, S_y = self._lstm.sampling(S_x, S_h.flatten(), S_c.flatten())
            sequence.append(token)

        # sample random tokens to continue sequence
        while True:
            if len(sequence) >= max_tokens:
                return sequence

            if end_tag and sequence[-1] == end_tag:
                return sequence

            S_x = np.asarray(self.w2v_embeddings[sequence[-1]])
            S_h, S_c, S_y = self._lstm.sampling(np.reshape(S_x, (1, -1)), S_h.flatten(), S_c.flatten())
            probabilities = tempered_softmax(S_y.flatten(), temperature=temperature)
            token = np.random.choice(tokens, 1, p=probabilities)[0]
            sequence.append(token)

    def _sequence(self, tokenized_sentence):
        rows = len(tokenized_sentence) - 1
        columns = len(self.sparse_embeddings)
        S_x = np.asarray([self.w2v_embeddings[t] for t in tokenized_sentence[:-1]], dtype=theano.config.floatX)
        Y = np.zeros((rows, columns), dtype=theano.config.floatX)
        for i in range(0, rows):
            t = tokenized_sentence[i+1]
            k = self.sparse_embeddings[t]
            Y[i][k] = 1
        return (S_x, Y)

    def _sequences(self, tokenized_sentences):
        for s in tokenized_sentences:
            yield self._sequence(s)

    def save(self, lm_file):
        objects = [self._input_layer_size,
                   self._hidden_layer_size,
                   self._output_layer_size,
                   self.sparse_embeddings,
                   self.w2v_embeddings,
                   self._lstm.W_xi,
                   self._lstm.W_hi,
                   self._lstm.W_ci,
                   self._lstm.b_i,
                   self._lstm.W_xf,
                   self._lstm.W_hf,
                   self._lstm.W_cf,
                   self._lstm.b_f,
                   self._lstm.W_xc,
                   self._lstm.W_hc,
                   self._lstm.b_c,
                   self._lstm.W_xo,
                   self._lstm.W_ho,
                   self._lstm.W_co,
                   self._lstm.b_o,
                   self._lstm.W_hy,
                   self._lstm.b_y]
        with open(lm_file, 'wb') as f:
            for obj in objects:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.print('saved model to file: %s' % lm_file)

class Normalization:
    def __init__(self,
                 tokenized_sentences,
                 lower_case=True,
                 min_count=20,
                 start_tag='<SEQ>',
                 end_tag='</SEQ>',
                 unknown_tag='<UNK/>',
                 digit_tag='<D/>'):

        self._tokenized_sentences = tokenized_sentences
        self._min_count = min_count
        self._unknown_tag= unknown_tag

        self._normalization_functions = []
        if digit_tag:
            self._digit_tag = digit_tag
            self._normalization_functions.append(self._replace_digits)
        if lower_case:
            self._normalization_functions.append(self._lower_case)

        if start_tag:
            self._start_list = [start_tag]
        else:
            self._start_list = []

        if end_tag:
            self._end_list = [end_tag]
        else:
            self._end_list = []

        self._vocab = self._generate_vocabulary()

    def normalize(self, tokenized_sentence):
        s = self._start_list + [self._normalize(token) for token in tokenized_sentence] + self._end_list
        return [token if token in self._vocab else self._unknown_tag for token in s]

    def _normalize(self, token):
        for f in self._normalization_functions:
            token = f(token)
        return token

    def _generate_vocabulary(self):
        c = Counter()
        for s in self._tokenized_sentences:
            s = self._start_list + [self._normalize(token) for token in s] + self._end_list
            c.update(s)
        return {key for key, val in c.items() if val >= self._min_count}

    def _replace_digits(self, token):
        return ''.join([self._digit_tag if c.isdigit() else c for c in token])

    def _lower_case(self, token):
        return token.lower()

class NormalizationIter:
    def __init__(self, normalization, tokenized_sentences):
        self._n = normalization
        self._tokenized_sentences = tokenized_sentences

    def __iter__(self):
        for s in self._tokenized_sentences:
            yield self._n.normalize(s)

def _echo(message):
    print(message)

def _silence(message):
    pass

def tempered_softmax(vals, temperature=1.0):
    exps = [np.exp(val/temperature) for val in vals]
    s = sum(exps)
    return [val/s for val in exps]