__author__ = 'Christoph Jansen'

import nltk
import os
import numpy as np

from brocas_lm.model import Normalization
from brocas_lm.model import NormalizationIter
from brocas_lm.model import LanguageModel

# create work dir

work_dir = os.path.join(os.path.expanduser('~'), 'brocas_models')
lm_file = os.path.join(work_dir, 'confusion_words_model.bin')
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# get text corpus

nltk.download('brown')
all_sents = nltk.corpus.brown.sents()

# initialize and train

class AdvancedCorpusSplitter:
    def __init__(self, tokenized_sentences, preserve_tokens, test_part=.2, cv_part=.2):
        self.sents = tokenized_sentences

        counters = {}
        for sentence in self.sents:
            for token in sentence:
                if token in preserve_tokens:
                    counters[token] = counters.get(token, 0) + 1

        test_counters = {}
        self.test_sents = set()
        cv_counters = {}
        self.cv_sents = set()
        for i, sentence in enumerate(self.sents):
            for j, token in enumerate(sentence):
                if counters.get(token, 0) * test_part > test_counters.get(token, 0):
                    test_counters[token] = test_counters.get(token, 0) + 1
                    self.test_sents.update([i])
                elif (not i in self.test_sents) and (counters.get(token, 0) * cv_part > cv_counters.get(token, 0)):
                    cv_counters[token] = cv_counters.get(token, 0) + 1
                    self.cv_sents.update([i])

    def __iter__(self):
        for i, sentence in enumerate(self.sents):
            if (i in self.test_sents) or (i in self.cv_sents):
                continue
            yield sentence

    def cv(self, token):
        for i, sentence in enumerate(self.sents):
            if not i in self.cv_sents:
                continue
            for j, t in enumerate(sentence):
                if t == token:
                    yield (sentence, j)

    def test(self, token):
        for i, sentence in enumerate(self.sents):
            if not i in self.test_sents:
                continue
            for j, t in enumerate(sentence):
                if t == token:
                    yield (sentence, j)

normalizer = Normalization(all_sents,
                           lower_case=True,
                           min_count=20,
                           start_tag='S',
                           end_tag='E',
                           unknown_tag='U',
                           digit_tag='D')

all_sents_normalized = NormalizationIter(normalizer, all_sents)

cs1 = ['than', 'then']
cs2 = ['except', 'accept']
cs3 = ['well', 'good']

acs = AdvancedCorpusSplitter(all_sents_normalized, cs1 + cs2 + cs3)

if os.path.isfile(lm_file):
    lm = LanguageModel(lm_file=lm_file)

else:
    lm = LanguageModel(verbose=True,
                       tokenized_sentences=acs,
                       input_layer_size=128,
                       hidden_layer_size=512)

    cost_log = lm.train(acs,
                        epochs=10,
                        backup_directory=work_dir,
                        return_cost=True,
                        log_interval=1000)
    lm.save(lm_file)

# sampling

print()
print('sampling...')

def print_samples(language_model, start_tokens, end_tag, num_samples, temperatures):
    for t in temperatures:
        print('temperature: ', t)
        for i in range(num_samples):
            print(' '.join(language_model.sample(start_tokens, end_tag, temperature=t)[1:-1]))
        print()

print_samples(lm, ['S'], 'E', 5, [0.8, 0.9, 1.0, 1.1, 1.2])

# apply model to solve confusion word problem

print('apply language model...')
print()

def accuracy(flags):
    c = correct(flags)
    w = wrong(flags)
    if c == 0:
        return 0
    return c / (c + w)

def correct(flags):
    tn, fp, tp, fn = flags
    return tp + tn

def wrong(flags):
    tn, fp, tp, fn = flags
    return fp + fn

def precision(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fp)

def recall(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fn)

def f_score(flags, beta=1):
    p = precision(flags)
    r = recall(flags)
    b = beta
    if p == 0 or r == 0:
        return 0
    return (1 + b*b) * (p*r / (b*b*p + r))

def w_flags(flags, artificial_error):
    e = artificial_error
    tn, fp, tp, fn = flags
    return (tn * (1.0-e), fp * (1.0-e), tp * e, fn * e)

def w_error_corpus(flags):
    tn, fp, tp, fn = flags
    error_corpus_weight = (tn + fp) / (tp + fn)
    return (tn, fp, tp * error_corpus_weight, fn * error_corpus_weight)

def switch_confusion_words(tokenized_sentences_with_index, token):
    for s, idx in tokenized_sentences_with_index:
        s[idx] = token
        yield (s, idx)

token_a = 'than'
token_b = 'then'

f_score_beta = 0.5
artificial_error = 0.01
thresholds = np.arange(0, 15, .1)

correct_a = list(acs.cv(token_a))
wrong_a = list(switch_confusion_words(acs.cv(token_b), token_a))
correct_b = list(acs.cv(token_b))
wrong_b = list(switch_confusion_words(acs.cv(token_a), token_b))

def predict_and_exp(tokenenized_sentences_with_idx):
    result = []
    for s, idx in tokenenized_sentences_with_idx:
        p = lm.predict(s)
        e = np.exp(p)
        result.append(e)
    return result

correct_a_predictions = predict_and_exp(correct_a)
wrong_a_predictions = predict_and_exp(wrong_a)
correct_b_predictions = predict_and_exp(correct_b)
wrong_b_predictions = predict_and_exp(wrong_b)

def optimized_softmax(exps):
    s = np.sum(exps)
    return np.divide(exps, s)

def log_probs(lm, predictions, tokenized_sentences_with_idx):
    result = []
    for prediction, (sentence, idx) in zip(predictions, tokenized_sentences_with_idx):
        log_probability = 0
        for pred, token in zip(prediction, sentence[1:]):
            probs = optimized_softmax(pred)
            idx = lm.sparse_embeddings[token]
            prob = probs[idx]
            log_probability += np.log(prob)
        result.append(log_probability)
    return result

def statistics(correct_a_log_probs, wrong_a_log_probs, correct_b_log_probs, wrong_b_log_probs, threshold):
        TN = 0
        FP = 0
        TP = 0
        FN = 0

        for ca, wb in zip(correct_a_log_probs, wrong_b_log_probs):
            if wb > ca + threshold:
                FP += 1
            else:
                TN += 1

        for wa, cb in zip(wrong_a_log_probs, correct_b_log_probs):
            if cb > wa + threshold:
                TP += 1
            else:
                FN += 1

        return (TN, FP, TP, FN)

cv_results = []

correct_a_log_probs = log_probs(lm, correct_a_predictions, correct_a)
wrong_a_log_probs = log_probs(lm, wrong_a_predictions, wrong_a)
correct_b_log_probs = log_probs(lm, correct_b_predictions, correct_b)
wrong_b_log_probs = log_probs(lm, wrong_b_predictions, wrong_b)

for threshold in thresholds:

    flags = statistics(correct_a_log_probs, wrong_a_log_probs, correct_b_log_probs, wrong_b_log_probs, threshold)
    weighted_flags = w_flags(w_error_corpus(flags), artificial_error)
    score = f_score(weighted_flags, f_score_beta)
    p = precision(weighted_flags)
    r = recall(weighted_flags)
    data = (score, threshold, p, r, flags)
    cv_results.append(data)

print('best crossvalidation results:')

cv_results.sort(reverse=True)

best_threshold = cv_results[0][1]

def print_result(result):
    score, threshold, p, r, (TN, FP, TP, FN) = result
    print('threshold:\t', threshold)
    print('TN\tFP\tTP\tFN')
    print('%d\t%d\t%d\t%d' % (TN, FP, TP, FN))
    print('f-score:\t', score)
    print('precision:\t', p)
    print('recall:\t', r)

print_result(cv_results[0])

print()
print('test results:')

correct_a = list(acs.test(token_a))
wrong_a = list(switch_confusion_words(acs.test(token_b), token_a))
correct_b = list(acs.test(token_b))
wrong_b = list(switch_confusion_words(acs.test(token_a), token_b))

correct_a_predictions = predict_and_exp(correct_a)
wrong_a_predictions = predict_and_exp(wrong_a)
correct_b_predictions = predict_and_exp(correct_b)
wrong_b_predictions = predict_and_exp(wrong_b)

test_results = []

correct_a_log_probs = log_probs(lm, correct_a_predictions, correct_a)
wrong_a_log_probs = log_probs(lm, wrong_a_predictions, wrong_a)
correct_b_log_probs = log_probs(lm, correct_b_predictions, correct_b)
wrong_b_log_probs = log_probs(lm, wrong_b_predictions, wrong_b)

flags = statistics(correct_a_log_probs, wrong_a_log_probs, correct_b_log_probs, wrong_b_log_probs, best_threshold)
weighted_flags = w_flags(w_error_corpus(flags), artificial_error)
score = f_score(weighted_flags, f_score_beta)
p = precision(weighted_flags)
r = recall(weighted_flags)
data = (score, best_threshold, p, r, flags)

print_result(data)