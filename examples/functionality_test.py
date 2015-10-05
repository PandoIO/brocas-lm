__author__ = 'Christoph Jansen'

import nltk
import os

from brocas_lm.model import Normalization
from brocas_lm.model import NormalizationIter
from brocas_lm.model import LanguageModel

# create work dir
work_dir = os.path.join(os.path.expanduser('~'), 'brocas_models')
lm_file = os.path.join(work_dir, 'test_model.bin')
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# get text corpus
nltk.download('brown')
sents = nltk.corpus.brown.sents()[:100]

# preprocessing
normalizer = Normalization(sents, min_count=15)
training_data = NormalizationIter(normalizer, sents)
lm = LanguageModel(tokenized_sentences=training_data, input_layer_size=64, hidden_layer_size=128)
print()

# train model
lm.train(training_data, epochs=5, backup_directory=work_dir, log_interval=20)
print()

# test trained model
normalized_sentence = normalizer.normalize(sents[0])
print('normalized sentence:')
print(' '.join(normalized_sentence))
print('probability: ', lm.sentence_log_probability(normalized_sentence))
print()
start_tag = normalized_sentence[0]
end_tag = normalized_sentence[-1]
print('sample:')
print(' '.join(lm.sample([start_tag], end_tag=end_tag)))
print()

# save, load and test loaded model
lm.save(lm_file)
print()
lm_clone = LanguageModel(lm_file=lm_file)
print()
print('probability: ', lm_clone.sentence_log_probability(normalized_sentence))
print()
print('sample:')
print(' '.join(lm_clone.sample([start_tag], end_tag=end_tag)))
print()

# use predict and token_probabilities functions
print('predict:')
print(lm_clone.predict(normalized_sentence))
print()

print('token probabilities:')
print(lm_clone.token_probabilities(normalized_sentence))
print()