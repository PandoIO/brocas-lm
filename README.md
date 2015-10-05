# Broca's Language Model

Broca's LM is a free python library providing a probabilistic language model based on a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM). It utilizes Gensim's Word2Vec implementation to transform input word sequences into a dense vector space. The output of the model is a seqeuence of probability distributions across the given vocabulary.
<br/><br/>
This library is named after <a href="https://en.wikipedia.org/wiki/Broca's_area">Broca's area</a>, one of the main language processors in the human brain. Broca's area is responsible for language comprehension as well as language production.

## Features

* determine sentence probabilities
* generate random sentences (sampling)
* continue incomplete word sequences (sampling)

## Dependencies

* Python >= 3.4
* <a href='http://www.deeplearning.net/software/theano/'>Theano</a>
* <a href='http://radimrehurek.com/gensim/'>Gensim</a>
* <a href='http://www.numpy.org/'>Numpy</a>
* <a href='http://www.nltk.org/'>NLTK</a> (for examples only)

The LSTM implementation is based on the Theano library. Theano and Gensim both support fast calculations with native CPU or GPU (<a href="https://developer.nvidia.com/cuda-zone">Cuda</a>) code. For more information see Cuda, Theano and Gensim documentations.

## Installation

Tested on Ubuntu 14.04:

```bash
sudo pip3 install brocas-lm
```

## Usage

Import packages:

```python
from brocas_lm.model import Normalization
from brocas_lm.model import NormalizationIter
from brocas_lm.model import LanguageModel
```

## Documentation

A complete documentation will be available soon. Take a look at the examples folder for basic usage information.