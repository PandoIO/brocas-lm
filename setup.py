#!/usr/bin/env python

from distutils.core import setup

setup(name="brocas-lm",
      version='1.0',
      summary='A probabilistic language model based on a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM).',
      description="Broca's LM is a free python library providing a probabilistic language model based on a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM). It utilizes Gensim's Word2Vec implementation to transform input word sequences into a dense vector space. The output of the model is a seqeuence of probability distributions across the given vocabulary.",
      author='Christoph Jansen',
      author_email='jansen@pando.io',
      url='https://github.com/PandoIO/brocas-lm',
      packages=['brocas_lm'],
      license='MIT',
      platforms=["any"],
      install_requires=['theano', 'gensim']
     )
