#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:28:30 2020

@author: mhosseina
"""


import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing
import argparse
from scipy.cluster import  hierarchy
def cluster(data, out_name):
    window = 5
    class callback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0
        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            if self.epoch == 0:
              print(' loss after epoch {}: {}'.format(self.epoch, loss))  
            else:
              print('loss after epoch {}: {}'.format(self.epoch, loss - self.pre_loss))  
            self.epoch += 1
            self.pre_loss = loss
            
    all_words = np.load(data, allow_pickle=True)
    
    print("building model...")
    word2vec = Word2Vec(min_count=5, workers=multiprocessing.cpu_count(), size=50, window=window, iter=100,
                        alpha=0.025,negative = 15, sg=1, hs=1, compute_loss=True)
    
    word2vec.build_vocab(all_words)
    vocabulary = word2vec.wv.vocab
    word2vec.train(
        all_words,
        epochs=word2vec.iter,
        total_examples=word2vec.corpus_count,
        compute_loss=True,
        callbacks=[callback()],
    )
    vocab = np.asarray(list(vocabulary.keys()))
    print("model has been built.")
    vectors = []
    for i in range(len(vocab)):
        vectors.append(word2vec.wv[vocab[i]])
        
    vectors = np.asarray(vectors)
    print("clustering...")
    threshold = 0.3
    Z = hierarchy.linkage(vectors,"average", metric="cosine")
    labels = hierarchy.fcluster(Z, threshold, criterion="distance")
    labels = np.reshape(labels, (len(labels), 1))
    vocab = np.reshape(vocab, (len(vocab), 1))
    output = np.concatenate((vocab,labels), axis = 1)
    np.save(out_name,output)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-o', '--output', action="store")
   args = parser.parse_args()
   cluster(args.input, args.output)
   
   
   
   
   