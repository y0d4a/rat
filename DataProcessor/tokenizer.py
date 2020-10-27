# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:52:23 2020

@author: MHIT
"""

import numpy as np
from tqdm import tqdm  
import argparse
 

def tokenizer(data, vocab, n, output):
    tokens = []
    data = np.load(data)
    vocab = np.load(vocab)
    vocab = list(vocab)
    vocab.sort(key = lambda s: len(s))
    vocab.reverse()
    for i, d in enumerate(tqdm(data)):
        temp = []
        while len(d) > 0:
            found = False
            for j, v in enumerate(vocab):
                if d[0:len(v)].lower() == v.lower():
                    found = True
                    temp.append(d[0:len(v)])
                    d = d[len(v):]
                    break
            if not found:
                temp.append(d[0])
                d = d[1:]
        tokens.append(np.asarray(temp))
        
        
    tokens = np.asarray(tokens)
    vocabulary = {}
    vocab_list = []
    
    data = tokens
    del tokens
    token_id = 1
    counter = 0
    tokens_list = []
    for d in tqdm(data):

        tokens = ["".join(d[i:i+n]) for i in range(len(d)-n+1)]
        tokens_list.append(tokens)
        for t in tokens:
            if t not in vocab_list:
                vocabulary.update({t : token_id})
                vocab_list.append(t)
                token_id += 1
        counter += 1
    
    del data
    tokens_list = np.asarray(tokens_list)
    np.save(output, tokens_list)
    


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-n', action="store", type=int)
   parser.add_argument('-t', '--tokens', action="store")
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-o', '--output', action="store")
   args = parser.parse_args()
   tokenizer(args.input, args.tokens, args.n, args.output)
            
    
