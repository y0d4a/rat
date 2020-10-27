#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:43:51 2020

@author: mhosseina
"""


import numpy as np
from tqdm import tqdm
import argparse
def binary_encoder(data, cluster, output):
    
    all_words = np.load(data, allow_pickle=True)
    data = np.load(cluster, allow_pickle=True)

    
    dictionary = {}
    for i, d in enumerate(tqdm(data)):
        dictionary.update({d[0] : int(d[1])-1})
        
    
    
    features = np.zeros((len(all_words), max(data[:,1].astype("int"))))
    for k, words in enumerate(tqdm(all_words)):
        
        for w in words:
            
            try:
                features[k][dictionary[w]] = 1
            except KeyError:
                pass
        

    features = np.asarray(features)
    
    np.save(output, features)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-c', '--cluster', action="store")
   parser.add_argument('-o', '--output', action="store")
   args = parser.parse_args()
   binary_encoder(args.input, args.cluster, args.output)
   