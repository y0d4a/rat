# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:50:46 2020

@author: MHIT
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import argparse
import math
import os, shutil


def clear_directory(directory):
    folder = directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def k_estimator(low, high, data):
    step = 1
    data = np.load(data)
    data = np.reshape(data,(data.shape[0],data.shape[1]))
    
    scores = [silhouette_score(data,
                              MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=128, init_size=3*k, verbose=0).fit_predict(data),
                              metric='euclidean', sample_size=np.min([50000, data.shape[0]]), random_state=42) for k in tqdm(range(low, high + step, step))]
    scores = np.asarray(scores)
    k = low + (np.argmax(scores) * step)
    return k

def cluster(payloads, features, k, directory):
    data = np.load(features)
    data = np.reshape(data,(data.shape[0],data.shape[1]))
    payloads = np.load(payloads)
    payloads = np.reshape(payloads,(payloads.shape[0],1))


    labels = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=32, init_size=3*k, verbose=1).fit_predict(data)
    labels = np.asarray(labels)
    labels = np.reshape(labels,(labels.shape[0],1))
    fulldata = np.concatenate((payloads,labels),axis=1)
    split(fulldata, directory)
    
def split(data, directory):
    clear_directory(directory)
    
    data = data[data[:,1].argsort()]
    j = int(float(data[0,1]))
    new_id = 0
    payloads = []
    for i, d in enumerate(tqdm(data)):
        if j == int(float(d[1])):
            payloads.append(d[0])
        else:
            np.save(directory+"/d_"+str(new_id), payloads)
            j = int(float(d[1]))
            new_id += 1
            payloads = []
            payloads.append(d[0])
            
    np.save(directory+"/d_"+str(new_id), payloads)
    
def extract_features(k, n, t, directory):
    char_level = False
    threshold = 0.05
    vocab = np.load(t)
    vocab = list(vocab)
    vocab.sort(key = lambda s: len(s))
    vocab.reverse()
    def Convert(lst): 
        res_dct = {lst[i]: i for i in range(0, len(lst), 1)} 
        return res_dct
    def IG(n):
        if len(all_words) == n:
            return 0
        return (-1 * (n/len(all_words)) * (math.log(n/len(all_words), 2))) - (((len(all_words)-n)/len(all_words)) * (math.log((len(all_words)-n)/len(all_words), 2)))
    for cluster in range(k):
        data = np.load(directory+"/d_"+str(cluster)+".npy")
        tokens = []

        if char_level:
            vocab = []
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
        token_id = 1
        counter = 0
        data = tokens
        tokens_list = []
        for d in tqdm(data):
            tokens = ["".join(d[i:i+n]) for i in range(len(d)-n+1)]
            tokens_list.append(np.asarray(tokens))
            for t in tokens:
                if t not in vocabulary:
                    vocabulary.update({t : token_id})
                    vocab_list.append(t)
                    token_id += 1
            counter += 1
        
        del data
        tokens_list = np.asarray(tokens_list)
        vocab_list = np.asarray(vocab_list)
        all_words = tokens_list
        data = []
        for i, words in enumerate(tqdm(all_words)):
            temp = []
            for w in words:
                if w not in temp:
                    temp.append(w)
            data.extend(temp)
        vocab = Convert(vocab_list)
        features = np.zeros(len(vocab_list))
        detailed = np.zeros(len(vocab_list))
        for i, d in enumerate(tqdm(data)):
            features[vocab[d]] += 1
        
        vocab = []
        for i, f in enumerate(tqdm(features)):
            if IG(f)>= threshold:
                vocab.append(vocab_list[i])
        features = [np.zeros(len(vocab)) for a in all_words]
        detailed = [np.zeros(len(vocab)) for a in all_words]
        for k, words in enumerate(tqdm(all_words)):

            for j,v in enumerate(vocab):
                if v in words:
                    detailed[k][j] = 1
                    features[k][j] = np.count_nonzero(words == v)
        features = np.asarray(features)
        np.save(directory+"/df_"+str(cluster), features)
        detailed = np.asarray(detailed)
    
        fv = []
        N = len(detailed)
        
        weights = []
        for i in tqdm(range(detailed.shape[1])):
            df = np.count_nonzero(detailed[:,i])
            idf = np.log(N/df)
            weights.append(idf)
            
        for i, d in enumerate(tqdm(detailed)):
            fv.append(np.multiply(weights, d))
            
        np.save(directory+"/wf_"+str(cluster), fv)
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-n', action="store", default=2, type=int)
   parser.add_argument('-m', '--minimum', action="store", default=25, type=int)
   parser.add_argument('-M', '--maximum', action="store", default=50, type=int)
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-t', '--tokens', action="store")
   parser.add_argument('-f', '--features', action="store")
   parser.add_argument('-d', '--directory', action="store")
   args = parser.parse_args()
   k = k_estimator(args.minimum, args.maximum, args.features)
   cluster(args.input, args.features, k, args.directory)
   extract_features(k, args.n, args.tokens, args.directory)