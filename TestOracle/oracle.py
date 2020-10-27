# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:53:56 2020

@author: MHIT
"""

import numpy as np
from random import randrange
import requests 
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import math
import time
import os, shutil
from tqdm import tqdm
import argparse

fixed_size = math.inf
random_search = False

n_attempt = 100
max_epsilon = 0.9
episode = 300
r = 1
punishment = 0.5
directory = 'temp'
total_average_reward = []

def clear_temp():
    if not os.path.exists(directory):
        os.makedirs(directory)
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
def copy_files(input_dir, k):
    src = input_dir
    src_files = os.listdir(src)
    for file_name in tqdm(src_files):
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, directory)
def min_distance(blocked, feature):
            return np.dot(blocked, feature)
        
def test_payload(payload, url):
        URL = url
        s = requests.Session()
        headers = requests.utils.default_headers()
        headers.update({'Cookie': "pid="+'payload',})
        retries = Retry(total=25,
                        backoff_factor=0.5,
                        status_forcelist=[ 500, 502, 503, 504 ])
    
        s.mount('http://', HTTPAdapter(max_retries=retries))
        PARAMS = {'q':payload}
        r = s.get(url = URL, params = PARAMS, headers=headers)
        code = r.status_code
        if code == 200:
            return True,code
        else:
            return False, code

def selectAcluster(clusteres, rewards, epsilon):
    if random.random() > epsilon:
        return np.argmax(rewards)
    else:
        return clusteres[0]
def oracle(url, input_dir, output_dir, cluster):
    n_try = 1
    total_times = []
    for nt in range(n_try):
        clear_temp()
        copy_files(input_dir,cluster)
        average_rewards = np.zeros(cluster)
        sum_of_rewards = np.zeros(cluster)
        number_of_games = np.zeros(cluster)
        per_cluster_times = []
        success_list = []
        result = []
        total_req = 0
        cluster_list = [i for i in range(cluster)]

       
        s = [list() for i in range(cluster)]
        blocked = [list() for i in range(cluster)]
        start_time = time.time()
        for ep in range(episode):
            total_average_reward.append(np.mean(average_rewards))
            c_start = time.time()

            epsilon = max_epsilon * np.exp(-0.005*ep)
            i = selectAcluster(cluster_list, average_rewards, epsilon)
            number_of_games[i] += 1
            
            
            cluster_list.append(cluster_list.pop(cluster_list.index(i)))

            payload = list(np.load(directory+"/d_"+str(i)+".npy"))
            detailed_feature = list(np.load(directory+"/df_"+str(i)+".npy"))
            wighted_feature = list(np.load(directory+"/wf_"+str(i)+".npy"))
            
            if len(payload) <= n_attempt:
                sum_of_rewards[i] = -1 * math.inf
                cluster_list.pop(cluster_list.index(i))
            j = 0
            
            print("cluster number ", i)
            
            if  len(s[i]) == 0:
                s[i] = np.ones(len(detailed_feature[0])).astype("float64")
                blocked[i] = np.zeros(len(detailed_feature[0])).astype("float64")
                initial = True
            else: 
                initial = False



            success_counter = 0
            while j < n_attempt and len(payload) > 0:
                if initial :
                    index = randrange(len(payload))
                    test_case = payload.pop(index)
                    feature = detailed_feature.pop(index)
                    wighted_feature.pop(index)
                    if not random_search:
                        initial = False
                else:
                    all_d = [min_distance(blocked[i], wighted_feature[k]) for k in range(len(payload))]
                    index = np.argmin(all_d)
                    test_case = payload.pop(index)
                    feature = detailed_feature.pop(index)
                    wighted_feature.pop(index)
                response, code = test_payload(test_case, url)
                if response:
                    m = np.multiply(s[i], feature)
                    s[i] = np.subtract(s[i],m)
                    blocked[i] = np.multiply(s[i], blocked[i])
                    success_list.append(test_case)
                    
                    j += 1
                    success_counter += 1
                    
                    sum_of_rewards[i] += r
                    
                else:
                    j += 1
                    feature = np.multiply(s[i], feature)
                    blocked[i] = np.add(blocked[i], feature)
                    sum_of_rewards[i] -= punishment
                total_req += 1
                print("cluster:"+str(i) + "-" + "total_req:"+str(total_req) + "-" +"uncovered:"+str(len(success_list)))
                result.append(len(success_list))
            per_cluster_times.append(time.time() - c_start)
            average_rewards[i] = sum_of_rewards[i] / number_of_games[i]
            np.save(directory+"/d_"+str(i), np.asarray(payload))
            np.save(directory+"/df_"+str(i), np.asarray(detailed_feature))
            np.save(directory+"/wf_"+str(i), np.asarray(wighted_feature))
        total_times.append(time.time() - start_time)
        np.save(output_dir+"/ep_"+str(nt), np.asarray(number_of_games))
        np.save(output_dir+"/r_"+str(nt), np.asarray(result))
        np.save(output_dir+"/bypassing_"+str(nt), np.asarray(success_list))
        np.save(output_dir+"/clusterTime_"+str(nt), np.asarray(per_cluster_times))
    np.save(output_dir+"/runtime", np.asarray(total_times))
            
            
        
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-c', '--cluster', action="store", type=int)
   parser.add_argument('-u', '--url', action="store")
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-o', '--output', action="store")
   args = parser.parse_args()
   oracle(args.url, args.input, args.output, args.cluster)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        