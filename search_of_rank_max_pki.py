#!/usr/bin/env python
# coding: utf-8

# In[6]:


import scipy
from scipy.stats import rankdata
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import pickle


# In[104]:


with open('results/selected_datasets.pickle', 'rb') as f:
    selected_datasets = pickle.load(f)


# In[106]:


def search_of_max_pki_in_test(dataset, method):
    data = []
    with open(f'x_and_y/{dataset}.pickle', "rb") as f:
        for _ in range(pickle.load(f)):
            data.append(pickle.load(f))
    x_train_ex, x_test_ex, y_r_test_ex, y_train_ex, y_test_ex = data[0], data[1], data[3], data[4], data[5]
    x_train_in, x_test_in, y_train_in, y_test_in, = train_test_split(x_train_ex, y_train_ex, test_size=len(y_train_ex)-10)
    max_pki_index_test = y_r_test_ex.idxmax()[0]
    max_pki_test = y_r_test_ex.max()[0]
    rank_of_max_pki = dict()
    iteration = 0
    iteration_of_max_pki='not found'
    while y_train_in.shape[0] < y_train_ex.shape[0]:
        rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
        rfc.fit(x_train_in.values, y_train_in['activity'].values)
        classes = set(y_train_in['activity'].values)

        probs = rfc.predict_proba(x_test_in.values)
        probs_difference, prob_of_label_1 = [], []
        for n, prob in enumerate(probs):
            try:
                probs_difference.append([x_test_in.index[n], abs(prob[0]-prob[1])])
                prob_of_label_1.append([x_test_in.index[n], prob[1]])
            except IndexError:
                probs_difference.append([x_test_in.index[n], 1])
                if 1 in classes:
                    prob_of_label_1.append([x_test_in.index[n], 1]) 
                else:
                    prob_of_label_1.append([x_test_in.index[n], 0])

        least_sure = [x[0] for x in sorted(probs_difference, key=lambda x: x[1], reverse=False)][:5]
        most_sure = [x[0] for x in sorted(prob_of_label_1, key=lambda x: x[1], reverse=True)][:5]

        if method == 'exploration':
            adding_points = least_sure
        elif method == 'exploitation':
            adding_points = most_sure
        elif method == 'mixed 1:4':
            adding_points = set(most_sure[0:4] + least_sure[0:1])
        elif method == 'mixed 2:3':
            adding_points = set(most_sure[0:3] + least_sure[0:2])
        else:
            try:
                adding_points = random.sample(list(x_test_in.index), 5)
            except ValueError:
                adding_points = list(x_test_in.index)

        for point in adding_points:
            x_train_in, y_train_in = x_train_in.append(x_test_in.loc[point]), y_train_in.append(y_test_in.loc[point])
            x_test_in, y_test_in = x_test_in.drop(point), y_test_in.drop(point)

        pred = rfc.predict(x_test_ex.values)
        proba = rfc.predict_proba(x_test_ex.values)
        if len(classes) == 1:
            if 1 in classes:
                proba = np.insert(proba, 0, 0, axis = 1)
            else:
                proba = np.insert(proba, 1, 0, axis = 1)

        prob_of_label_1_test = []
        for n, prob in enumerate(proba):
            prob_of_label_1_test.append([x_test_ex.index[n], prob[1]])
        
        ordered_probs = dict()
        ranked_probs = rankdata(prob_of_label_1_test, method='dense', axis=0)
        for n, el in enumerate(prob_of_label_1_test):
            ordered_probs[el[0]] = ranked_probs[n][1]
        rank_of_max_pki[iteration] = ordered_probs[max_pki_index_test]
        #print(iteration, rank_of_max_pki[iteration])
        
        while iteration_of_max_pki=='not found':
            number_of_top = 0
            top_indexes = []
            i=1
            while number_of_top<5:
                number_of_top += list(ordered_probs.values()).count(i)
                for index, rank in ordered_probs.items():
                    if rank == i:
                        top_indexes.append(index)
                #print(i, top_indexes, number_of_top)
                i+=1


            if max_pki_index_test in top_indexes:
                iteration_of_max_pki = iteration          
            
        iteration += 1
        
    return rank_of_max_pki, iteration_of_max_pki, max_pki_test


# In[107]:


def search_rank(best_or_worst_datasets):
    result = defaultdict(dict)
    for dataset in tqdm(selected_datasets[f'{best_or_worst_datasets}_ba']):
        for i in tqdm(range(10)): 
            result_for_iteration = dict()
            for method in tqdm(['exploration', 'exploitation', 'random', 'mixed 1:4', 'mixed 2:3']):  
            #for method in tqdm(['exploration', 'exploitation', 'random']):   
                result_for_iteration[method] = search_of_max_pki_in_test(dataset,method)
            result[dataset][i] = result_for_iteration
        with open(f'results_searching_rank_for_{best_or_worst_datasets}_ba.pickle', 'wb') as f:
            pickle.dump(result, f)


# In[ ]:


result_search_best_datasets_ba = search_rank('best')


# In[ ]:


result_search_worst_datasets_ba = search_rank('worst')


# In[ ]:




