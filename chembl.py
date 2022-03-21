#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[153]:


def descriptors(file):
    data = pd.read_csv(f'datasets/{file}', names=['smiles', 'chembl_id', 'pKi'])     
    data['activity'] = np.where(data.pKi >= 7, 1, 0)
    mols = [Chem.MolFromSmiles(str(m)) for m in data['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024) for m in mols]
    np_fps = []

    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        np_fps.append(arr)
    np_fps = pd.DataFrame(np_fps)
    
    return data, np_fps


# In[133]:


def model(file):
    data, np_fps = descriptors(file)

    x_train_external, x_test_external, y_c_train_external, y_c_test_external = train_test_split(fps, data['activity'].values.tolist(), test_size=0.2, random_state=42)

    rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)

    rfc.fit(x_train_external, y_c_train_external)
    y_c_pred = rfc.predict(x_test_external)
    ba = balanced_accuracy_score(y_c_test_external, y_c_pred)
    proba = rfc.predict_proba(x_test_external)
    fpr, tpr, _ = roc_curve(y_c_test_external, [x[1] for x in proba])
    AUC = auc(fpr, tpr)

    return ba, AUC


# In[24]:


results = dict()
for element in tqdm(os.listdir('/home/ilnura/active learning/datasets')):
    results[element] = model(element)


# In[17]:


df = pd.DataFrame(results)
df.columns = ['dataset', 'ba', 'auc']
datasets_ordred_by_quality_increase = list(df.sort_values(by=['ba'], ascending=False)['dataset'])
datasets_ordred_by_quality_decrease = list(df.sort_values(by=['ba'], ascending=True)['dataset'])


# In[44]:


best_ba, worst_ba = [], []

for dataset in datasets_ordred_by_quality_increase:
    if len(best_ba) < 10:
        data = pd.read_csv(f'datasets/{dataset}', names=['smiles', 'chembl_id', 'pKi'])     
        data['activity'] = np.where(data.pKi > 7, 1, 0)
        activity = data['activity'].values.tolist()
        if 0.1<=activity.count(0)/activity.count(1)<=10:
            best_ba.append(dataset)
        else:
            pass       


# In[145]:


for dataset in datasets_ordred_by_quality_decrease: 
    if len(worst_ba) < 10:
        data = pd.read_csv(f'datasets/{dataset}', names=['smiles', 'chembl_id', 'pKi'])     
        data['activity'] = np.where(data.pKi > 7, 1, 0)
        activity = data['activity'].values.tolist()
        if 0.1<=activity.count(0)/activity.count(1)<=10:
            worst_ba.append(dataset)
        else:
            pass


# In[15]:


with open ('best_ba.pickle', 'rb') as f:
    best_ba = pickle.load(f)
with open ('worst_ba.pickle', 'rb') as f:
    worst_ba = pickle.load(f)


# In[165]:


def data_preparation(file):
    data, np_fps = descriptors(file)
    
    x_train_external, x_test_external, y_train_external, y_test_external = train_test_split(np_fps, pd.DataFrame(data['activity']), test_size=0.2)
    x_train_internal, x_test_internal, y_train_internal, y_test_internal = train_test_split(x_train_external, y_train_external,
                                                                                                test_size=len(y_train_external)-10)
    pkis = dict()
    for index in x_test_internal.index:
        pkis[index] = data.iloc[index]['pKi']
    max_pki_index = max(pkis, key=pkis.get)
    return data, x_train_internal, x_test_internal, y_train_internal, y_test_internal, x_test_external, y_test_external, max_pki_index


# In[147]:


def active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, max_pki_index, method):
    results = []
    number_of_iterations = len(x_test_in)//5
    while len(results) < number_of_iterations:
        rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
        rfc.fit(x_train_in.loc[:, ].to_numpy(), y_train_in['activity'].values.tolist())
        classes = set(y_train_in['activity'].values.tolist())
        
        probs = rfc.predict_proba(x_test_in.loc[:, ].to_numpy())
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

        pred = rfc.predict(x_test_ex.loc[:, ].to_numpy())
        proba = rfc.predict_proba(x_test_ex.loc[:, ].to_numpy())
        if len(classes) == 1:
            if 1 in classes:
                proba = np.insert(proba, 0, 0, axis = 1)
            else:
                proba = np.insert(proba, 1, 0, axis = 1)
        ba = balanced_accuracy_score(y_test_ex.values.tolist(), pred)
        fpr, tpr, _ = roc_curve(y_test_ex.loc[:, ].to_numpy(), [x[1] for x in proba])
        AUC = auc(fpr, tpr)
        results.append([(len(y_train_in)-10)//5, ba, AUC])

        if method == 'exploration':
            adding_points = least_sure
        elif method == 'exploitation':
            adding_points = most_sure
        else:
            adding_points = random.sample(list(x_test_in.index), 5)

        if max_pki_index in adding_points:
            iteration_of_max_pki = (len(y_train_in)-10)//5
        # else:
        #     iteration_of_max_pki = 'not found'

        for point in adding_points:
            x_train_in, y_train_in = x_train_in.append(x_test_in.loc[point]), y_train_in.append(y_test_in.loc[point])
            x_test_in, y_test_in = x_test_in.drop(point), y_test_in.drop(point)
        #print(results[-1])

    return results, iteration_of_max_pki


# In[223]:


def search_of_max_pki(dataset, method):
    
    data, x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, _ = data_preparation(dataset)
    max_pki_index = data[['pKi']].idxmax()[0]
    max_pki = data[['pKi']].max()[0]
    
    random_point = random.choice(list(x_test_in.index))
    if max_pki_index in y_test_ex.index:
        # x_test_ex, y_test_ex = x_test_ex.append(x_test_in.loc[random_point]), y_test_ex.append(x_test_in.loc[random_point])       
        x_test_in, y_test_in = x_test_in.append(x_test_ex.loc[max_pki_index]), y_test_in.append(y_test_ex.loc[max_pki_index])
        # x_test_ex, y_test_ex = x_test_ex.drop(max_pki_index), y_test_ex.drop(max_pki_index)
        x_test_in, y_test_in = x_test_in.drop(random_point), y_test_in.drop(random_point)
        
    elif max_pki_index in y_train_in.index:
        x_test_in, y_test_in = x_test_in.append(x_train_in.loc[max_pki_index]), y_test_in.append(y_train_in.loc[max_pki_index])
        x_train_in, y_train_in = x_train_in.append(x_test_in.loc[random_point]), y_train_in.append(y_test_in.loc[random_point])              
        x_train_in, y_train_in = x_train_in.drop(max_pki_index), y_train_in.drop(max_pki_index)
        x_test_in, y_test_in = x_test_in.drop(random_point), y_test_in.drop(random_point)        
    
 
    while True:
        rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
        rfc.fit(x_train_in.loc[:, ].to_numpy(), y_train_in['activity'].values.tolist())
        classes = set(y_train_in['activity'].values.tolist())
        
        probs = rfc.predict_proba(x_test_in.loc[:, ].to_numpy())
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
        else:
            try:
                adding_points = random.sample(list(x_test_in.index), 5)
            except ValueError:
                adding_points = list(x_test_in.index)
                
        for point in adding_points:
            x_train_in, y_train_in = x_train_in.append(x_test_in.loc[point]), y_train_in.append(y_test_in.loc[point])
            x_test_in, y_test_in = x_test_in.drop(point), y_test_in.drop(point)

        if max_pki_index in adding_points:
            iteration_of_max_pki = (len(y_train_in)-10)//5
            break

    return iteration_of_max_pki, max_pki


# In[216]:


best_data_search_of_max_pki = defaultdict(dict)
for dataset in tqdm(best_ba):
    for i in tqdm(range(10)): 
        result_for_iteration = dict()
        for method in tqdm(['exploration', 'exploitation', 'random']):        
            result_for_iteration[method] = search_of_max_pki(dataset,method)
        best_data_search_of_max_pki[dataset][i] = result_for_iteration
    with open ('best_data_search_of_max_pki.pickle', 'wb') as f:
        pickle.dump(best_data_search_of_max_pki, f)


# In[224]:


worst_data_search_of_max_pki = defaultdict(dict)
for dataset in tqdm(worst_ba):
    for i in tqdm(range(10)): 
        result_for_iteration = dict()
        for method in tqdm(['exploration', 'exploitation', 'random']):        
            result_for_iteration[method] = search_of_max_pki(dataset,method)
        worst_data_search_of_max_pki[dataset][i] = result_for_iteration
    with open ('worst_data_search_of_max_pki.pickle', 'wb') as f:
        pickle.dump(worst_data_search_of_max_pki, f)


# In[113]:


results_for_best = defaultdict(dict)
for dataset in tqdm(best_ba):
    for method in tqdm(['exploration', 'exploitation', 'random']):
        result_for_iteration = []
        for i in tqdm(range(10)):
            x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, max_pki_index = data_preparation(dataset) 
            result_for_iteration.append(active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, max_pki_index, method))
        results_for_best[dataset][method] = result_for_iteration
    with open ('results_for_best.pickle', 'wb') as f:
        pickle.dump(results_for_best, f)


# In[150]:


results_for_worst = defaultdict(dict)
for dataset in tqdm(worst_ba):
    for method in tqdm(['exploration', 'exploitation', 'random']):
        result_for_iteration = []
        for i in tqdm(range(10)):
            x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, max_pki_index = data_preparation(dataset) 
            result_for_iteration.append(active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, max_pki_index, method))
        results_for_worst[dataset][method] = result_for_iteration
    with open ('results_for_worst.pickle', 'wb') as f:
        pickle.dump(results_for_worst, f)

