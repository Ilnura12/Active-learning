#!/usr/bin/env python
# coding: utf-8

# In[145]:


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
import seaborn as sns
from CGRtools import RDFRead
from CGRtools.files import RDFWrite


# In[161]:


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
    
    x_train_external, x_test_external, y_r_train_external, y_r_test_external = train_test_split(np_fps, pd.DataFrame(data['pKi']), test_size=0.2, random_state=42)
    x_train_external, x_test_external, y_c_train_external, y_c_test_external = train_test_split(np_fps, pd.DataFrame(data['activity']), test_size=0.2, random_state=42)
    
    return x_train_external, x_test_external, y_r_train_external, y_r_test_external, y_c_train_external, y_c_test_external


# In[26]:


def model(x_train_external, x_test_external, y_r_train_external, y_r_test_external, y_c_train_external, y_c_test_external):
    
    rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
    rfr = RandomForestRegressor(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
    
    rfc.fit(x_train_external, y_c_train_external)
    rfr.fit(x_train_external, y_r_train_external)
    
    y_c_pred = rfc.predict(x_test_external)
    ba = balanced_accuracy_score(y_c_test_external, y_c_pred)
    proba = rfc.predict_proba(x_test_external)
    fpr, tpr, _ = roc_curve(y_c_test_external, [x[1] for x in proba])
    AUC = auc(fpr, tpr)
    
    y_r_pred = rfr.predict(x_test_external)
    r2 = r2_score(y_r_test_external, y_r_pred)
    rmse = mean_squared_error(y_r_test_external, y_r_pred, squared=False)

    return ba, AUC, r2, rmse


# In[27]:


results = dict()
for element in tqdm(os.listdir('/home/ilnura/active learning/datasets')):
    x_train_external, x_test_external, y_r_train_external, y_r_test_external, y_c_train_external, y_c_test_external = descriptors(element)
    results[element] = model(x_train_external, x_test_external, y_r_train_external, y_r_test_external, y_c_train_external, y_c_test_external)
with open ('results/result_on_all_dataset.pickle', 'wb') as f:
    pickle.dump(results, f)


# In[ ]:





# In[140]:


df = pd.DataFrame(results).T
df.columns = ['ba', 'auc', 'r2', 'rmse']
best_ba = list(df.sort_values(by=['ba'], ascending=False).index)[:10]
worst_ba = list(df.sort_values(by=['ba'], ascending=True).index)[:10]
best_r2 = list(df.sort_values(by=['r2'], ascending=False).index)[:10]
worst_r2 = list(df.sort_values(by=['r2'], ascending=True).index)[:10]
selected_datasets = {'best_ba': best_ba, 'worst_ba': worst_ba, 'best_r2': best_r2, 'worst_r2': worst_r2}
with open ('results/selected_datasets.pickle', 'wb') as f:
    pickle.dump(selected_datasets, f)


# In[158]:


datasets = set()
for element in selected_datasets.values():
    for dataset in element:
        datasets.add(dataset)


# In[162]:


for file in datasets:
    x_and_ys = descriptors(file)    
    with open(f'x_and_y/{file}.pickle', "wb") as f:
        pickle.dump(len(x_and_ys), f)
        for value in x_and_ys:
            pickle.dump(value, f)


# In[235]:


def active_learning_classification(dataset, method):
    data = []
    with open(f'x_and_y/{dataset}.pickle', "rb") as f:
        for _ in range(pickle.load(f)):
            data.append(pickle.load(f))
    x_train_ex, x_test_ex, y_train_ex, y_test_ex = data[0], data[1], data[4], data[5]
    x_train_in, x_test_in, y_train_in, y_test_in, = train_test_split(x_train_ex, y_train_ex, test_size=len(y_train_ex)-10)
    results = []
    iteration = 0
    while y_train_in.shape[0] <= y_train_ex.shape[0]:
        
        rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
        rfc.fit(x_train_in.values, y_train_in['activity'].values)
        classes = set(y_train_in['activity'].values)
        
        pred = rfc.predict(x_test_ex.values)
        proba = rfc.predict_proba(x_test_ex.values)
        if len(classes) == 1:
            if 1 in classes:
                proba = np.insert(proba, 0, 0, axis = 1)
            else:
                proba = np.insert(proba, 1, 0, axis = 1)
        ba = balanced_accuracy_score(y_test_ex.values, pred)
        fpr, tpr, _ = roc_curve(y_test_ex.values, [x[1] for x in proba])
        AUC = auc(fpr, tpr)
        results.append([iteration, ba, AUC])
        # print(results[-1], len(x_train_in), len(x_test_in))
        
        if y_test_in.shape[0] != 0:
        
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
            else:
                try:
                    adding_points = random.sample(list(x_test_in.index), 5)
                except ValueError:
                    adding_points = list(x_test_in.index)

            for point in adding_points:
                x_train_in, y_train_in = x_train_in.append(x_test_in.loc[point]), y_train_in.append(y_test_in.loc[point])
                x_test_in, y_test_in = x_test_in.drop(point), y_test_in.drop(point)
        else:
            break

        iteration += 1 

    return results


# In[243]:


#result_on_all_dataset_selected_data = dict()
for dataset in tqdm(selected_datasets['worst_ba']):
    data = []
    with open(f'x_and_y/{dataset}.pickle', "rb") as f:
        for _ in range(pickle.load(f)):
            data.append(pickle.load(f))
    x_train_ex, x_test_ex, y_train_ex, y_test_ex = data[0], data[1], data[4], data[5]
    rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
    rfc.fit(x_train_ex.values, y_train_ex['activity'].values)
    pred = rfc.predict(x_test_ex.values)
    proba = rfc.predict_proba(x_test_ex.values)
    ba = balanced_accuracy_score(y_test_ex.values, pred)
    fpr, tpr, _ = roc_curve(y_test_ex.values, [x[1] for x in proba])
    AUC = auc(fpr, tpr)
    result_on_all_dataset_selected_data[dataset] = [ba, AUC]


# In[236]:


def classification(best_or_worst_datasets):
    result = defaultdict(dict)
    for dataset in tqdm(selected_datasets[f'{best_or_worst_datasets}_ba']):
        for i in tqdm(range(10)): 
            result_for_iteration = dict()
            #for method in tqdm(['exploration', 'exploitation', 'random', 'mixed 1:4', 'mixed 2:3']):  
            for method in tqdm(['exploration', 'exploitation', 'random']):   
                result_for_iteration[method] = active_learning_classification(dataset,method)
            result[dataset][i] = result_for_iteration
        with open(f'results_for_{best_or_worst_datasets}_ba.pickle', 'wb') as f:
            pickle.dump(result, f)


# In[ ]:


result_best_datasets_ba = classification('best')


# In[ ]:


result_worst_datasets_ba = classification('worst')


# In[ ]:




