#!/usr/bin/env python
# coding: utf-8

# In[148]:


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
from collections import defaultdict


# In[14]:


def model(file):
    data = pd.read_csv(f'datasets/{file}', names=['smiles', 'chembl_id', 'pKi'])     
    data['activity'] = np.where(data.pKi > 7, 1, 0)

    mols = [Chem.MolFromSmiles(str(m)) for m in data['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024) for m in mols]
    np_fps = []

    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        np_fps.append(arr)

    # pKi = data['pKi'].values.tolist()
    activity = data['activity'].values.tolist()

    # x_train_external, x_test_external, y_r_train_external, y_r_test_external = train_test_split(np_fps, pKi, test_size=0.2, random_state=42)
    x_train_external, x_test_external, y_c_train_external, y_c_test_external = train_test_split(np_fps, activity, test_size=0.2, random_state=42)

    rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
    # rfr = RandomForestRegressor(random_state=1, n_estimators=500, max_features='log2', n_jobs=20)

    rfc.fit(x_train_external, y_c_train_external)
    y_c_pred = rfc.predict(x_test_external)
    ba = balanced_accuracy_score(y_c_test_external, y_c_pred)

    # rfr.fit(x_train_external, y_r_train_external)
    # y_r_pred = rfc.predict(x_test_external)
    # r2 = r2_score(y_r_test_external, y_r_pred)
    # rmse = mean_squared_error(y_r_test_external, y_r_pred, squared=False)
    #return ba, r2, rmse
    return ba


# In[15]:


results = []
#datasets = os.listdir('/home/ilnura/active learning/datasets')
for element in tqdm(os.listdir('/home/ilnura/active learning/datasets')):
    # result = list(model(element))
    # result.append(element)
    results.append([element, model(element)])


# In[17]:


df = pd.DataFrame(results)
# df.columns = ['ba', 'r2', 'rmse', 'datasdatasets/
df.columns = ['dataset', 'ba']
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


# In[47]:


def data_preparation(file):
    data = pd.read_csv(f'datasets/{file}', names=['smiles', 'chembl_id', 'pKi'])     
    data['activity'] = np.where(data.pKi > 7, 1, 0)

    mols = [Chem.MolFromSmiles(str(m)) for m in data['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024) for m in mols]
    np_fps = []

    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        np_fps.append(arr)


    activity = data['activity'].values.tolist()
    x_train_external, x_test_external, y_train_external, y_test_external = train_test_split(np_fps, activity, test_size=0.2, random_state=42)
    x_train_internal, x_test_internal, y_train_internal, y_test_internal = train_test_split(x_train_external, y_train_external,
                                                                                                test_size=len(y_train_external)-10, random_state=2, stratify=y_train_external)
    
    return x_train_internal, x_test_internal, y_train_internal, y_test_internal, x_test_external, y_test_external


# In[154]:


def active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex, method):
    results = []
    number_of_iterations = len(x_test_in)//5
    while len(results) < number_of_iterations:
        rfc = RandomForestClassifier(random_state=42, n_estimators=500, max_features='log2', n_jobs=20)
        rfc.fit(x_train_in, y_train_in)
        probs = rfc.predict_proba(x_test_in)

        probs_difference = []
        prob_of_label_1 = []
        for n, prob in enumerate(probs):
            probs_difference.append([n, abs(prob[0]-prob[1])])
            prob_of_label_1.append([n, prob[1]])
        least_sure = [x[0] for x in sorted(probs_difference, key = lambda x: x[1], reverse=False)][:5]
        least_sure.sort(reverse=True)
        most_sure = [x[0] for x in sorted(prob_of_label_1, key = lambda x: x[1], reverse=True)][:5]
        most_sure.sort(reverse=True)
 
        pred = rfc.predict(x_test_ex)
        ba = balanced_accuracy_score(y_test_ex, pred)
        fpr, tpr, _ = roc_curve(y_test_ex, pred)
        AUC = auc(fpr, tpr)
        results.append([len(y_train_in),ba, AUC])
        
        if method == 'exploration':
            adding_points = least_sure
        else:
            adding_points = most_sure
            
        for point in adding_points:
            x_train_in.append(x_test_in[point])
            y_train_in.append(y_test_in[point])
            del x_test_in[point]
            del y_test_in[point]
        # print(results[-1])    
        
    return results


# In[135]:


#results_for_best = defaultdict(dict)
for dataset in tqdm(best_ba[1:]):
    x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex = data_preparation(dataset) 
    results_for_best[dataset]['exploration'] = active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex,'exploration')
    x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex = data_preparation(dataset) 
    results_for_best[dataset]['exploitation'] = active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex,'exploitation')
    


# In[147]:


results_for_worst = defaultdict(dict)
for dataset in tqdm(worst_ba):
    x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex = data_preparation(dataset)
    results_for_worst[dataset]['exploration'] = active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex,'exploration')
    x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex = data_preparation(dataset)
    results_for_worst[dataset]['exploitation'] = active_learning(x_train_in, x_test_in, y_train_in, y_test_in, x_test_ex, y_test_ex,'exploitation')


# In[ ]:




