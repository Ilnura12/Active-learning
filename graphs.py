#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pickle
import pandas as pd
import matplotlib
from collections import defaultdict


# In[557]:


def result_of_search(file):
    lst = []
    for dataset in file:
        for el in file[dataset]:
            for method in file[dataset][el]:
                lst.append([dataset, method, file[dataset][el][method][0], file[dataset][el][method][1]])
    search = pd.DataFrame(lst)
    search.columns = ['dataset', 'method', 'iteration', 'max_pKi']
    search = search.set_index(['dataset', 'max_pKi'], append=False)
    return search


# In[558]:


search_worst = result_of_search(search_of_max_pki_worst)
search_best = result_of_search(search_of_max_pki_best)


# In[609]:


for dataset in search_of_max_pki_worst.keys():  
    plt = search_worst.loc[f'{dataset}'].boxplot(column='iteration', by='method')
    minor_ticks = np.arange(0, max(search_worst.loc[f'{dataset}']['iteration']), 10) 
    plt.set_yticks(minor_ticks, minor=True)
    plt.set_ylabel('number of iterations')
    plt.set_xlabel('method')
    plt.set_ylim(0,None)
    plt.set_title(' ')
    plt.grid(True, which='minor', alpha=1)
    fig = plt.get_figure()
    fig.suptitle(f'{dataset}')
    fig.show()
    fig.savefig(dataset + ' boxplot.png', dpi=600, bbox_inches='tight', facecolor='w')


# In[371]:


all_pki = pd.DataFrame()
names = []
for dataset in worst:
    data = descriptors(dataset)
    names.append(dataset.replace('CHEMBL', '').replace('.smi', ''))
    all_pki = pd.concat([pd.DataFrame(data['pKi']), all_pki], axis = 1)


all_pki.columns = names

boxplot = all_pki.boxplot()
fig = boxplot.get_figure()
    # fig.suptitle(f'{dataset}')
fig.savefig('boxplot_worst.png', dpi=600, bbox_inches='tight', facecolor='w')


# In[138]:


def graph_median(dataset, best_or_worst):
    methods = ['exploitation', 'exploration', 'random']
    median_results = pd.DataFrame()
    for method in methods:
        df = pd.DataFrame(results[dataset][f'{method}'][0][0])
        df.columns = ['iteration', 'ba', 'auc']
        df = df[['iteration', 'auc']]
        for i in range (1, 10):
            one_iteration = pd.DataFrame(results[dataset][f'{method}'][i][0])
            one_iteration.columns = ['iteration', 'ba', 'auc']
            df = df.merge(one_iteration[['iteration', 'auc']], on=['iteration'])
        columns = [x for x in range(1, 11)]
        columns.insert(0, 'iteration')
        df.columns = columns
        df['median'] = df.iloc[:, 1:11].median(axis=1)
        median_results = pd.concat([median_results, df['median']], axis=1)
    median_results.columns = methods
    median_results['iteration'] = df['iteration']
    plt = median_results.plot(x='iteration', grid=True, title = f'{dataset}')
    plt.axhline(res_on_all_dataset[dataset][1], color='black')
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=1)
    plt.set_ylabel('AUC')
    plt.set_xlabel('iteration')
    plt.set_ylim(None, 1)
    plt.figure.savefig(dataset + best_or_worst + '.png', dpi=600, bbox_inches='tight', facecolor='w')


# In[382]:


def graph_median_ba(dataset, best_or_worst):
    methods = ['exploitation', 'exploration', 'random']
    median_results = pd.DataFrame()
    for method in methods:
        df = pd.DataFrame(results[dataset][f'{method}'][0][0])
        df.columns = ['iteration', 'ba', 'auc']
        df = df[['iteration', 'ba']]
        for i in range (1, 10):
            one_iteration = pd.DataFrame(results[dataset][f'{method}'][i][0])
            one_iteration.columns = ['iteration', 'ba', 'auc']
            df = df.merge(one_iteration[['iteration', 'ba']], on=['iteration'])
        columns = [x for x in range(1, 11)]
        columns.insert(0, 'iteration')
        df.columns = columns
        df['median'] = df.iloc[:, 1:11].median(axis=1)
        median_results = pd.concat([median_results, df['median']], axis=1)
    median_results.columns = methods
    median_results['iteration'] = df['iteration']
    plt = median_results.plot(x='iteration', grid=True, title = f'{dataset}')
    plt.axhline(res_on_all_dataset[dataset][0], color='black')
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=1)
    plt.set_ylabel('ba')
    plt.set_xlabel('iteration')
    plt.set_ylim(None, 1)
    plt.figure.savefig(dataset + best_or_worst + '.png', dpi=600, bbox_inches='tight', facecolor='w')


# In[383]:


for dataset in worst:
    graph_median_ba(dataset, 'worst')


# In[ ]:




