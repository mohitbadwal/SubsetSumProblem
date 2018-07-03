import pandas as pd
import numpy as np
import os


dict_customer = {}

def LMH_assign(val):
    if val<=100:
        return 'L1'
    elif val<=500:
        return 'L2'
    elif val<=1000:
        return 'L3'
    elif val<=5000:
        return 'M'
    else:
        return 'H'

def smh_func(df, cust_number, dict_customer):
    df_customer = df[df['customer_number'].astype(str) == cust_number]
    k = pd.cut(df_customer['amount'], [0, 100, 500, 1000, 5000, 1000000], labels=['L1', 'L2', 'L3', 'M', 'H'])
    df_customer['invoice_type'] = k
    df_customer['invoice_count_per_category'] = df_customer.groupby('invoice_type')['invoice_type'].transform('count')
    df_customer['invoice_percent'] = (df_customer['invoice_count_per_category'] / df_customer.shape[0]) * 100
    smh = df_customer['invoice_type'].value_counts().index
    totals = df_customer['invoice_type'].value_counts().values
    count_total = sum(totals)
    percent = [(i / count_total) * 100 for i in totals]
    lst = []
    for i, j in zip(percent, totals):
        lst.append((i, j))
    smh = df_customer['invoice_type'].value_counts().index

    dict_smh = {}
    for i in range(0, len(smh)):
        dict_smh[smh[i]] = lst[i]

    # dict_customer={}
    dict_customer[df_customer['customer_number'].values[0]] = dict_smh
    return dict_customer


LMH_cut=[0, 100, 500, 1000, 5000, 1000000]

for i in os.listdir('D:\\SubsetSumProblem\\GeneratedFeaturesCSV'):
    print('readcsv ' + str(i))
    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\' + str(i), sep=',', index_col=0)

    cust_num=dataset['customer_number'].unique()[0]
    local = smh_func(dataset, str(cust_num) , dict_customer)

    print(local)
    print(cust_num)
    print(local.get(cust_num))
    print('L1=',local.get(cust_num).get('L1'))
    print('L2=', local.get(cust_num).get('L2'))
    print('L3=', local.get(cust_num).get('L3'))
    print('M=', local.get(cust_num).get('M'))
    print('H=', local.get(cust_num).get('H'))

    dataset['LMH'] = dataset['amount'].apply(LMH_assign)

    dataset['L1_perc']=0
    dataset['L2_perc'] = 0
    dataset['L3_perc'] = 0
    dataset['M_perc']=0
    dataset['H_perc']=0
    dataset['LMH_cumulative']=0

    for j in dataset['payment_id'].unique():
        for k in dataset[dataset['payment_id']==j]['subset_number'].unique():
            temp=dataset[(dataset['payment_id']==j) & (dataset['subset_number']==k)]
            L1=dataset.loc[(dataset['payment_id']==j) & (dataset['subset_number']==k),'L1_perc']=len(temp[temp['LMH']=='L1'])/len(temp)
            L2=dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'L2_perc'] = len(temp[temp['LMH'] == 'L2']) / len(temp)
            L3=dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'L3_perc'] = len(temp[temp['LMH'] == 'L3']) / len(temp)
            M=dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'M_perc'] = len(temp[temp['LMH'] == 'M']) / len(temp)
            H=dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'H_perc'] = len(temp[temp['LMH'] == 'H']) / len(temp)
            dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'LMH_cumulative']=L1*local.get(cust_num).get('L1')[0] + L2*local.get(cust_num).get('L2')[0] + L3*local.get(cust_num).get('L3')[0] + M*local.get(cust_num).get('M')[0] + H*local.get(cust_num).get('H')[0]

    dataset.to_csv("D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\" + str(i))