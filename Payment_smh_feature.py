import json
import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

def SMH_assign(low,high):
    interval=(high-low)/3
    print(interval)
    SMH_intervals = []
    SMH_intervals.insert(0,low)
    SMH_intervals.insert(1, low+interval)
    SMH_intervals.insert(2, low + 2 * interval)
    SMH_intervals.insert(3,high)
    return SMH_intervals

def SMH_assign_values(SMH_payment_bins,payment_amount):
    val1 = SMH_payment_bins[1]
    val2 = SMH_payment_bins[2]
    if payment_amount<=val1:
        return 0
    elif payment_amount<=val2:
        return 1
    else:
        return 2

for i in os.listdir('D:\\SubsetSumProblem\\Subset_level'):
    print('readcsv ' + str(i))

    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\Subset_level\\' + str(i), sep=',', index_col=0)
    if(len(dataset)>0):
        dataset['SMH_payment_bins']=0

        uniq_pay_amount = dataset['payment_amount'].unique()
        uniq_pay_amount = sorted(uniq_pay_amount)


        low=uniq_pay_amount[0]
        high=uniq_pay_amount[len(uniq_pay_amount)-1]

        SMH_payment_bins=SMH_assign(low,high)

        print(SMH_payment_bins)

        for j in dataset['payment_id'].unique():
            temp=dataset[dataset['payment_id']==j]
            payment_amount=temp['payment_amount'].unique()[0]
            cat=SMH_assign_values(SMH_payment_bins,payment_amount)
            dataset.loc[dataset['payment_id']==j,'SMH_payment_bins']=cat
        dataset.to_csv("D:\\SubsetSumProblem\\Subset_level\\" + str(i))