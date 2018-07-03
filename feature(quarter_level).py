import json
import pandas as pd
import numpy as np
import os

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC


def LMH_assign(val):
    if val <= 100:
        return 'L1'
    elif val <= 500:
        return 'L2'
    elif val <= 1000:
        return 'L3'
    elif val <= 5000:
        return 'M'
    else:
        return 'H'


dictionary = json.load(open("D:\ML_03_Subset_Sum\customer_level_features.json", 'r'))

for i in os.listdir('D:\\SubsetSumProblem\\GeneratedFeaturesCSV'):
    print('readcsv ' + str(i))

    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\' + str(i), sep=',', index_col=0)

    cust_num = str(i).split('.')[0]

    print(cust_num)

    dataset['avg_of_invoices_closed'] = dictionary.get(cust_num).get('avg_of_invoices_closed')

    dataset['avg_of_all_delays'] = dictionary.get(cust_num).get('avg_of_all_delays')

    dataset['payment_count_quarter_q1'] = dictionary.get(cust_num).get('payment_count_quarter').get('q1')
    dataset['payment_count_quarter_q2'] = dictionary.get(cust_num).get('payment_count_quarter').get('q2')
    dataset['payment_count_quarter_q3'] = dictionary.get(cust_num).get('payment_count_quarter').get('q3')
    dataset['payment_count_quarter_q4'] = dictionary.get(cust_num).get('payment_count_quarter').get('q4')

    dataset['invoice_count_quarter_q1'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q1')
    dataset['invoice_count_quarter_q2'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q2')
    dataset['invoice_count_quarter_q3'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q3')
    dataset['invoice_count_quarter_q4'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q4')

    dataset['avg_of_invoices_closed'] = dictionary.get(cust_num).get('avg_of_invoices_closed')
    dataset['avg_of_all_delays'] = dictionary.get(cust_num).get('avg_of_all_delays')

    dataset['LMH_payment'] = dataset['amount'].apply(LMH_assign)

    dataset['LMH_invoices'] = dataset['LMH']

    dataset.to_csv("D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\" + str(i))