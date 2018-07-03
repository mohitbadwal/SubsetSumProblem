import pandas as pd
import os
temp = pd.DataFrame()
final = pd.DataFrame()

variables = ['customer_number', 'payment_id', 'subset_number', 'output', 'average_delay',
'variance_categorical',
'L1_perc', 'L2_perc', 'L3_perc', 'M_perc', 'H_perc',
'LMH_cumulative',
'avg_of_invoices_closed',
'avg_of_all_delays',
'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
'payment_count_quarter_q4',
'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
'invoice_count_quarter_q4',
'LMH_payment', 'LMH_invoices', 'payment_amount', 'subset_number', 'number_invoices_closed',
'payment_date']
for i in os.listdir('D:\\SubsetSumProblem\\GeneratedFeaturesCSV'):
    final = pd.DataFrame()
    print('readcsv ' + str(i))
    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\' + str(i), sep=',', index_col=0)
    for j in dataset['payment_id'].unique():
        for k in dataset[dataset['payment_id'] == j]['subset_number'].unique():
            temp = dataset[(dataset['payment_id'] == j) & (dataset['subset_number'] == k)][variables]
            print(temp.iloc[[0]])
            final = final.append(temp.iloc[[0]], ignore_index=True)
    final.to_csv("D:\\SubsetSumProblem\\Subset_level\\" + str(i))