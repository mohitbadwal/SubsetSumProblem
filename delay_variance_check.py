import pandas as pd
import numpy as np
import os
def func(x, un_var):
    index = np.where(un_var == x)
    return index[0][0]

for i in os.listdir('D:\\SubsetSumProblem\\GeneratedFeaturesCSV'):
    print('readcsv ' + str(i))
    data = pd.read_csv(r'D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\' + str(i), sep=',',index_col=0)
    if len(data)>1:

        data.drop(labels='avg_delay_categorical',inplace=True,axis=1)
        grouped_payment_id = data.groupby(by=['payment_id'])
        new_data_final=pd.DataFrame()
        for name,group in grouped_payment_id:
            unique_delay = group['average_delay'].unique()
            print(type(unique_delay),unique_delay)
            sorted_delay = np.asarray(sorted(unique_delay, reverse=True))
            print(type(sorted_delay),sorted_delay)
            group['avg_delay_categorical'] = group['average_delay'].apply(func, args=(sorted_delay,))
            new_data_final = pd.concat([new_data_final, group], axis=0)
        new_data_final.to_csv("D:\\Subset_new\\" + str(i))