import pandas as pd
import numpy as np
import os

def func(x, un_var):
    index = np.where(un_var == x)
    return index[0][0]


temporary = pd.DataFrame()
for i in os.listdir('D:\\SubsetSumProblem\\GeneratedFeaturesCSV'):
    print('readcsv ' + str(i))
    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\GeneratedFeaturesCSV\\' + str(i), sep=',',index_col=0)
    if len(dataset)>1:
        dataset['index']=dataset.index.values
        print(dataset['index'])
        dataset['payment_date'] = pd.to_datetime(dataset['payment_date'])

        dataset['due_date']= pd.to_datetime(dataset['due_date'])
        dataset['delay_due'] = dataset['payment_date'].subtract(dataset['due_date'], axis=0)
        dataset['delay_due'] = dataset['delay_due'].apply(lambda x: pd.Timedelta(x).days)

        print(len(dataset))
        group = dataset.groupby(by=['payment_id', 'index'])
        new_data = pd.DataFrame()
        for name, data in group:
            payment_id, subset_number = name[0], name[1]

            if len(data) > 1:
                data['delay_variance'] = data['delay_due'].var()

            else:
                data['delay_variance'] = 0


            print("here", name)
            new_data = pd.concat([new_data, data])

        grouped_payment_id = new_data.groupby(by=['payment_id'])
        new_data['delay_due_categorical'] = 0

        new_data_more_new = pd.DataFrame()
        for name, data in grouped_payment_id:
            unique_variance = data['delay_variance'].unique()
            print(unique_variance)
            unique_variance.sort()
            data['delay_due_categorical'] = data['delay_variance'].apply(func, args=(unique_variance,))
            new_data_more_new = pd.concat([new_data_more_new, data])


        new_data_more_new.to_csv("D:\\SubsetSumProblem\\Testing_payment-due_categorical\\" + str(i))
