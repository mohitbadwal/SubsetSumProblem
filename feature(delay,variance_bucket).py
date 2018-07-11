import pandas as pd
import numpy as np
import os

def func(x, un_var):
    index = np.where(un_var == x)
    return index[0][0]


temporary = pd.DataFrame()
for i in os.listdir('D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Starbucks\\Generated'):
    print('readcsv ' + str(i))
    dataset = pd.read_csv(r'D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Starbucks\\Generated\\' + str(i), sep=',',index_col=0)
    if len(dataset)>1:
        dataset['index']=dataset.index.values
        print(dataset['index'])
        dataset['payment_date'] = pd.to_datetime(dataset['payment_date'])
        dataset['invoice_date'] = pd.to_datetime(dataset['invoice_date'])
        dataset['delay'] = dataset['payment_date'].subtract(dataset['invoice_date'], axis=0)
        dataset['delay'] = dataset['delay'].apply(lambda x: pd.Timedelta(x).days)

        print(len(dataset))
        group = dataset.groupby(by=['payment_id', 'index'])
        number_of_invoices = len(dataset['invoice'].unique())
        new_data = pd.DataFrame()
        for name, data in group:
            payment_id, subset_number = name[0], name[1]
            data['number_invoices_closed'] = len(data) / number_of_invoices
            data['output'] = 0

            if len(data) > 1:
                data['variance'] = data['delay'].var()
                if data['payment_hdr_id'].unique()[0] == payment_id and data['payment_hdr_id'].var() == 0:
                    data['output'] = 1
            else:
                data['variance'] = 0
                if data['payment_hdr_id'].unique()[0] == payment_id:
                    data['output'] = 1
            data['average_delay'] = data['delay'].mean()

            print("here", name)
            data['payment_id'] = payment_id
            data['subset_number'] = subset_number
            new_data = pd.concat([new_data, data])
            print("here 2", name)
            # new_data = pd.concat(
            #     [new_data, pd.DataFrame({"payment_id": payment_id, "subset_number": subset_number}, index=[0])], axis=1,
            #     ignore_index=True)

        grouped_payment_id = new_data.groupby(by=['payment_id'])
        new_data['variance_categorical'] = 0

        new_data_more_new = pd.DataFrame()
        for name, data in grouped_payment_id:
            unique_variance = data['variance'].unique()
            print(unique_variance)
            data['payment_id'] = name
            unique_variance.sort()
            data['variance_categorical'] = data['variance'].apply(func, args=(unique_variance,))
            new_data_more_new = pd.concat([new_data_more_new, data])
            new_data_more_new.to_csv("D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Starbucks\\Mohit_starbucks\\Generated\\" + str(i))