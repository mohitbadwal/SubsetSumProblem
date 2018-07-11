"""
    test.py created by mohit.badwal
    on 7/11/2018

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import itertools
import os
import itertools
from json import JSONDecodeError
import time
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import json


def temp(h, amounts, payment):
    result = []
    for seq in itertools.combinations(amounts, h):
        if sum(seq) == payment:
            result.append(seq)
    return result


# payment -> payment_amount
def ssum(amounts, payment, invoices, payment_id):
    # result = [i for h in range(len(amounts), 0, -1) for i in itertools.combinations(enumerate(amounts), h)
    #           if sum(x[1] for x in i) == payment]
    result = []

    for h in range(len(amounts), 0, -1):
        for i in itertools.combinations(enumerate(amounts), h):
            if sum(x[1] for x in i) == payment:
                li = [invoices[x[0]] for x in i]
                result.append([[x[1] for x in i], li, payment_id, payment])
    return result


def functionSubSet(i, data_):
    print("Started ", i)
    payment = data_[data_['payment_hdr_id'] == i]['payment_amount'].unique()[0]
    amounts = data_['invoice_amount_norm'].values
    invoices = data_['invoice_number_norm'].values
    # print(payment, amounts , invoices)
    sd = ssum(amounts, payment, invoices, i)
    # print(payment, sd)
    print("Ended", i)
    return sd


import json


def sssum(i, data_):
    print("Started", i, len(data_))
    try:

        if len(data_) > 30:
            raise Exception("Too many rows", len(data_))
        payment = data_[data_['payment_hdr_id'] == i]['payment_amount'].unique()[0]
        payment_date = data_[data_['payment_id'] == i]['effective_date'].unique()[0]
        customer_name = data_[data_['payment_id'] == i]['customer_number_norm'].unique()[0]
        datas = data_[data_['payment_id'] == i].loc[:, ['invoice_amount_norm', 'invoice_number_norm', 'due_date_norm',
                                                        'invoice_date_norm', 'payment_hdr_id']].values
        #    print(datas)
        amounts = datas[:, 0].tolist()
        amounts = [round(float(x), 5) for x in amounts]
        invoices = datas[:, 1].tolist()
        due_dates = datas[:, 2].tolist()
        invoice_dates = datas[:, 3].tolist()
        payment_hdr_id = datas[:, 4].tolist()
        data_inFunction = [amounts, invoices, due_dates, invoice_dates, payment_hdr_id]
        currents = '{"customer_name":"' + str(customer_name) + '","payment_id":"' + str(
            i) + '","payment_amount":' + str(
            payment) + ',"payment_date":"' + str(payment_date) + '","subsets":['
        current = ''
        li = []
        ssum_h(data_inFunction, len(amounts), current, payment, li, i, payment)
        s = ",".join(li)
        s = currents + s
        s = s[:-1]
        if len(li) > 0:
            s = s + ']]}'
        else:
            s = s + '[]}'
        try:
            dic = json.loads(s)
        except JSONDecodeError:
            print("s = ", s)
        print("Ended", i)
        return dic
    except Exception as e:
        print(e, " this error occurred for payment id  ", i)
        return {}


def sum_temp(lists, list1, sum1, id):
    li = []
    indices = [j for j, x in enumerate(list1) if x == "1"]
    if indices:
        k1 = [lists[0][q] for q in indices]
        temp_sum = sum(k1)
        if temp_sum == sum1:
            li.append([k1, [lists[1][q] for q in indices], sum1, id])
            return li


def sum_bits(lists, sum1, id):
    list1 = []
    for i in range(pow(2, len(lists[0]))):
        b = bin(i)[2:]
    l = len(b)
    b = str(0) * (len(lists[0]) - l) + b
    list1.append(b)
    p = ThreadPool(processes=2)
    res = [p.apply_async(sum_temp, args=(lists, i, sum1, id)) for i in list1]
    res = [rest.get() for rest in res]
    return res


def ssum_h(lists, n, subset, sum, li, iot, l):
    if sum == round(float(0), 5):
        subset = subset[:-1]
        subset = '[' + subset + ']'
        # print(subset)
        li.append(subset)
        # print("found ",len(str(subset).split('[')))
        return

    if n == 0:
        return
    if lists[0][n - 1] <= round(float(sum), 5):
        ssum_h(lists, n - 1, subset, sum, li, iot, l)

        ssum_h(lists, n - 1,
               subset + '{"amount":' + str(lists[0][n - 1]) + ',"invoice":"' + str((lists[1][n - 1])) +
               '","invoice_date":"' + str(lists[3][n - 1]) + '","due_date":"' + str(lists[2][n - 1]) +
               '","payment_hdr_id":"' + str(lists[4][n - 1]) + '"},',
               round(float(sum - lists[0][n - 1]), 2), li, iot, l)
    else:
        ssum_h(lists, n - 1, subset, sum, li, iot, l)


import numpy as np

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


if __name__ == '__main__':
    columns = ['invoice_amount_norm', 'payment_amount', 'effective_date', 'invoice_date_norm', 'due_date_norm',
               'payment_method', 'payment_hdr_id', 'payment_id', 'invoice_number_norm', 'customer_number_norm']

    # temp = pd.DataFrame()
    # for i in os.listdir('D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Mohit'):
    #     print('readcsv ' + str(i))
    #     temp = temp.append(
    #         pd.read_csv(r'D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Mohit\\' + str(i), sep=',', index_col=0),
    #         ignore_index=True)
    temp = pd.concat([pd.read_csv(r'D:\backup\PycharmProjects\test\caa_ml_03\Starbucks\Mohit_starbucks\4.csv').reset_index(drop=True),
                      pd.read_csv(r'D:\backup\PycharmProjects\test\caa_ml_03\Starbucks\Mohit_starbucks\5.csv').reset_index(drop=True),
                      pd.read_csv(r'D:\backup\PycharmProjects\test\caa_ml_03\Starbucks\Mohit_starbucks\6.csv').reset_index(drop=True),
                      pd.read_csv(r'D:\backup\PycharmProjects\test\caa_ml_03\Starbucks\Mohit_starbucks\7.csv').reset_index(drop=True)],
                     ignore_index=True)
    print(len(temp))
    # temp = reduce_mem_usage(temp)
    if temp['customer_number_norm'].dtype == np.float64:
        temp['customer_number_norm'] = temp['customer_number_norm'].astype(np.int64)
    temp['customer_number_norm'] = temp['customer_number_norm'].astype(str)
    with open(r'D:\backup\PycharmProjects\test\caa_ml_03\Starbucks\customer_level_features.json') as f:
        data_dict = json.load(f)

    temp['payment_date'] = pd.to_datetime(temp['effective_date'])
    temp['invoice_date'] = pd.to_datetime(temp['invoice_date_norm'])
    temp['delay'] = temp['payment_date'].subtract(temp['invoice_date'], axis=0)
    temp['delay'] = temp['delay'].apply(lambda x: pd.Timedelta(x).days)
    # temp['invoice_number_norm'] = temp['invoice_number_norm'].astype(np.int64)
    temp['invoice_number_norm'] = temp['invoice_number_norm'].astype(str)
    for i in temp['customer_number_norm'].unique():
        data = temp[temp['customer_number_norm'] == i]
        print(i)
        print(len(data))

        if data_dict[i]['max_payment_window'] == 0:
            data_dict[i]['max_payment_window'] = 400
        # data = data[data['delay'] <= data_dict[i]['max_payment_window']]
        data = data[data['payment_amount'] >= data['invoice_amount_norm']]

        print(len(data))
        if len(data) > 1:
            print("Data read")
            pool = mp.Pool(processes=1)
            transformed_dataframe = pd.DataFrame()
            payment_groups = data.groupby(by='payment_id')
            start = time.time()
            results = [pool.apply_async(sssum, args=(payment_ids, payments)) for payment_ids, payments in
                       payment_groups]
            results = [res.get() for res in results]
            # print(results)
            for r in results:
                if len(r.keys()) > 0:
                    f = r['subsets']
                    print(len(f))
                    if 0 < len(f) <= 5000:
                        payment_ids = r['payment_id']
                        payment_amount = r['payment_amount']
                        customer_name = r['customer_name']
                        payment_date = r['payment_date']
                        gr = pd.DataFrame()
                        for h, sub in enumerate(f):
                            for eachInvoice in sub:
                                e = pd.DataFrame(eachInvoice, index=[h])
                                # print(e)
                                e = pd.concat([e, pd.DataFrame({"payment_id": payment_ids}, index=[h]),
                                               pd.DataFrame({"payment_amount": payment_amount}, index=[h]),
                                               pd.DataFrame({"payment_date": payment_date}, index=[h]),
                                               pd.DataFrame({"customer_number": customer_name}, index=[h])], axis=1)
                                gr = pd.concat([gr, e])

                        transformed_dataframe = pd.concat(
                            [transformed_dataframe,
                             gr]
                            , ignore_index=False)
            # print(type(results[0]))
            end = time.time()
            print("------ Took {0} seconds-----".format(end - start))
            pool.close()
            pool.join()
            # print(transformed_dataframe)
            transformed_dataframe.to_csv("D:\\backup\\PycharmProjects\\test\\caa_ml_03\\Starbucks\\Generated\\" + str(i) + ".csv")