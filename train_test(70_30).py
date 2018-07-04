import pandas as pd
import os
data = pd.DataFrame()
data2 = pd.DataFrame()
count = 0
customers = 0


for i in os.listdir('D:\\SubsetSumProblem\\Subset_level'):
    print('readcsv ' + str(i))
    dataset = pd.read_csv(r'D:\\SubsetSumProblem\\Subset_level\\' + str(i), sep=',', index_col=0)
    if(len(dataset)!=0):

        dataset=dataset.sample(frac=1)
        count = 0
        customers = customers + 1
        print(customers)
        n_payments = len(dataset['payment_id'].unique())

        for j in dataset['payment_id'].unique():
            count = count + 1
            if (count / n_payments) <= 0.7:
                data = data.append(dataset[dataset['payment_id'] == j], ignore_index=True)
            else:
                data2 = data2.append(dataset[dataset['payment_id'] == j], ignore_index=True)
        data.to_csv("D:\\SubsetSumProblem\\train_test_subset_gst\\data_70.csv")
        data2.to_csv("D:\\SubsetSumProblem\\train_test_subset_gst\\data2_30.csv")