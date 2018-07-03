import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np
features = ['average_delay',
            'variance_categorical',
            # 'L1_perc','L2_perc','L3_perc','M_perc','H_perc',
            'LMH_cumulative',
            'avg_of_invoices_closed',
            'avg_of_all_delays',
            'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
            'payment_count_quarter_q4',
            'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
            'invoice_count_quarter_q4',
            'number_invoices_closed']

data = pd.read_csv(r"D:\\SubsetSumProblem\\train_test_subset_gst\\data_70.csv", sep=',')
data2 = pd.read_csv(r"D:\\SubsetSumProblem\\train_test_subset_gst\\data2_30.csv", sep=',')

lb_make = LabelEncoder()
data['LMH_invoices'] = lb_make.fit_transform(data['LMH_invoices'])
data2['LMH_invoices'] = lb_make.transform(data2['LMH_invoices'])
data['LMH_payment'] = lb_make.transform(data['LMH_payment'])
data2['LMH_payment'] = lb_make.transform(data2['LMH_payment'])

X_train = data[features]
y_train = data['output']
X_validation = data2[features]
y_validation = data2['output']

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

predictions = rfc.predict(X_validation)
predictions_prob = rfc.predict_proba(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

data2['predictions'] = predictions
for i in range(0, data2.shape[0]):
    data2.at[i, 'pred_proba_0'] = predictions_prob[i][0]
    data2.at[i, 'pred_proba_1'] = predictions_prob[i][1]
data2.to_csv("D:\\SubsetSumProblem\\train_test_subset_gst\\check.csv")

# FEATURE IMPORTANCE
importances = rfc.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

cor = data.append(data2, ignore_index=True).corr()
print(cor)