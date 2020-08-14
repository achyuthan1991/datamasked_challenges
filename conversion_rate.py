import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import h2o
from h2o.estimators import H2ORandomForestEstimator

sns.set()

df = pd.read_csv('../Challenges/1_conversion_data.csv')
# print(df.describe())
# age has outliers for 111 and 123 [based on plots below], only 2 elements
# preprocessing in pandas instead of h2o since I am new to h2o
df = df.loc[df.age < 111]
target_col = ["converted"]
features = list(set(df.columns) - set(target_col))
target_col = target_col[0]

for col in features:
    agg_df = df.groupby(col).agg(num_converted=(target_col, "sum"), num_cases=(target_col, "count"))
    agg_df['percent_convert'] = agg_df['num_converted']/agg_df['num_cases']*100
    # plt.figure()
    agg_df[['percent_convert']].plot.bar()
    plt.xlabel(col)
    plt.tight_layout()
    # break
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_col], test_size=0.2, random_state=42)

# clf = RandomForestClassifier(max_depth = 2, random_state = 0)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

h2o.init(ip="127.0.0.1", port="8080")
data = h2o.H2OFrame(df)
# Setting the target variable as a factor is important.
# this tells the model that target is not numeric, and enables classification related performance metrics
data[target_col] = data[target_col].asfactor()
data['new_user'] = data['new_user'].asfactor()
train, test = data.split_frame(ratios=[0.8])
model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
model.train(x=features, y=target_col, training_frame=train)
performance = model.model_performance(test_data=test)
# Evaluating the standard F1 metric on the test set
print("F1: ", performance.F1()) # prints best threshold and value of f1 for that threshold
# Evaluating the confusion matrix since data is highly imbalanced. This gives a better sense of the model performance
print(performance.confusion_matrix())
# since data is imbalanced, checking the area under the precision recall curve. See link for more metric explanations
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html
# Also printing the MCC (Mathew's correlation coefficient)
print("AUC for Precision Recall Curve:", performance.aucpr())
print("Mathew's Correlation Coefficient:", performance.mcc())
# printing all the metrics
print("Printing all metrics")
print(performance)
# Checking mathew's correlation coefficient (MCC is good for evaluating classification performance). See link for why:
# https://www.kdnuggets.com/2016/12/best-metric-measure-accuracy-classification-models.html/2
# The performance of the model is pretty decent, with an AUC for precision recall curve (AUCPR) being about 0.8
# and an MCC of 0.74 on the test set
# This solves the problem. From the confusion matrix, it can be seen that the model precision and recall are
# also fairly ok. More fine tuning can improve the results
h2o.cluster().shutdown()