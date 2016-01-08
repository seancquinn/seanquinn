# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import urllib
import requests
import os
import json
import time
import tempfile

endpoint = "https://api.civisanalytics.com"
api_key = os.environ['CIVIS_API_KEY']
script = requests.post('https://api.civisanalytics.com/scripts/sql/',
                       auth=requests.auth.HTTPBasicAuth(api_key, ''),
                       json={"name":"CS Python Export","sql":"select state, account_length, area_code, phone, intl_plan, vmail_plan, vmail_message, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, night_calls, night_charge, intl_mins, intl_calls, intl_charge, custserv_calls, churn from public.telecom_customer_data limit 3300;","remoteHostId":308,"credentialId":2076}).json()

script_id = script['id']

run = requests.post(urllib.parse.urljoin(endpoint,'/scripts/sql/%d/runs' % script_id),auth=requests.auth.HTTPBasicAuth(api_key, '')).json()

run_id = run['id']

print("Script ID %d, run ID %d" % (script_id, run_id))

script_run = requests.get(urllib.parse.urljoin(endpoint,'/scripts/sql/%d/runs/%d' % (script_id,run_id)),
                       auth=requests.auth.HTTPBasicAuth(api_key, ''),
                       json={}).json()

while script_run['state'] in ('queued','running'):
    time.sleep(10)
    script_run = requests.get(urllib.parse.urljoin(endpoint,'/scripts/sql/%d/runs/%d' % (script_id,run_id)),
                       auth=requests.auth.HTTPBasicAuth(api_key, ''),
                       json={}).json()


script_history = requests.get(urllib.parse.urljoin(endpoint, '/scripts/%d/history' % script_id),
    auth=requests.auth.HTTPBasicAuth(api_key, ''),
    json = {}).json()


file_loc = script_history[0]['output'][0]['path']
print(file_loc)

with tempfile.NamedTemporaryFile() as temp:
	response = requests.get(file_loc).iter_content(10000)
	for r in response:
		temp.write(r)
	churn_df = pd.read_csv(temp.name)

col_names = churn_df.columns.tolist()

churn_result = churn_df['churn']
y = np.where(churn_result == 'True.', 1, 0)

to_drop = ['state','area_code','phone','churn']
churn_feat_space = churn_df.drop(to_drop, axis=1)

yes_no_cols = ["intl_plan", "vmail_plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("%d rows and %d features" % X.shape)
print("targets:", np.unique(y))


def run_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

print("Support vector machines:")
print("%.3f" % accuracy(y, run_cv(X, y, SVC)))
print("Random forest:")
print("%.3f" % accuracy(y, run_cv(X, y, RF)))
print("K-nearest-neighbors:")
print("%.3f" % accuracy(y, run_cv(X, y, KNN)))
