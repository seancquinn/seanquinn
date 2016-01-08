# -*- coding: utf-8 -*-
from __future__ import division
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import KFold
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.neighbors import KNeighborsClassifier as KNN
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
