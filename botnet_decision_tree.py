from __future__ import print_function
import pandas as pd
from scapy.all import *
from IPython.display import display, HTML
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score



import os
import subprocess

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

pd.set_option('display.max_columns', None)

mal = ["192.168.2.112","131.202.243.84", "192.168.5.122" , "198.164.30.2","192.168.2.110","192.168.4.118","192.168.2.113","192.168.1.103","192.168.4.120","192.168.2.112","192.168.2.109","192.168.2.105","147.32.84.180","147.32.84.170","147.32.84.150","147.32.84.140","147.32.84.130","147.32.84.160","10.0.2.15","192.168.106.141","192.168.106.131","172.16.253.130","172.16.253.131","172.16.253.129","172.16.253.240","74.78.117.238"
,"158.65.110.24"
,"192.168.3.35"
,"192.168.3.25"
,"192.168.3.65"
,"172.29.0.116"
,"172.29.0.109"
,"172.16.253.132"
,"192.168.248.165"
,"10.37.130.4"]



df = pd.read_csv('ISCX_TRAIN2.csv')
df_test = pd.read_csv('ISCX_TEST.csv')

df = df[df.frame_len != "147.32.84.180"]
df = df[df.frame_len != "147.32.84.79"]
df = df[df.frame_len != "61.191.41.53"]
df_test = df_test[df_test.frame_len != "147.32.84.180"]
df_test = df_test[df_test.frame_len != "147.32.84.79"]
df_test = df_test[df_test.frame_len != "61.191.41.53"]
df['window_average'] = df['frame_len'].rolling(window=10, center=False).mean()
df_test['window_average'] = df_test['frame_len'].rolling(window=10, center=False).mean()

df = df.fillna(0)
df_test = df_test.fillna(0)
df['Botnet'] = df.apply(lambda x: 1 if x['ip.src'] in mal else 0, axis=1)
df_test['Botnet'] = df_test.apply(lambda x: 1 if x['ip.src'] in mal else 0, axis=1)


print("Training set: \n",df.head())
print(df.tail())

print("Test set: \n",df_test.head())
print(df_test.tail())


features = list(df.columns[5:7])
print("* features:", features, sep="\n")
y = df["Botnet"]
X = df[features]
y_test = df_test["Botnet"]
X_test = df_test[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99, max_depth=3)
dt.fit(X, y)

pred = dt.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,pred)*100)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f,
                    feature_names=features)

command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
try:
    subprocess.check_call(command)
except:
    exit("Could not run dot, ie graphviz, to "
         "produce visualization")

