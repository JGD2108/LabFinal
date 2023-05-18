import socket
import pickle
import struct
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score, auc, roc_curve, classification_report 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from holoviews import opts
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the server's IP address and port
SERVER_IP = '192.168.80.11'
SERVER_PORT = 1234

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the IP address and port
server_socket.bind((SERVER_IP, SERVER_PORT))

# Listen for incoming connections
server_socket.listen(1)
print('Server is listening for connections...')

# Accept a client connection
client_socket, client_address = server_socket.accept()
print('Client connected:', client_address)

# Receive the header from the client
header_data = b''
while len(header_data) < 4:
    chunk = client_socket.recv(4 - len(header_data))
    if not chunk:
        break
    header_data += chunk

# Unpack the header
data_length = struct.unpack('!I', header_data)[0]

# Receive the data from the client
data = b''
while len(data) < data_length:
    chunk = client_socket.recv(data_length - len(data))
    if not chunk:
        break
    data += chunk


# Deserialize the received data
random_search, space = pickle.loads(data)

# Use the test size and random state as needed in your code
print('Received random_search:', random_search)
print('Received space:', space)
df_train = pd.read_csv("New_train.csv")
df_test = pd.read_csv("New_test.csv")

X = df_train.drop(['Response'],axis=1)
y = df_train['Response']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('Positive cases % in validation set: ', round(100 * len(y_test[y_test == 1]) / len(y_test), 3), '%')
print('Positive cases % in train set: ', round(100 * len(y_train[y_train == 1]) / len(y_train), 3), '%')


clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10, 
                               cv = 4, verbose= 1, random_state= 42, n_jobs = -1)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print (classification_report(y_test, y_pred))

y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('Random Forest ROC curve: CC Fraud')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))


space={ 'max_depth': hp.quniform("max_depth", 3,10,1),
        'gamma': hp.uniform ('gamma', 1,5),
        'reg_alpha' : hp.quniform('reg_alpha', 40,100,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,0.5),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 5, 1),
        'n_estimators': 300,
        'seed': 0
      }

def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    accuracy = accuracy_score(y_test, pred>0.5)
    Roc_Auc_Score = roc_auc_score(y_test, y_score)
    print ("ROC-AUC Score:",Roc_Auc_Score)
    print ("SCORE:", accuracy)
    return {'loss': -Roc_Auc_Score, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 10,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

xgb_model=xgb.XGBClassifier(n_estimators = space['n_estimators'], max_depth = 7, gamma = 1.6267154347279935, reg_lambda = 0.03803056245297226,
                            reg_alpha = 54.0, min_child_weight=5.0,colsample_bytree = 0.5610464094502983 )

xgb_model.fit(X_train,y_train)

y_score = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('XGBoost ROC curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='dashed', color='black')  # Changed plot to use [0, 1] instead of (0, 1)
plt.show()

print('Area under curve (AUC):', auc(fpr, tpr))
auc_value = auc(fpr,tpr)

# Serialize the training history
data = pickle.dumps(auc_value)
client_socket.sendall(data)

# Close the sockets
client_socket.close()
server_socket.close()

