import socket
import pickle
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holoviews.plotting.mpl
import seaborn as sns
import holoviews as hv
from holoviews.plotting import mpl
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
SERVER_PORT = 9090

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
    chunk = client_socket.recv(4096)
    data += chunk

# Access the variables
variables = pickle.loads(data)

# Access the variables
solver = variables['solver']
random_state = int(variables['random_state'])



# Use the test size and random state as needed in your code
print('Received solver:', solver)
print('Received space:', random_state)
# Deserialize the received data

df_train = pd.read_csv("New_train.csv")
df_test = pd.read_csv("New_test.csv")

X = df_train.drop(['Response'],axis=1)
y = df_train['Response']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('Positive cases % in validation set: ', round(100 * len(y_test[y_test == 1]) / len(y_test), 3), '%')
print('Positive cases % in train set: ', round(100 * len(y_train[y_train == 1]) / len(y_train), 3), '%')

GLM = LogisticRegression(solver=solver, random_state=random_state)
GLM_fit = GLM.fit(X_train, y_train)
GLM_probability = pd.DataFrame(GLM_fit.predict_proba(X_test))
GLM_probability.columns = GLM_probability.columns.astype(str)

GLM_probability.mean()

print("We expect: " + format(round(float(GLM_probability.iloc[:, 1].mean() * X_test.shape[0]))) + " 1's.")


plt.plot(GLM_probability.iloc[:, 1])
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Probability of retention')
plt.show()
fpr, tpr, _ = roc_curve(y_test, GLM_fit.predict_proba(X_test)[:,1])

plt.title('Logistic regression ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ' ,format(round(auc(fpr,tpr),5)))
# Serialize the training history
auc_value = auc(fpr,tpr)

# Serialize the training history
data = pickle.dumps(auc_value)
client_socket.sendall(data)

# Close the sockets
client_socket.close()
server_socket.close()
