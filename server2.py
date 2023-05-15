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

# Define the server's IP address and port
SERVER_IP = '10.20.46.34'
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
header_size = 4
header_data = b''
while len(header_data) < header_size:
    chunk = client_socket.recv(header_size - len(header_data))
    if not chunk:
        break
    header_data += chunk

# Unpack the header to get the data length
data_length = struct.unpack('!I', header_data)[0]

# Receive the data from the client
data = b''
while len(data) < data_length:
    chunk = client_socket.recv(data_length - len(data))
    if not chunk:
        break
    data += chunk

# Deserialize the received data
test_size, random_state = pickle.loads(data)

# Use the test size and random state as needed in your code
print('Received test size:', test_size)
print('Received random state:', random_state)

df = pd.read_csv("model.csv")
Num_features=df.select_dtypes(include=[np.number]).columns
df[Num_features]=preprocessing.MinMaxScaler().fit_transform(df[Num_features])

# Dividir en datos de entrenamiento y prueba
X = df[Num_features]
y = df['Distance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state= random_state)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear conjunto de datos de TensorFlow
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Definir modelo de red neuronal
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

# Compilar modelo
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse', metrics=['mae'])

# Entrenar modelo
history = model.fit(train_data.shuffle(1000).batch(128), epochs=1, validation_data=test_data.batch(128))

# Serialize the training history
history_data = pickle.dumps(history.history)
#print(history_data)

# Send the training history back to the client
client_socket.sendall(history_data)

# Close the sockets
client_socket.close()
server_socket.close()

