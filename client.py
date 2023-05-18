import socket
import pickle
import struct
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Define the server's IP address and port
SERVER_IP = '192.168.80.11'
SERVER_PORT = 9090
SERVER_PORT1 = 1234

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_IP, SERVER_PORT1))
print('Connected to the server 1.')

client_socket1.connect((SERVER_IP, SERVER_PORT))
print('Connected to the server 2.')

# Define the random_search and space variables
random_search = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [4, 6],
    'min_samples_split': [5, 10],
    'n_estimators': [100, 200]
}

space = {
    'max_depth': hp.quniform("max_depth", 3, 10, 1),
    'gamma': hp.uniform('gamma', 1, 5),
    'reg_alpha': hp.quniform('reg_alpha', 40, 100, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 0.5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 5, 1),
    'n_estimators': 300,
    'seed': 0
}

# Serialize the random_search and space variables
data = pickle.dumps((random_search, space))

# Get the data length
data_length = len(data)

# Pack the data length into the header
header = struct.pack('!I', data_length)

# Send the header to the server
client_socket.sendall(header)

# Send the data to the server
client_socket.sendall(data)
variables = {
    'solver': 'liblinear',
    'random_state': 0
}
data = pickle.dumps(variables)

# Prepare the header indicating the length of the data
header_data = struct.pack('!I', len(data))

# # Send the header to the server
client_socket1.sendall(header_data)

# # Send the data to the server
client_socket1.sendall(data)
# Receive the response from the server
response_data = b''
while True:
    chunk = client_socket.recv(4096)
    if not chunk:
        break
    response_data += chunk

# Deserialize the received response
response = pickle.loads(response_data)

# Print the received response
print('Received response 1:', response)

# # Receive the response from the server
response_data = b''
while True:
    chunk = client_socket1.recv(4096)
    if not chunk:
        break
    response_data += chunk

# # Deserialize the received response
response = pickle.loads(response_data)

# # Print the received response
print('Received response 2:', response)

# # Close the sockets
client_socket.close()
client_socket1.close()
