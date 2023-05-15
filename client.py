import socket
import pickle
import struct

# Define the server's IP address and port
SERVER_IP = '10.20.46.34'
SERVER_PORT = 9090
SERVER_PORT1 = 1234

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_IP, SERVER_PORT))
print('Connected to the server 1.')

client_socket1.connect((SERVER_IP, SERVER_PORT1))
print('Connected to the server 2.')

# Define the test size and random state
test_size = 0.2
random_state = 42

# Serialize the test size and random state
data = pickle.dumps((test_size, random_state))

# Get the data length
data_length = len(data)

# Pack the data length into the header
header = struct.pack('!I', data_length)

# Send the header to the server
client_socket.sendall(header)

# Send the data to the server
client_socket.sendall(data)

test_size = 0.3
random_state = 40

# Serialize the test size and random state
data = pickle.dumps((test_size, random_state))

# Get the data length
data_length = len(data)

# Pack the data length into the header
header = struct.pack('!I', data_length)

# Send the header to the server
client_socket1.sendall(header)

# Send the data to the server
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

response_data = b''
while True:
    chunk = client_socket1.recv(4096)
    if not chunk:
        break
    response_data += chunk

# Deserialize the received response
response = pickle.loads(response_data)

# Print the received response
print('Received response 2:', response)

# Close the socket
client_socket.close()
