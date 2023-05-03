import socket
import pickle
from sklearn.linear_model import LinearRegression

HOST1 = '10.20.46.60'  # Replace with the IP address of server 1
PORT1 = 9090

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to a specific address and port
    s.bind((HOST1, PORT1))
    # Listen for incoming connections
    s.listen()
    print(f"Server 1 listening on {HOST1}:{PORT1}")
    while True:
        # Wait for a connection
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data = b''
            chunk = conn.recv(1024)
            data += chunk
            print(data)
                
            params = pickle.loads(data)
            X, y = params['X'], params['y']
            # Fit a linear regression model to the data
            model = LinearRegression().fit(X, y)
            response = pickle.dumps(model)
            conn.sendall(response)



