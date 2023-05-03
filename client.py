import socket
import pickle
import numpy as np

HOST1 = '10.20.46.60'  # Replace with the IP address of server 1
PORT1 = 9090

# Generate some sample data for the regression
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

params = {'X': X, 'y': y}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect((HOST1, PORT1))
        data = pickle.dumps(params)
        s.sendall(data)
        response = b''
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            response += chunk
        model = pickle.loads(response)
        print(model.coef_)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        s.close()
