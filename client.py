import socket
import pickle
import numpy as np

HOST = '192.168.47.199'  # Replace with the IP address of the server
PORT = 49152

# Generate some sample data for the regression
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Set hyperparameters
hyperparameters = {'X': X, 'y': y}

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Connect to the server
        s.connect((HOST, PORT))
        
        # Send hyperparameters to the server
        data = pickle.dumps(hyperparameters)
        s.sendall(data)
        
        # Receive the trained model from the server
        response = s.recv(1024)
        model = pickle.loads(response)
        
        # Use the trained model for predictions or further processing
        y_pred = model.predict(X)
        print("Predicted values:", y_pred)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        s.close()
