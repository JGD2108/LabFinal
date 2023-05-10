import socket
import pickle
from sklearn.linear_model import LinearRegression

HOST = '192.168.47.199'  # Replace with the IP address of the server
PORT = 49152

def handle_client(conn):
    # Receive hyperparameters from the client
    data = conn.recv(1024)
    hyperparameters = pickle.loads(data)
    
    # Extract hyperparameters
    X = hyperparameters['X']
    y = hyperparameters['y']
    
    # Fit a linear regression model to the data
    model = LinearRegression().fit(X, y)
    
    # Send the trained model back to the client
    response = pickle.dumps(model)
    conn.sendall(response)
    
    # Close the connection
    conn.close()

def start_server():
    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to a specific address and port
        s.bind((HOST, PORT))
        # Listen for incoming connections
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            # Wait for a connection
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            
            # Handle the client request in a separate thread or process
            handle_client(conn)

start_server()

