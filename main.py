import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    @staticmethod
    def load_data(train_csv, test_csv):
        test_data = pd.read_csv(test_csv)
        train_data = pd.read_csv(train_csv)
        X_train = train_data.drop(columns=['lables']).values / 255.0
        Y_train = test_data['lables'].values
        X_test = train_data.drop(columns=['lables']).values / 255.0
        Y_test = test_data['lables'].values
        return X_train, Y_train, X_test, Y_test
    
    @staticmethod
    def on_hot_encode(y, num_classes):
        return np.eye(num_classes)[y]
    
    

if __name__ == "__main__":
    pass