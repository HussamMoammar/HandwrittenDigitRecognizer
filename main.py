
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
        train_data = pd.read_csv(train_csv)
        test_data = pd.read_csv(test_csv)
        X_train = train_data.drop(columns=['label']).values / 255.0
        y_train = train_data['label'].values
        X_test = test_data.drop(columns=['label']).values / 255.0
        y_test = test_data['label'].values
        return X_train, y_train, X_test, y_test

    @staticmethod
    def one_hot_encode(y, num_classes):
        return np.eye(num_classes)[y]

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                acc = self.accuracy(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


if __name__ == "__main__":
    input_size = 784 
    hidden_size = 64
    output_size = 10
    learning_rate = 0.1

    
    X_train, y_train, X_test, y_test = NeuralNetwork.load_data('mnist_train.csv', 'mnist_test.csv')
    y_train_encoded = NeuralNetwork.one_hot_encode(y_train, output_size)
    y_test_encoded = NeuralNetwork.one_hot_encode(y_test, output_size)

    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    nn.train(X_train, y_train_encoded, epochs=1000)

    
    y_test_pred = nn.forward(X_test)
    test_acc = NeuralNetwork.accuracy(y_test_encoded, y_test_pred)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
