import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None

    # def fit(self, X, y, lr=0.01, n_iters=1000):
    #     n_samples, n_features = X.shape
    #     self.weights = np.zeros(n_features)
    #     self.bias = 0

    #     for _ in range(n_iters):
    #         y_predicted = np.dot(X, self.weights) + self.bias

    #         # Gradient tính theo đạo hàm MSE
    #         dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
    #         db = (1 / n_samples) * np.sum(y_predicted - y)

    #         # Cập nhật weights và bias
    #         self.weights -= lr * dw
    #         self.bias -= lr * db


    def fit(self, X, y, lr=0.01, n_iters=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Kiểm tra xem y_predicted có NaN không
            if np.isnan(y_predicted).any():
                print("Lỗi: NaN trong y_predicted")
                break

            # Gradient tính theo đạo hàm MSE
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Cập nhật weights và bias
            self.weights -= lr * dw
            self.bias -= lr * db


    # def predict(self, X):
    #     return np.dot(X, self.weights) + self.bias
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
    
        # Kiểm tra NaN trong y_pred
        if np.isnan(y_pred).any():
            print("Lỗi: NaN trong y_pred")
    
        return y_pred
