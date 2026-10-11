import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, C=0.01, nb_iterations = 1000):
        self.lr = learning_rate
        self.C = C
        self.nb_iterations = nb_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        nb_samples, nb_features = X.shape

        rng = np.random.default_rng()
        self.w = rng.random(nb_features)
        self.b = rng.uniform(-1,1)
        # comment je modifie b???
        for _ in range(self.nb_iterations):
            for i, x_i in enumerate(X):
                condition = y[i] * (np.dot(x_i, self.w) - self.b) <= 1
                self.w = (1-self.lr) * self.w
                if condition:
                    self.w += self.lr * self.C * y[i] *x_i
                
    def predict(self, X):
        pass