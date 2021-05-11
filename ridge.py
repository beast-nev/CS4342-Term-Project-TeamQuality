import pandas as p
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


def ridge(file_name):
    with open(file_name) as f:
        df = p.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = linear_model.Ridge()

        alphas = np.logspace(-6, 6, 200)
        coefs = []
        scores = []
        mse = []
        for a in alphas:
            clf.set_params(alpha=a)
            clf.fit(X_train, y_train)
            yp = clf.predict(X_test)
            coefs.append(clf.coef_)
            mse.append(mean_squared_error(yp, y_test))
            scores.append(clf.score(X_test, y_test))
            print("%0.2f score with %0.2f alpha" % (clf.score(X_test, y_test), a))
        # clf.fit(X_train, y_train)
        plt.figure(figsize=(20, 6))
        print("Best score on test set: %0.2f" % np.max(scores))

        plt.subplot(121)
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(alphas, mse)
        ax.set_xscale('log')
        plt.xlabel('alpha')
        plt.ylabel('mse')
        plt.title('Coefficient error as a function of the regularization')
        plt.axis('tight')

        plt.show()


if __name__ == "__main__":
    ridge('winequality-white.csv')
    ridge('winequality-red.csv')
