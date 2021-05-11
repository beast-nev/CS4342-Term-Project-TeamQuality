import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def find_knn(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]
        X = sm.add_constant(X)
        best_score, best_model, best_k = 0, None, 0
        for k in range(1, 101):
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X, y).mean()
            if score > best_score:
                best_score, best_model, best_k = score, knn, k

        print("Best KNN k={}, cross validation score: {:.4f}".format(best_k, best_score))

        start = [X.min().iloc[index] for index, col in enumerate(X)]
        stop = [X.max().iloc[index] for index, col in enumerate(X)]
        X_tree = np.linspace(stop, start, 200)
        best_model.fit(X, y)
        y_tree = best_model.predict(X_tree)

        # graph each predictor vs quality and the predicted decision
        for index, col in enumerate(X):
            plt.figure()
            plt.scatter(X.iloc[:, index], y, s=20, edgecolor="black", c="darkorange", label="data")
            plt.plot(X_tree[:, index], y_tree, color="cornflowerblue", label="K={}".format(best_k),
                     linewidth=2)
            plt.xlabel(col)
            plt.ylabel("Quality")
            plt.title("K-Nearest Neighbors")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    find_knn('winequality-red.csv')
    find_knn('winequality-white.csv')

