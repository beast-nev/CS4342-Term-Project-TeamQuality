import pandas as pd
import statsmodels.api as sm
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


if __name__ == "__main__":
    find_knn('winequality-red.csv')
    # find_knn('winequality-white.csv')
