import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'
              .format(knn.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'
              .format(knn.score(X_test, y_test)))


if __name__ == "__main__":
    knn('winequality-white.csv')
