from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split


def qda(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)

        print('Accuracy of QDA classifier on training set: {:.2f}'
              .format(clf.score(X_train, y_train)))
        print('Accuracy of QDA classifier on test set: {:.2f}'
              .format(clf.score(X_test, y_test)))


def qdaAll(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)

        print('Accuracy of QDA classifier on training set: {:.2f}'
              .format(clf.score(X_train, y_train)))
        print('Accuracy of QDA classifier on test set: {:.2f}'
              .format(clf.score(X_test, y_test)))


if __name__ == "__main__":
    # qdaAll('winequality-white.csv')
    qda('winequality-white.csv')
    qda('winequality-red.csv')
