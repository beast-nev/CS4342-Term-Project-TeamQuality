import pandas as p
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def lasso(file_name):
    with open(file_name) as f:
        df = p.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf = linear_model.Lasso(alpha=0.1)
        clf.fit(X_train, y_train)
        print("Model coefficients:", clf.coef_)
        print("Model score on test set: %0.2f" % clf.score(X_test, y_test))


if __name__ == "__main__":
    lasso('winequality-white.csv')
    lasso('winequality-red.csv')
