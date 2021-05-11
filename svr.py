import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        regr.fit(X_train, y_train)
        print('Accuracy of SVR on training set: {:.2f}'
              .format(regr.score(X_train, y_train)))
        print('Accuracy of SVR on test set: {:.2f}'
              .format(regr.score(X_test, y_test)))


if __name__ == "__main__":
    svr('winequality-red.csv')
    svr('winequality-white.csv')
