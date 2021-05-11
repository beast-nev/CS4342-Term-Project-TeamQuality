import pandas as pd
import numpy as np
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
        scores = []
        epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(0, 10):
            regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=epsilons[i]))
            regr.fit(X_train, y_train)
            # print('Accuracy of SVR on training set: %0.8f with epsilon: %0.2f'
            #       % (regr.score(X_train, y_train), i))
            print('Accuracy of SVR on test set: %0.8f with epsilon: %0.1f'
                  % (regr.score(X_test, y_test), epsilons[i]))
            scores.append(regr.score(X_test, y_test))

        print(np.max(scores))


if __name__ == "__main__":
    svr('winequality-red.csv')
    svr('winequality-white.csv')
