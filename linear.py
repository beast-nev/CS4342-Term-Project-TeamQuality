import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import svm
from sklearn.base import BaseEstimator, RegressorMixin


def multiple_linear_predictors(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()
        yhat = model.predict(X_test)
        print("Confusion matrix: ", confusion_matrix(y_test, list(map(round, yhat))))
        print("Accuracy score: ", accuracy_score(y_test, list(map(round, yhat))))
        sm.graphics.influence_plot(model)
        print(model.summary())
        plt.show()


def multiple_smallp_linear(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        # 5 fold cross validation
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print("Scores from cross validation: \n", scores)
        print("Average cross validation score: ", scores.mean())

        lin_reg = sm.OLS(y_train, X_train).fit()
        print(lin_reg.summary())


def multiple_smallp_logistic(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
              .format(logreg.score(X_train, y_train)))
        print('Accuracy of Logistic regression classifier on test set: {:.2f}'
              .format(logreg.score(X_test, y_test)))
        # print(log_reg.predict(X_test))


def smallp_linear_winteractive(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        # interactive predictors with high correlation
        df['cit_fix'] = df['citric acid'] * df['fixed acidity']
        df['tot_free'] = df['total sulfur dioxide'] * df['free sulfur dioxide']
        df['den_fix'] = df['density'] * df['fixed acidity']

        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol", "cit_fix", "tot_free", "den_fix"]]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        # # 5 fold cross validation
        # scores = cross_val_score(clf, X_train, y_train, cv=3)
        # print("Scores from cross validation: \n", scores)
        # print("Average cross validation score: ", scores.mean())

        lin_reg = sm.OLS(y_train, X_train).fit()
        print(lin_reg.summary())


def polynomial_features(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")

        y = df['quality']
        X = df[["volatile acidity", "sulphates", "alcohol"]]
        X = sm.add_constant(X)

        # poly = PolynomialFeatures(2)
        poly = PolynomialFeatures(2)
        X = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        lin_reg = sm.OLS(y_train, X_train)
        print(lin_reg.fit().summary())
        scores = cross_val_score(SMWrapper(sm.OLS), X_train, y_train, cv=3)
        print("Scores from cross validation: \n", scores)
        print("Average cross validation score: ", scores.mean())


# from : https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation
# -is-it-possible
class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


def single_linear_predictors(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        corr = df.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.title("Correlation Matrix of Red Wine Predictors")
        plt.show()

        # iterate through each predictor
        for index, col in enumerate(df.columns):
            if col == 'quality': break

            col_vals = df[col]

            # print basic predictor data
            print(col.ljust(20) + ' range = ' + "{:.2f}".format(max(col_vals) - min(col_vals)).ljust(8) +
                  ' mean = ' + "{:.2f}".format(np.mean(col_vals)).ljust(8) + ' std dev = ' +
                  "{:.2f}".format(np.std(col_vals)))

            # make training data
            x_train = df[col]
            x_train = sm.add_constant(x_train)
            y_train = df['quality']

            # NOTE: probably should do cross validation here and assess accuracy
            clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
            # 5 fold cross validation
            scores = cross_val_score(clf, x_train, y_train, cv=3)
            print("Scores from cross validation: \n", scores)
            print("Average cross validation score: ", scores.mean())


            # fit model
            model = sm.OLS(y_train, x_train).fit()
            print(model.summary())
            # todo: replace predict on test set
            yhat = model.predict(x_train)
            print("Confusion matrix: ", confusion_matrix(y_train, list(map(round, yhat))))
            print("Accuracy score: ", accuracy_score(y_train, list(map(round, yhat))))
            # sm.graphics.influence_plot(model)

            # setup regression line info
            p = model.params
            reg_line_values = np.arange(min(x_train[col]), max(x_train[col]))

            # plot data as scatter plot
            ax = df.plot(x=col, y='quality', kind='scatter')
            # plot regression line using params
            ax.plot(reg_line_values, p[0] + p[1] * reg_line_values, color='r')
            # plt.show()


if __name__ == "__main__":
    multiple_linear_predictors('winequality-red.csv')
    # multiple_linear_predictors('winequality-white.csv')
    # multiple_smallp_predictors('winequality-red.csv')
    # single_linear_predictors('winequality-red.csv')
    # single_linear_predictors('winequality-white.csv')
    multiple_smallp_logistic('winequality-red.csv')
    multiple_smallp_logistic('winequality-white.csv')
    # multiple_smallp_linear('winequality-red.csv')
    # smallp_linear_winteractive('winequality-red.csv')
    polynomial_features('winequality-red.csv')
    polynomial_features('winequality-white.csv')
    single_linear_predictors('winequality-red.csv')
    # single_linear_predictors('winequality-white.csv')
