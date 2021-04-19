import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import svm


def multiple_linear_predictors(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")
        y = df['quality']
        x = df[
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
        sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        print(model.summary())


def single_linear_predictors(file_name):
    with open(file_name) as f:
        df = pd.read_csv(f, sep=";")

        # plot correlation matrix
        corr = df.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
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
            scores = cross_val_score(clf, x_train, y_train, cv=5)
            print("Scores from cross validation: \n", scores)
            print("Average cross validation score: ", scores.mean())

            # fit model
            model = sm.OLS(y_train, x_train).fit()
            print(model.summary())

            # setup regression line info
            p = model.params
            reg_line_values = np.arange(min(x_train[col]), max(x_train[col]))

            # plot data as scatter plot
            ax = df.plot(x=col, y='quality', kind='scatter')
            # plot regression line using params
            ax.plot(reg_line_values, p[0] + p[1] * reg_line_values, color='r')
            plt.show()


if __name__ == "__main__":
    multiple_linear_predictors('winequality-red.csv')
    # single_linear_predictors('winequality-red.csv')
    # single_linear_predictors('winequality-white.csv')
