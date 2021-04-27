import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import tree
from itertools import combinations
from sklearn.model_selection import cross_val_score


def best_subset_finder(estimator, X, y, max_size=8, cv=5):
	n_features = X.shape[1]
	subsets = (combinations(range(n_features), k + 1) for k in range(min(n_features, max_size)))

	best_size_subset = []
	for subsets_k in subsets:  # for each list of subsets of the same size
		best_score = -np.inf
		best_subset = None
		for subset in subsets_k:  # for each subset
			estimator.fit(X.iloc[:, list(subset)], y)
			# get the subset with the best score among subsets of the same size
			score = estimator.score(X.iloc[:, list(subset)], y)
			if score > best_score:
				best_score, best_subset = score, subset
		# to compare subsets of different sizes we must use CV
		# first store the best subset of each size
		best_size_subset.append(best_subset)

	# compare best subsets of each size
	best_score = -np.inf
	best_subset = None
	list_scores = []
	for subset in best_size_subset:
		score = cross_val_score(estimator, X.iloc[:, list(subset)], y, cv=cv).mean()
		list_scores.append(score)
		if score > best_score:
			best_score, best_subset = score, subset

	return best_subset, best_score, best_size_subset, list_scores


def decision_tree(f):
	df = pd.read_csv(f, sep=";")
	# these predictors chosen based off
	X = df[
		["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
		 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
	y = df['quality']

	best_depth, best_tree, best_score, best_subset = 0, None, 0, None

	depth = 4
	tree_reg = tree.DecisionTreeRegressor(max_depth=depth)

	sub, score, _, _ = best_subset_finder(tree_reg, X, y, max_size=11, cv=5)

	if score > best_score:
		best_depth, best_tree, best_score, best_subset = depth, tree_reg, score, sub

	print("The best depth was " + str(best_depth))
	print("The best subset of predictors for the decision tree was " + str(best_subset))
	print("The accuracy score was " + str(best_score))
	predictors = df.iloc[:, list(best_subset)]

	tree_reg.fit(predictors, y)

	start = [predictors.min().iloc[index] for index, col in enumerate(predictors)]
	stop = [predictors.max().iloc[index] for index, col in enumerate(predictors)]
	X_test = np.linspace(stop, start, 200)
	y_test = tree_reg.predict(X_test)

	for index, col in enumerate(predictors):
		plt.figure()
		plt.scatter(predictors.iloc[:, index], y, s=20, edgecolor="black", c="darkorange", label="data")
		plt.plot(X_test[:, index], y_test, color="cornflowerblue", label="max_depth=2", linewidth=2)
		plt.xlabel(col)
		plt.ylabel("quality")
		plt.title("Decision Tree Regression")
		plt.legend()
		plt.show()


if __name__ == "__main__":
	decision_tree('winequality-red.csv')
