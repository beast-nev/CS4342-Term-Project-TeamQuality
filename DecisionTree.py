import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import tree
from itertools import combinations
from sklearn.model_selection import cross_val_score

# https://nbviewer.jupyter.org/github/pedvide/ISLR_Python/blob/master/Chapter6_Linear_Model_Selection_and_Regularization.ipynb#6.5.1-Best-Subset-Selection
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
	X = df[
		["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
		 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
	y = df['quality']

	best_depth, best_tree, best_score, best_subset = 0, None, 0, None

	for depth in range(2, 7):
		# build tree with iterative depth
		tree_reg = tree.DecisionTreeRegressor(max_depth=depth)

		# find best combo of predictors using best subset
		sub, score, _, _ = best_subset_finder(tree_reg, X, y, max_size=11, cv=5)

		if score > best_score:
			best_depth, best_tree, best_score, best_subset = depth, tree_reg, score, sub

	# print data
	predictors = df.iloc[:, list(best_subset)]
	print("The best depth was " + str(best_depth))
	print("The best subset of predictors for the decision tree were " + str([col for index, col in enumerate(predictors)]))
	print("The accuracy score was " + str(best_score))

	# fit regression with best predictors
	tree_reg.fit(predictors, y)

	# generate decision tree line for graph
	start = [predictors.min().iloc[index] for index, col in enumerate(predictors)]
	stop = [predictors.max().iloc[index] for index, col in enumerate(predictors)]
	X_tree = np.linspace(stop, start, 200)
	y_tree = tree_reg.predict(X_tree)

	# graph each predictor vs quality and the predicted decision
	for index, col in enumerate(predictors):
		plt.figure()
		plt.scatter(predictors.iloc[:, index], y, s=20, edgecolor="black", c="darkorange", label="data")
		plt.plot(X_tree[:, index], y_tree, color="cornflowerblue", label="max_depth=2", linewidth=2)
		plt.xlabel(col)
		plt.ylabel("quality")
		plt.title("Decision Tree Regression")
		plt.legend()
		plt.show()


if __name__ == "__main__":
	# decision_tree('winequality-white.csv')
	decision_tree('winequality-red.csv')
