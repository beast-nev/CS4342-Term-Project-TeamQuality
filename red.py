import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'winequality-red.csv'  # 'winequality-white.csv'

def single_predictors():
	with open(file_name) as f:
		df = pd.read_csv(f, sep=";")

		corr = df.corr()
		ax = sns.heatmap(
			corr,
			vmin=-1, vmax=1, center=0,
			cmap=sns.diverging_palette(20, 220, n=200),
			square=True
		)
		plt.show()

	for index, col in enumerate(df.columns):
		if col == 'quality': break

		col_vals = df[col]

		print(col.ljust(20) + ' range = ' + "{:.2f}".format(max(col_vals) - min(col_vals)).ljust(8) +
			  ' mean = ' + "{:.2f}".format(np.mean(col_vals)).ljust(8) + ' std dev = ' +
			  "{:.2f}".format(np.std(col_vals)))

		# print(col + ': ' + str(max(col_vals) - min(col_vals)))

		plt.scatter(col_vals, df['quality'])
		plt.xlabel(col)
		plt.ylabel("quality")
		plt.show()


if __name__ == "__main__":
	single_predictors()
