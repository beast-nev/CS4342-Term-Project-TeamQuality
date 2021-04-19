import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def red_single_var():
	with open("winequality-red.csv") as f:
		df = pd.read_csv(f, sep=";")
		print(df)

		corr = df.corr()
		ax = sns.heatmap(
			corr,
			vmin=-1, vmax=1, center=0,
			cmap=sns.diverging_palette(20, 220, n=200),
			square=True
		)
		plt.show()

if __name__ == "__main__":
	red_single_var()