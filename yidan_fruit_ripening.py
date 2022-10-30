import jax.numpy as jnp
import pandas as pd


def F_stats():
	pass


def perm(m, teststats_fn):
	""" randomly permute treatment assignments and compute test statistics 

	"""
	pass


def perm_treat(m):
	"""
	Inputs:
	m (int): monte carlo samples
	"""
	pass






if __name__ == "__main__":
	banana = pd.read_csv("banana.csv")
	data = pd.read_csv("data.csv")

	# combine green and magnitude respectively
	col_names = banana.columns.values
	g_col = pd.Series(col_names).str.contains('_g').tolist()
	mag_col = pd.Series(col_names).str.contains('_mag').tolist()

	# normalise the green with magnitude
	g_banana = banana.loc[:, g_col]
	mag_banana = banana.loc[:, mag_col]
	gnorm_banana = pd.DataFrame(g_banana.values/mag_banana.values, columns=col_names[g_col])
	gnorm_diff_banana = gnorm_banana.diff(periods=1, axis=1)
	
	# combine with banana information and difference of softness
	g_df = pd.concat([data.iloc[:,:2], data.iloc[:,2:4].diff(axis=1).iloc[:,1:], gnorm_diff_banana.iloc[:,1:]], axis =1)
	g_df_melted = pd.melt(g_df, id_vars=["number", "treatment"], var_name="covariates", value_name="value")
	print(g_df_melted)
	g_df_melted.to_csv("g_df_melted.csv")










