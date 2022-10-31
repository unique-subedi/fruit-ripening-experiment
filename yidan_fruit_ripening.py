import jax.numpy as jnp
import jax.random as jrm
from jax import jit, vmap
from jax.lax import fori_loop

import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns


def _median(val, perm_trt, idx_trt):

	out = jnp.zeros_like(idx_trt, dtype=jnp.float32)
	for i in range(len(idx_trt)):
		idx = jnp.argwhere(perm_trt == i, size=3)
		out = out.at[i].set(jnp.median(val.at[idx].get()))
	return out

# if variance is mostly explained by between treatment variance
# then F statistics is large, which corresponds to p-val=P(F>=obs)
def var_median(val, perm_trt, idx_trt):

	global_med = jnp.median(val)
	local_med = _median(val, perm_trt, idx_trt)

	num = jnp.sum((local_med - global_med)**2) * 3 / (len(idx_trt)-1)
	den = 0.

	for i in range(len(idx_trt)):
		idx = jnp.argwhere(perm_trt == i, size=3)
		sqrt = (val.at[idx].get() - local_med.at[i].get())**2
		den += jnp.sum(sqrt)

	F = num / den * (len(val) - len(idx_trt))
	return F

def _perm(key, trt, idx_trt, val, stats):
	"""
	key (PRNGKey)
	m (int): number of Monte carlo samples
	trt (jax vector): treatment assignment for subjects
	val (jax vector): containing measurements values, raveled by column (1,2,3,4,1,2,3,4,...)
	stats (str): name of test statistics to use
	"""
	# idx_trt, n_trt = jnp.unique(trt, return_counts=True)
	rep_trt = jnp.repeat(trt[:,jnp.newaxis], repeats=len(idx_trt), axis=1) # col is day
	# regard each day as blocks, permute within block
	perm_trt = jrm.permutation(key, rep_trt, axis=0, independent=True).ravel("F")

	# test for if there is any difference in treatments comparing to each other
	if stats == "var_median":
		perm_stats = var_median(val, perm_trt, idx_trt)
	else:
		raise NotImplementedError

	return perm_stats


def perm(key, m, trt, idx_trt, val, stats):
	""" randomly permute treatment assignments and compute test statistics 
	
	key (PRNGKey)
	m (int): number of Monte carlo samples
	trt (jax array): treatment assignment for subjects (row) on each day (col)
	"""
	keys = jrm.split(key, m)
	ite_perm = jit(vmap(partial(_perm, trt=trt, idx_trt=idx_trt, val=val, stats=stats), (0)))
	stats_perm = ite_perm(keys)

	return stats_perm


if __name__ == "__main__":

	""" We use normalised (divide by magnitude) green RGB value for each of the data point, then we take 1 lag difference
	to remove linear trend in time.
	Then we test for if there is a difference in any treatment comparing to each other with a modified F-statistics 
	based on between group / withiin group variance of median.
	We use a permutation test, where we treat each day as a block, and permute within each of the block, after which we 
	compute the test statistics using all the data.
	Alternatively, we can compute an average over the 5 days for each of the treatment, however, as seen in HW3, we may
	fail to control type 1 error.
	After finding the test statistics distrbution under the null hypothesis, we can find the p-value P(F >= obs).
	"""

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
	gnorm_diff_banana = gnorm_banana.diff(periods=1, axis=1) # remove trend
	soft_diff = data.iloc[:,2:4].diff(periods=1, axis=1)
	
	# combine with banana information and normalised green
	g_df = pd.concat([data.iloc[:,:2], gnorm_diff_banana.iloc[:,1:]], axis =1)
	soft_df = pd.concat([data.iloc[:,:2], soft_diff.iloc[:,1:]], axis =1)
	g_df_melted = pd.melt(g_df, id_vars=["number", "treatment"], var_name="covariates", value_name="value")

	# # save the entirety of df
	# df = pd.concat([data.iloc[:,:2], soft_diff.iloc[:,1:], gnorm_diff_banana.iloc[:,1:]], axis=1)
	# df_melted = pd.melt(df, id_vars=["number", "treatment"], var_name="covariates", value_name="value")
	# # print(df_melted)
	# df_melted.to_csv("df_melted.csv")

	############################### Stats based on Median ###############################
	g_val = jnp.asarray(g_df_melted["value"].values) # vector 12*5=72
	obs_trt = g_df["treatment"].values # vector 12
	idx_trt = jnp.unique(obs_trt).astype(jnp.int32)
	n_trt = len(idx_trt)

	m = 200
	key = jrm.PRNGKey(130)
	stats = "var_median"
	perm_stats = perm(key, m, obs_trt, idx_trt, g_val, stats)

	# obs_median = _median(g_val, g_df_melted["treatment"].values, idx_trt)
	obs_stats = var_median(g_val, g_df_melted["treatment"].values, idx_trt)
	pval = jnp.mean(perm_stats >= obs_stats)

	plt.figure(figsize=(12, 7))
	sns.histplot(np.asarray(perm_stats), kde=True)
	plt.axvline(obs_stats, color="r", label="observed")
	plt.xlabel("Test Stats under Sharp Null (p value = {})".format(pval))
	plt.title("Variance of Between Treatment Medians/Within Treatment Medians")
	plt.savefig("./figs/Ftest_5.png")










