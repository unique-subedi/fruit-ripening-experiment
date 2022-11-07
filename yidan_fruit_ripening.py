import jax.numpy as jnp
import jax.random as jrm
from jax import jit, vmap
from jax.lax import fori_loop

import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns


def _median(val, perm_trt, idx_trt, nsize):

	out = jnp.zeros_like(idx_trt, dtype=jnp.float32)
	for i in range(len(idx_trt)):
		idx = jnp.argwhere(perm_trt == idx_trt[i], size=nsize) # 3 bananas for 5 days
		out = out.at[i].set(jnp.median(val.at[idx].get()))
	return out

# if variance is mostly explained by between treatment variance
# then F statistics is large, which corresponds to p-val=P(F>=obs)
def var_median(val, perm_trt, idx_trt, nsize):

	global_med = jnp.median(val)
	local_med = _median(val, perm_trt, idx_trt, nsize)

	num = jnp.sum((local_med - global_med)**2) * nsize / (len(idx_trt)-1)
	den = 0.

	for i in range(len(idx_trt)):
		idx = jnp.argwhere(perm_trt == idx_trt[i], size=nsize)
		sqrt = (val.at[idx].get() - local_med.at[i].get())**2
		den += jnp.sum(sqrt)

	F = num / den * (len(val) - len(idx_trt))
	return F

def _perm(key, trt, idx_trt, val, stats, days=5):
	"""
	key (PRNGKey)
	m (int): number of Monte carlo samples
	trt (jax vector): treatment assignment for subjects
	val (jax vector): containing measurements values, raveled by column (1,2,3,4,1,2,3,4,...)
	stats (str): name of test statistics to use
	"""
	# idx_trt, n_trt = jnp.unique(trt, return_counts=True)
	rep_trt = jnp.repeat(trt[:,jnp.newaxis], repeats=days, axis=1) # col is day (5 in total)
	# regard each day as blocks, permute within block
	perm_trt = jrm.permutation(key, rep_trt, axis=0, independent=False).ravel("F")

	# test for if there is any difference in treatments comparing to each other
	if stats == "var_median":
		perm_stats = var_median(val, perm_trt, idx_trt, 3*days)
	elif stats == "max_median":
		perm_stats = jnp.max(jnp.abs(_median(val, perm_trt, idx_trt, 3*days)))
	elif stats == "mean_median":
		perm_stats = jnp.mean(jnp.abs(_median(val, perm_trt, idx_trt, 3*days)))
	elif stats == "diff_median":
		med =  _median(val, perm_trt, idx_trt, 3*days)
		perm_stats = med[1] - med[0]
	else:
		raise NotImplementedError

	return perm_stats
	# return perm_trt, perm_stats


def perm(key, m, trt, idx_trt, val, stats, days=5):
	""" randomly permute treatment assignments and compute test statistics 
	
	key (PRNGKey)
	m (int): number of Monte carlo samples
	trt (jax array): treatment assignment for subjects (row) on each day (col)
	"""
	keys = jrm.split(key, m)
	ite_perm = jit(vmap(partial(_perm, trt=trt, idx_trt=idx_trt, val=val, stats=stats, days=days), (0)))
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
	# gnorm_banana = pd.DataFrame(g_banana.values/mag_banana.values, columns=col_names[g_col])
	gnorm_banana = g_banana
	# gnorm_banana = mag_banana
	gnorm_diff_banana = gnorm_banana.diff(periods=1, axis=1) # remove trend
	soft_diff = data.iloc[:,2:4].diff(periods=1, axis=1)
	
	# combine with banana information and normalised green
	g_df = pd.concat([data.iloc[:,:2], gnorm_diff_banana.iloc[:,1:]], axis =1)
	soft_df = pd.concat([data.iloc[:,:2], soft_diff.iloc[:,1:]], axis =1)
	g_df_melted = pd.melt(g_df, id_vars=["number", "treatment"], var_name="covariates", value_name="value")

	# # save the entirety of df
	df = pd.concat([data.iloc[:,:2], soft_diff.iloc[:,1:], gnorm_diff_banana.iloc[:,1:]], axis=1)
	df_melted = pd.melt(df, id_vars=["number", "treatment"], var_name="covariates", value_name="value")
	# print(df_melted)
	df_melted.to_csv("mag_df_melted.csv")
	g_df_melted.to_csv("g_df_melted.csv")

	########################## Stats based on Median 5 days ############################
	# g_val = jnp.asarray(g_df_melted["value"].values) # vector 12*5=72
	# obs_trt = g_df["treatment"].values # vector 12
	# idx_trt = jnp.unique(obs_trt).astype(jnp.int32)
	# n_trt = len(idx_trt)
	# print(g_df_melted)
	# print("gval", g_val)
	# m = 10000
	# key = jrm.PRNGKey(130)
	# stats = "var_median"
	# days = 5
	# perm_stats = perm(key, m, obs_trt, idx_trt, g_val, stats, days=days)
	# # print(g_df_melted["treatment"].values)
	# # print(_perm(key, obs_trt, idx_trt, g_val, stats))
	# if stats == "var_median":
	# 	obs_stats = var_median(g_val, g_df_melted["treatment"].values, idx_trt, nsize=3*days)
	# 	# title = "Variance of Between Treatment Medians/Within Treatment Medians"
	# 	title = "Consecutive Change in RGB Magnitude"
	# elif stats == "max_median":
	# 	obs_stats = jnp.max(jnp.abs(_median(g_val, g_df_melted["treatment"].values, idx_trt, nsize=3*days)))
	# 	title = "Max of Absolute Medians"
	# elif stats == "mean_median":
	# 	obs_stats = jnp.mean(jnp.abs(_median(g_val, g_df_melted["treatment"].values, idx_trt, nsize=3*days)))
	# 	title = "Mean of Absolute Medians"
	# else:
	# 	raise NotImplementedError

	# print("obs stats {}".format(stats), obs_stats)

	# pval = jnp.mean(perm_stats >= obs_stats)

	# plt.figure(figsize=(12, 7))
	# sns.histplot(np.asarray(perm_stats), kde=True)
	# plt.axvline(obs_stats, color="r", label="observed")
	# plt.xlabel("Test Stats under Sharp Null (p value = {})".format(pval))
	# plt.title(title)
	# plt.savefig("./figs/5_mag_{}.png".format(stats))

	############################# mean across days #####################################

	# g_val_mean = jnp.asarray(g_df.iloc[:,2:].values).mean(axis=1)
	# obs_trt = g_df["treatment"].values
	# idx_trt = jnp.unique(obs_trt).astype(jnp.int32)
	# n_trt = len(idx_trt)
	# print(obs_trt)
	# print(g_val_mean)

	# m = 10000
	# key = jrm.PRNGKey(130)
	# stats = "var_median"
	# days = 1
	# perm_stats = perm(key, m, obs_trt, idx_trt, g_val_mean, stats, days=days)
	# # print(g_df_melted["treatment"].values)
	# # print(_perm(key, obs_trt, idx_trt, g_val, stats))

	# if stats == "var_median":
	# 	obs_stats = var_median(g_val_mean, g_df_melted["treatment"].values, idx_trt, nsize=3*1)
	# 	title = "Variance of Between Treatment Medians/Within Treatment Medians"
	# elif stats == "max_median":
	# 	obs_stats = jnp.max(jnp.abs(_median(g_val_mean, g_df_melted["treatment"].values, idx_trt, nsize=3*days)))
	# 	title = "Max of Absolute Medians"
	# elif stats == "mean_median":
	# 	obs_stats = jnp.mean(jnp.abs(_median(g_val_mean, g_df_melted["treatment"].values, idx_trt, nsize=3*days)))
	# 	title = "Mean of Absolute Medians"
	# else:
	# 	raise NotImplementedError
	
	# pval = jnp.mean(perm_stats >= obs_stats)

	# plt.figure(figsize=(12, 7))
	# sns.histplot(np.asarray(perm_stats), kde=True)
	# plt.axvline(obs_stats, color="r", label="observed")
	# plt.xlabel("Test Stats under Sharp Null (p value = {})".format(pval))
	# plt.title(title)
	# plt.savefig("./figs/1_mag_{}.png".format(stats))


	############################ difference of median compare to control #######################

	g_val = jnp.asarray(g_df_melted["value"].values) # vector 12*5=60
	obs_trt = jnp.asarray(g_df_melted["treatment"].values) # vector 12*5=60
	m = 10000
	key = jrm.PRNGKey(130)
	stats = "diff_median"
	days = 5

	p_val = []

	# control vs. apple, cucumber, cucumber + apple
	for i in [1,2,3]:
		g_val_2 = g_val[(obs_trt==0) + (obs_trt==i)]
		obs_trt_2 = obs_trt[(obs_trt==0) + (obs_trt==i)]
		idx_trt_2 = jnp.array([0,i])

		perm_stats = perm(key, m, obs_trt_2[:6], idx_trt_2, g_val_2, stats, days=days)
		med = _median(g_val_2, obs_trt_2, idx_trt_2, nsize=3*days)
		obs_stats = med[1] - med[0]

		p_val.append(jnp.mean(obs_stats >= perm_stats))


	# apple vs. cucumber, cucumber + apple
	for i in [0,2,3]:
		g_val_2 = g_val[(obs_trt==1) + (obs_trt==i)]
		obs_trt_2 = obs_trt[(obs_trt==1) + (obs_trt==i)]
		idx_trt_2 = jnp.array([1,i])

		perm_stats = perm(key, m, obs_trt_2[:6], idx_trt_2, g_val_2, stats, days=days)
		med = _median(g_val_2, obs_trt_2, idx_trt_2, nsize=3*days)
		obs_stats = med[1] - med[0]

		p_val.append(jnp.mean(obs_stats >= perm_stats))


	for i in [0,1,3]:
		g_val_2 = g_val[(obs_trt==2) + (obs_trt==i)]
		obs_trt_2 = obs_trt[(obs_trt==2) + (obs_trt==i)]
		idx_trt_2 = jnp.array([2,i])

		perm_stats = perm(key, m, obs_trt_2[:6], idx_trt_2, g_val_2, stats, days=days)
		med = _median(g_val_2, obs_trt_2, idx_trt_2, nsize=3*days)
		obs_stats = med[1] - med[0]

		p_val.append(jnp.mean(obs_stats >= perm_stats))

	for i in [0,1,2]:
		g_val_2 = g_val[(obs_trt==3) + (obs_trt==i)]
		obs_trt_2 = obs_trt[(obs_trt==3) + (obs_trt==i)]
		idx_trt_2 = jnp.array([3,i])

		perm_stats = perm(key, m, obs_trt_2[:6], idx_trt_2, g_val_2, stats, days=days)
		med = _median(g_val_2, obs_trt_2, idx_trt_2, nsize=3*days)
		obs_stats = med[1] - med[0]

		p_val.append(jnp.mean(obs_stats >= perm_stats))


	# # cucumber vs cucumber + apple
	# g_val_2 = g_val[(obs_trt==2) + (obs_trt==3)]
	# obs_trt_2 = obs_trt[(obs_trt==2) + (obs_trt==3)]
	# idx_trt_2 = jnp.array([2,3])

	# perm_stats = perm(key, m, obs_trt_2[:6], idx_trt_2, g_val_2, stats, days=days)
	# med = _median(g_val_2, obs_trt_2, idx_trt_2, nsize=3*days)
	# obs_stats = med[1] - med[0]

	# p_val.append(jnp.mean(obs_stats >= perm_stats))



	print(obs_stats)
	print(perm_stats)
	print(p_val)
	# output
	# [DeviceArray(0.3924, dtype=float32), DeviceArray(0.5022, dtype=float32), DeviceArray(0.4021, dtype=float32), DeviceArray(0.65389997, dtype=float32), DeviceArray(0.3425, dtype=float32), DeviceArray(1., dtype=float32)]


	
	# perm_stats = perm(key, m, obs_trt, idx_trt, g_val_mean, stats, days=days)
	# obs_stats = var_median(g_val, g_df_melted["treatment"].values, idx_trt, nsize=3*days)
	# print(g_df_melted["treatment"].values)
	# print(_perm(key, obs_trt, idx_trt, g_val, stats))











