import numpy as np

from scipy.stats._hypotests import _get_wilcoxon_distr
from scipy.stats._mannwhitneyu import _mwu_state

def signrankpmf(k,n):
    "Probability mass function for null distribution of the Wilcoxon signed-rank statistic"
    count_k = _get_wilcoxon_distr(n)
    tot = np.sum(count_k)
    if np.isscalar(k):
        k_k = np.arange(len(count_k))
        return count_k[k_k==k][0]/tot
    else:
        return count_k[k]/tot

def signrankcdf(k,n):
    "Cumulative distribution function for null distribution of the Wilcoxon signed-rank statistic"
    count_k = _get_wilcoxon_distr(n)
    tot = np.sum(count_k)
    if np.isscalar(k):
        k_k = np.arange(len(count_k))
        return np.sum(count_k[k_k<=k])/tot
    else:
        cume_k = np.cumsum(count_k)
        return cume_k[k]/tot

def signrankppf(p,n):
    "Percentile of null distribution of the Wilcoxon signed-rank statistic"
    # A little trickery to get the correct answer for p=0
    count_k = np.concatenate(([0],_get_wilcoxon_distr(n)))
    tot = np.sum(count_k)
    cume_k = np.cumsum(count_k)
    k_k = np.arange(-1,len(count_k)-1)
    return k_k[cume_k >= tot * p][0]

def mannwhitneypmf(u,m,n):
    "Probability mass function for null distribution of the Mann-Whitney U statistic"
    return _mwu_state.pmf(u,m,n)

def mannwhitneycdf(u,m,n):
    "Cumulative distribution function for null distribution of the Mann-Whitney U statistic"
    return _mwu_state.cdf(u,m,n)

def mannwhitneyppf(p,m,n):
    "Percentile of null distribution of the Mann-Whitney U statistic"
    if p <= 0:
        return -1
    else:
        u_u = np.arange(m*n+1)
        cdf_u = mannwhitneycdf(u_u,m,n)
        return u_u[cdf_u>=p][0]
