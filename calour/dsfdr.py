from logging import getLogger
import types
import numpy as np
import scipy as sp
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.special import comb

logger = getLogger(__name__)


# data transformation
def rankdata(data):
    logger.debug('ranking the data')
    rdata = np.zeros(np.shape(data))
    for crow in range(np.shape(data)[0]):
        rdata[crow, :] = sp.stats.rankdata(data[crow, :])
    return rdata


def log2data(data):
    logger.debug('log2 transforming the data')
    data[data < 2] = 2
    data = np.log2(data)
    return data


def binarydata(data):
    logger.debug('binary transforming the data')
    data[data != 0] = 1
    return data


def normdata(data):
    logger.debug('normalizing the data')
    data = data / np.sum(data, axis=0)
    return data


# different methods to calculate test statistic
def meandiff(data, labels):
    mean0 = np.mean(data[:, labels == 0], axis=1)
    mean1 = np.mean(data[:, labels == 1], axis=1)
    tstat = mean1 - mean0
    return tstat


def stdmeandiff(data, labels):
    mean0 = np.mean(data[:, labels == 0], axis=1)
    mean1 = np.mean(data[:, labels == 1], axis=1)
    sd0 = np.std(data[:, labels == 0], axis=1, ddof=1)
    sd1 = np.std(data[:, labels == 1], axis=1, ddof=1)
    sdsum = sd0 + sd1
    # if feature has identical values in all samples in each group, std is 0
    # fix it to 1 so won't divide by 0 (mean/std is undefined)
    sdsum[sdsum == 0] = 1
    tstat = (mean1 - mean0) / sdsum
    return tstat


def mannwhitney(data, labels):
    group0 = data[:, labels == 0]
    group1 = data[:, labels == 1]
    tstat = np.array([scipy.stats.mannwhitneyu(group0[i, :], group1[i, :], alternative='two-sided')
                      .statistic for i in range(np.shape(data)[0])])
    return tstat


# kruwallis give a column vector while others give row vector
def kruwallis(data, labels):
    n = len(np.unique(labels))
    allt = np.zeros(np.shape(data)[0])
    for cbact in range(np.shape(data)[0]):
        group = []
        for j in range(n):
            group.append(data[cbact, labels == j])
        tstat = scipy.stats.kruskal(*group).statistic
        allt[cbact] = tstat
    return allt


def pearson(data, labels):
    tstat = np.array([scipy.stats.pearsonr(data[i, :],
                     labels)[0] for i in range(np.shape(data)[0])])
    return tstat


def spearman(data, labels):
    tstat = np.array([scipy.stats.spearmanr(data[i, :],
                     labels).correlation for i in range(np.shape(data)[0])])
    return tstat


# new fdr method
def dsfdr(data, labels, transform_type='rankdata', method='meandiff',
          alpha=0.1, numperm=1000, fdr_method='dsfdr', shuffler=None, random_seed=None):
    '''
    calculate the Discrete FDR for the data

    Parameters
    ----------
    data : N x S numpy array
        each column is a sample (S total), each row a feature (N total)
    labels : a 1d numpy array (length S)
        the labels of each sample (same order as data) with the group
        (0/1 if binary, 0-G-1 if G groups, or numeric values for correlation)
    transform_type : str or None
        transformation to apply to the data before caluculating
        the test statistic
        'rankdata' : rank transfrom each feature
        'log2data' : calculate log2 for each feature using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'binarydata' : convert to binary absence/presence
         None : no transformation to perform
    method : str or function
        the method to use for calculating test statistics:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitney u-test (binary)
        'kruwallis' : kruskal-wallis test (multiple groups)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        'spearman' : spearman correlation (numeric)
        'pearson' : pearson correlation (numeric)
        'nonzerospearman' : spearman correlation only non-zero entries
                            (numeric)
        'nonzeropearson' : pearson correlation only non-zero entries (numeric)
        function : use this function to calculate the test statistic
        (input is data,labels, output is array of float)
    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform
    fdr_method : str
        the FDR procedure to determine significant bacteria
        'dsfdr' : discrete FDR method
        'bhfdr' : Benjamini-Hochberg FDR method
        'byfdr' : Benjamini-Yekutielli FDR method
        'filterBH' : Benjamini-Hochberg FDR method with filtering
    shuffler: function or None, optional
        if None, use shuffling on all samples (using the random_seed supplied)
        if function, use thi supplied function to shuffle to labels for random iteration. Can be used for paired shuffling, etc.
        Input to the function is the labels (np.array), and the random number generator (np.radnom.Generator), output is the shuffled labels (np.array)
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    reject : np array of bool (length N)
        True for features where the null hypothesis is rejected
    tstat : np array of float (length N)
        the test statistic value for each feature (for effect size)
    pvals : np array of float (length N)
        the p-value (uncorrected) for each feature
    qvals: np array of float (length N)
        the q-value (corrected p-value) for each feature.
    '''

    logger.debug('dsfdr using fdr method: %s' % fdr_method)

    # create the numpy.random.Generator
    rng = np.random.default_rng(random_seed)

    # create the shuffler if not supplied
    if shuffler is None:
        shuffler = rng.permutation

    data = data.copy()

    if fdr_method == 'filterBH':
        index = []
        n0 = np.sum(labels == 0)
        n1 = np.sum(labels == 1)

        for i in range(np.shape(data)[0]):
            nonzeros = np.count_nonzero(data[i, :])
            if nonzeros < min(n0, n1):
                pval_min = (comb(n0, nonzeros, exact=True)
                            + comb(n1, nonzeros, exact=True)) / comb(n0 + n1, nonzeros)
                if pval_min <= alpha:
                    index.append(i)
            else:
                index.append(i)
        data = data[index, :]

    # transform the data
    if transform_type == 'rankdata':
        data = rankdata(data)
    elif transform_type == 'log2data':
        data = log2data(data)
    elif transform_type == 'binarydata':
        data = binarydata(data)
    elif transform_type == 'normdata':
        data = normdata(data)
    elif transform_type is None:
        pass
    else:
        raise ValueError('transform type %s not supported' % transform_type)

    numbact = np.shape(data)[0]

    labels = labels.copy()

    numbact = np.shape(data)[0]
    labels = labels.copy()

    logger.debug('start permutation')
    if method == 'meandiff':
        # fast matrix multiplication based calculation
        method = meandiff
        tstat = method(data, labels)
        t = np.abs(tstat)
        numsamples = np.shape(data)[1]
        p = np.zeros([numsamples, numperm])
        k1 = 1 / np.sum(labels == 0)
        k2 = 1 / np.sum(labels == 1)
        for cperm in range(numperm):
            labels = shuffler(labels)
            p[labels == 0, cperm] = k1
        p2 = np.ones(p.shape) * k2
        p2[p > 0] = 0
        mean1 = np.dot(data, p)
        mean2 = np.dot(data, p2)
        u = np.abs(mean1 - mean2)
    elif method == 'mannwhitney' or method == \
                   'kruwallis' or method == 'stdmeandiff':
        if method == 'mannwhitney':
            method = mannwhitney
        if method == 'kruwallis':
            method = kruwallis
        if method == 'stdmeandiff':
            method = stdmeandiff

        tstat = method(data, labels)
        t = np.abs(tstat)
        u = np.zeros([numbact, numperm])
        for cperm in range(numperm):
            rlabels = shuffler(labels)
            rt = method(data, rlabels)
            u[:, cperm] = rt

    elif method == 'spearman' or method == 'pearson':
        # fast matrix multiplication based correlation
        if method == 'spearman':
            data = rankdata(data)
            labels = sp.stats.rankdata(labels)
        meanval = np.mean(data, axis=1).reshape([data.shape[0], 1])
        data = data - np.repeat(meanval, data.shape[1], axis=1)
        labels = labels - np.mean(labels)
        tstat = np.dot(data, labels)
        t = np.abs(tstat)

        # calculate the normalized test statistic
        stdval = np.std(data, axis=1).reshape([data.shape[0], 1])
        # to fix problem with 0 std divide by zero (since we permute it's ok)
        # note we don't remove from mutiple hypothesis - could be done better
        stdval[stdval == 0] = 1
        tdata = data / np.repeat(stdval, data.shape[1], axis=1)
        meanval = np.mean(tdata, axis=1).reshape([tdata.shape[0], 1])
        tdata = tdata - np.repeat(meanval, tdata.shape[1], axis=1)
        meanval = np.mean(data, axis=1).reshape([data.shape[0], 1])
        tdata = tdata - np.repeat(meanval, tdata.shape[1], axis=1)

        tlabels = labels / np.std(labels)
        # fix for n since we multiply without normalizing for n
        tlabels = tlabels / len(tlabels)
        tlabels = tlabels - np.mean(tlabels)
        tstat = np.dot(tdata, tlabels)

        permlabels = np.zeros([len(labels), numperm])
        for cperm in range(numperm):
            rlabels = shuffler(labels)
            permlabels[:, cperm] = rlabels
        u = np.abs(np.dot(data, permlabels))

    elif method == 'nonzerospearman' or method == 'nonzeropearson':
        t = np.zeros([numbact])
        tstat = np.zeros([numbact])
        u = np.zeros([numbact, numperm])
        for i in range(numbact):
            index = np.nonzero(data[i, :])
            label_nonzero = labels[index]
            sample_nonzero = data[i, :][index]
            if len(sample_nonzero) == 0:
                continue
            if method == 'nonzerospearman':
                sample_nonzero = sp.stats.rankdata(sample_nonzero)
                label_nonzero = sp.stats.rankdata(label_nonzero)
            sample_nonzero = sample_nonzero - np.mean(sample_nonzero)
            label_nonzero = label_nonzero - np.mean(label_nonzero)
            tstat[i] = np.dot(sample_nonzero, label_nonzero)
            t[i] = np.abs(tstat[i])
            if np.std(sample_nonzero) == 0:
                continue
            tstat[i] = tstat[i] / (np.std(sample_nonzero) * np.std(label_nonzero) * len(sample_nonzero))
            permlabels = np.zeros([len(label_nonzero), numperm])
            for cperm in range(numperm):
                rlabels = shuffler(label_nonzero)
                permlabels[:, cperm] = rlabels
            u[i, :] = np.abs(np.dot(sample_nonzero, permlabels))

    elif isinstance(method, types.FunctionType):
        # call the user-defined function of statistical test
        t = method(data, labels)
        tstat = t.copy()
        # Get the abs() of the statistic since we are doing a double-sided test for dsFDR
        t = np.abs(tstat)

        u = np.zeros([numbact, numperm])
        for cperm in range(numperm):
            rlabels = shuffler(labels)
            rt = method(data, rlabels)
            u[:, cperm] = rt
        u = np.abs(u)
    else:
        raise ValueError('unsupported method %s' % method)

    # fix floating point errors (important for permutation values!)
    # https://github.com/numpy/numpy/issues/8116
    for crow in range(numbact):
        closepos = np.isclose(t[crow], u[crow, :])
        u[crow, closepos] = t[crow]

    # calculate permutation p-vals
    pvals = np.zeros([numbact])  # p-value for original test statistic t
    qvals = np.ones([numbact])  # q-value (corrected p-value) for each feature.
    pvals_u = np.zeros([numbact, numperm])
    # pseudo p-values for permutated test statistic u
    for crow in range(numbact):
        allstat = np.hstack([t[crow], u[crow, :]])
        stat_rank = sp.stats.rankdata(allstat, method='min')
        allstat = 1 - ((stat_rank - 1) / len(allstat))
        # assign ranks to t from biggest as 1
        pvals[crow] = allstat[0]
        pvals_u[crow, :] = allstat[1:]

    # calculate FDR
    if fdr_method == 'dsfdr':
        # sort unique p-values for original test statistics biggest to smallest
        pvals_unique = np.unique(pvals)
        sortp = pvals_unique[np.argsort(-pvals_unique)]

        # find a data-dependent threshold for the p-value
        foundit = False
        allfdr = []
        allt = []
        for cp in sortp:
            realnum = np.sum(pvals <= cp)
            fdr = (realnum + np.count_nonzero(
                pvals_u <= cp)) / (realnum * (numperm + 1))
            allfdr.append(fdr)
            allt.append(cp)
            if fdr <= alpha:
                if not foundit:
                    realcp = cp
                    foundit = True

        if not foundit:
            # no good threshold was found
            reject = np.repeat([False], numbact)
            return reject, tstat, pvals, qvals

        # fill the reject null hypothesis
        reject = np.zeros(numbact, dtype=int)
        reject = (pvals <= realcp)

        # fill the q-values
        for idx, cfdr in enumerate(allfdr):
            # fix for qval > 1 (since we count on all features in random permutation)
            cfdr = np.min([cfdr, 1])
            cpval = allt[idx]
            qvals[pvals == cpval] = cfdr

    elif fdr_method == 'bhfdr' or fdr_method == 'filterBH':
        t_star = np.array([t, ] * numperm).transpose()
        pvals = (np.sum(u >= t_star, axis=1) + 1) / (numperm + 1)
        reject, qvals, *_ = multipletests(pvals, alpha=alpha, method='fdr_bh')

    elif fdr_method == 'byfdr':
        t_star = np.array([t, ] * numperm).transpose()
        pvals = (np.sum(u >= t_star, axis=1) + 1) / (numperm + 1)
        reject, qvals, *_ = multipletests(pvals, alpha=alpha, method='fdr_by')

    else:
        raise ValueError('fdr method %s not supported' % fdr_method)

    return reject, tstat, pvals, qvals
