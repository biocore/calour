from logging import getLogger

import numpy as np
import scipy as sp
import scipy.stats
import types


logger = getLogger(__name__)


# data transformation
def rankdata(data):
    logger.debug('rank transforming')
    rdata = np.zeros(np.shape(data))
    for crow in range(np.shape(data)[0]):
        rdata[crow, :] = sp.stats.rankdata(data[crow, :])
    return rdata


def logdata(data):
    logger.debug('log transforming')
    data[data < 2] = 2
    data = np.log2(data)
    return data


def apdata(data):
    logger.debug('binary transforming')
    data[data != 0] = 1
    return data


def normdata(data):
    logger.debug('normalizing')
    data = data / np.sum(data, axis=0)
    return data


# different methods to calculate test statistic
def meandiff(data, labels, useabs=True):
    mean0 = np.mean(data[:, labels == 0], axis=1)
    mean1 = np.mean(data[:, labels == 1], axis=1)
    tstat = mean1 - mean0
    if useabs:
        tstat = np.abs(tstat)
    return tstat


def mannwhitney(data, labels):
    group0 = data[:, labels == 0]
    group1 = data[:, labels == 1]
    tstat = np.array([scipy.stats.mannwhitneyu(group0[i, :], group1[i, :]).statistic for i in range(np.shape(data)[0])])
    return tstat


# kruwallis give a column vector while others give row vector
def kruwallis(data, labels):
    n = len(np.unique(labels))
    allt = []
    for cbact in range(np.shape(data)[0]):
        group = []
        for j in range(n):
            group.append(data[cbact, labels == j])
        tstat = scipy.stats.kruskal(*group).statistic
        allt.append(tstat)
    return allt


def stdmeandiff(data, labels, useabs=True):
    mean0 = np.mean(data[:, labels == 0], axis=1)
    mean1 = np.mean(data[:, labels == 1], axis=1)
    sd0 = np.std(data[:, labels == 0], axis=1, ddof=1)
    sd1 = np.std(data[:, labels == 1], axis=1, ddof=1)
    tstat = (mean1 - mean0)/(sd1 + sd0)
    if useabs:
        tstat = np.abs(tstat)
    return tstat


# new fdr method
def pbfdr(data, labels, method='meandiff', transform='rankdata', alpha=0.1, numperm=1000, fdrmethod='pbfdr'):
    '''
    calculate the permutation FDR for the data

    input:
    data : N x S numpy array
        each column is a sample (S total), each row an OTU (N total)
    labels : a 1d numpy array (length S)
        the labels of each sample (same order as data) with the group (0/1 if binary, 0-G-1 if G groups, or numeric values for correlation)
    method : str or function
        the method to use for the t-statistic test. options:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitneu u-test (binary)
        'kruwallis' : kruskal-wallis test (multiple groups)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        'spearman' : spearman correlation (numeric)
        'pearson' : pearson correlation (numeric)
        'nonzerospearman' : spearman correlation only non-zero entries (numeric)
        'nonzeropearson' : pearson correlation only non-zero entries (numeric)
        function : use this function to calculate the t-statistic (input is data,labels, output is array of float)

    transform : str or ''
        transformation to apply to the data before caluculating the statistic
        'rankdata' : rank transfrom each OTU reads
        'logdata' : calculate log2 for each OTU using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'apdata' : convert to binary absence/presence

    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform

    output:
    reject : np array of bool (length N)
        True for OTUs where the null hypothesis is rejected
    tstat : np array of float (length N)
        the t-statistic value for each OTU (for effect size)
    '''

    # transform the data
    if transform == 'rankdata':
        data = rankdata(data)
    elif transform == 'logdata':
        data = logdata(data)
    elif transform == 'apdata':
        data = apdata(data)
    elif transform == 'normdata':
        data = normdata(data)

    numbact = np.shape(data)[0]

    if method == "meandiff":
        logger.debug('mean diff (matrix multiplication')
        # do matrix multiplication based calculation
        method = meandiff
        t = method(data, labels)
        tstat = method(data, labels, useabs=False)
        numsamples = np.shape(data)[1]
        p = np.zeros([numsamples, numperm])
        k1 = 1/np.sum(labels == 0)
        k2 = 1/np.sum(labels == 1)
        for cperm in range(numperm):
            np.random.shuffle(labels)
            p[labels == 0, cperm] = k1
        p2 = np.ones(p.shape)*k2
        p2[p > 0] = 0
        mean1 = np.dot(data, p)
        mean2 = np.dot(data, p2)
        u = np.abs(mean1 - mean2)

    elif method == 'mannwhitney':
        method = mannwhitney
        t = method(data, labels)
        tstat = t.copy()
        u = np.zeros([numbact, numperm])
        for cperm in range(numperm):
            rlabels = np.random.permutation(labels)
            rt = method(data, rlabels)
            u[:, cperm] = rt

    elif method == 'kruwallis':
        method = kruwallis
        t = method(data, labels)
        tstat = t.copy()
        u = np.zeros([numbact, numperm])
        for cperm in range(numperm):
            rlabels = np.random.permutation(labels)
            rt = method(data, rlabels)
            u[:, cperm] = rt

    elif method == 'stdmeandiff':
        method = stdmeandiff
        t=method(data,labels)
        tstat=method(data,labels,useabs=False)
        u=np.zeros([numbact,numperm])
        for cperm in range(numperm):
            rlabels=np.random.permutation(labels)
            rt=method(data,rlabels)
            u[:,cperm]=rt

    elif method == 'spearman' or method == 'pearson':
        # do fast matrix multiplication based correlation
        if method == 'spearman':
            data = rankdata(data)
            labels = sp.stats.rankdata(labels)
        meanval=np.mean(data,axis=1).reshape([data.shape[0],1])
        data=data-np.repeat(meanval,data.shape[1],axis=1)
        labels=labels-np.mean(labels)
        tstat=np.dot(data, labels)
        t=np.abs(tstat)
        permlabels = np.zeros([len(labels), numperm])
        for cperm in range(numperm):
            rlabels=np.random.permutation(labels)
            permlabels[:,cperm] = rlabels
        u=np.abs(np.dot(data,permlabels))

    elif method == 'nonzerospearman' or method == 'nonzeropearson':
        t = np.zeros([numbact])
        tstat = np.zeros([numbact])
        u = np.zeros([numbact, numperm])
        for i in range(numbact):
            index = np.nonzero(data[i, :])
            label_nonzero = labels[index]
            sample_nonzero = data[i, :][index]
            if method == 'nonzerospearman':
                sample_nonzero = sp.stats.rankdata(sample_nonzero)
                label_nonzero = sp.stats.rankdata(label_nonzero)
            sample_nonzero = sample_nonzero - np.mean(sample_nonzero)
            label_nonzero = label_nonzero - np.mean(label_nonzero)
            tstat[i] = np.dot(sample_nonzero, label_nonzero)
            t[i]=np.abs(tstat[i])

            permlabels = np.zeros([len(label_nonzero), numperm])
            for cperm in range(numperm):
                rlabels=np.random.permutation(label_nonzero)
                permlabels[:,cperm] = rlabels
            u[i, :] = np.abs(np.dot(sample_nonzero, permlabels))

    elif isinstance(method, types.FunctionType):
        # if it's a function, call it
        t=method(data,labels)
        tstat=t.copy()
        u=np.zeros([numbact,numperm])
        for cperm in range(numperm):
            rlabels=np.random.permutation(labels)
            rt=method(data,rlabels)
            u[:,cperm]=rt
    else:
        print('unsupported method %s' % method)
        return None,None

    # fix floating point errors (important for permutation values!)
    for crow in range(numbact):
        closepos=np.isclose(t[crow],u[crow,:])
        u[crow,closepos]=t[crow]

    # calculate permutation p-vals
    for crow in range(numbact):
        cvec=np.hstack([t[crow],u[crow,:]])
        cvec=1-(sp.stats.rankdata(cvec,method='min')/len(cvec))
        t[crow]=cvec[0]
        u[crow,:]=cvec[1:]

    # calculate FDR
    if fdrmethod=='pbfdr':
        # sort the p-values from big to small
        sortt=list(set(t))
        sortt=np.sort(sortt)
        sortt=sortt[::-1]

        foundit=False
        allfdr=[]
        allt=[]
        for cp in sortt:
            realnum=np.sum(t<=cp)
            fdr=(realnum+np.count_nonzero(u<=cp)) / (realnum*(numperm+1))
            allfdr.append(fdr)
            allt.append(cp)
            if fdr<=alpha:
                realcp=cp
                foundit=True
                break

        if not foundit:
            # no good threshold was found
            reject=np.zeros(numbact,dtype=int)
            reject= (reject>10)  # just want to output "FALSE"
            return reject,tstat

        # fill the reject null hypothesis
        reject=np.zeros(numbact,dtype=int)
        reject= (t<=realcp)
    elif fdrmethod=='bhfdr':
        import statsmodels.sandbox.stats.multicomp
        trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
        pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
        reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='fdr_bh')
    elif fdrmethod=='byfdr':
        import statsmodels.sandbox.stats.multicomp
        trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
        pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
        reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='fdr_by')

    return reject, tstat

