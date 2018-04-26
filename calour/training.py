'''
machine learning (:mod:`calour.training`)
=========================================

.. currentmodule:: calour.training

This module contains the functions related to machine learning.

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   add_sample_metadata_as_features
'''

from collections import defaultdict
from logging import getLogger
import itertools

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import interp
from scipy.sparse import hstack
import pandas as pd
import numpy as np

from .experiment import Experiment
from .amplicon_experiment import AmpliconExperiment


logger = getLogger(__name__)


@Experiment._record_sig
def add_sample_metadata_as_features(exp: Experiment, fields, sparse=None, inplace=False):
    '''Add covariates from sample metadata to the data table as features for machine learning.

    This will convert the columns of categorical strings using one-hot
    encoding scheme and add them into the data table as new features.

    .. note:: This is only for numeric and/or nominal covariates in
    sample metadata. If you want to add a ordinal column as a feature,
    use `pandas.Series.map` to convert ordinal column to numeric
    column first.

    Examples
    --------
    >>> exp = Experiment(np.array([[1,2], [3, 4]]), sparse=False,
    ...                  sample_metadata=pd.DataFrame({'category': ['A', 'B'],
    ...                                                'ph': [6.6, 7.7]},
    ...                                               index=['s1', 's2']),
    ...                  feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['otu1', 'otu2']))
    >>> exp
    Experiment with 2 samples, 2 features

    Let's add the columns of `category` and `ph` as features into data table:

    >>> new = exp.add_sample_metadata_as_features(['category', 'ph'])
    >>> new
    Experiment with 2 samples, 5 features
    >>> new.feature_metadata
               motile
    category=A    NaN
    category=B    NaN
    ph            NaN
    otu1            y
    otu2            n
    >>> new.data  # doctest: +SKIP
    array([[1. , 0. , 6.6, 1. , 2. ],
           [0. , 1. , 7.7, 3. , 4. ]])

    Parameters
    ----------
    fields : list of str
        the column names in the sample metadata. These columns will be
        converted to one-hot numeric code and then concatenated to the
        data table
    sparse : bool or None, optional
        use sparse or dense data matrix. When it is ``None``, it will follow
        the same sparsity of the current data table in the :class:`.Experiment` object
    inplace : bool
        change the :class:`.Experiment` object in place or return a copy of changed.

    Returns
    -------
    Experiment

    See Also
    --------
    sklearn.preprocessing.OneHotEncoder
    '''
    logger.debug('Add the sample metadata {} as features'.format(fields))
    if inplace:
        new = exp
    else:
        new = exp.copy()
    md = new.sample_metadata[fields]
    if sparse is None:
        sparse = new.sparse

    vec = DictVectorizer(sparse=sparse)
    encoded = vec.fit_transform(md.to_dict(orient='records'))

    if sparse:
        new.data = hstack((encoded, new.data), format='csr')
    else:
        new.data = np.concatenate([encoded, new.data], axis=1)
    # the order in the concatenation should be consistent with the data table
    new.feature_metadata = pd.concat([pd.DataFrame(index=vec.get_feature_names()), new.feature_metadata])
    return new


def split_train_test(exp: Experiment, field, test_size, train_size=None, stratify=None, random_state=None):
    '''Split experiment data into train and test set.

    '''
    if isinstance(stratify, str):
        stratify = exp.sample_metadata[stratify]
    y = exp.sample_metadata[field]
    train_X, test_X, train_y, test_y = train_test_split(
        exp.data, y, test_size=test_size, train_size=train_size, stratify=stratify, random_state=random_state)
    return train_X, test_X, train_y, test_y


def classify_cv(exp: Experiment, field, estimator, cv, param_grid=None, pos_label=None):
    '''Do the CV

    Yields
    ------
    tuple of model and figure
    '''
    X = exp.data
    y = exp.sample_metadata[field]
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    cmap = mpl.cm.get_cmap('Dark2')
    col = dict(zip(y.unique(), itertools.cycle(cmap.colors)))

    if param_grid is None:
        # use sklearn default param values for the given estimator
        param_grid = [{}]
    mean_fpr = np.linspace(0, 1, 100)
    pos_label_ = None
    for param in param_grid:
        logger.debug('run classification with parameters: %r' % param)
        tprs = defaultdict(list)
        aucs = defaultdict(list)
        for train, test in cv.split(X, y):
            model = estimator(**param)
            model.fit(X[train], y[train])
            probas = model.predict_proba(X[test])
            # binarize with model.classes_ to make sure it orders the classes in the same order
            y_test = label_binarize(y[test], classes=model.classes_)
            if len(model.classes_) == 2:
                pos_label_ = np.where(model.classes_ == pos_label)[0][0]
            for i in range(y_test.shape[1]):
                cls = model.classes_[i]
                fpr, tpr, thresholds = roc_curve(y_test[:, i], probas[:, i], pos_label=pos_label_)
                mean_tpr = interp(mean_fpr, fpr, tpr)
                tprs[cls].append(mean_tpr)
                tprs[cls][-1][0] = 0.0
                roc_auc = auc(mean_fpr, mean_tpr)
                aucs[cls].append(roc_auc)

        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='black', label='Luck', alpha=.5)
        for cls in tprs:
            mean_tpr = np.mean(tprs[cls], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs[cls])
            std_auc = np.std(aucs[cls])
            ax.plot(mean_fpr, mean_tpr, color=col[cls],
                    label='{0} ({1:.2f} $\pm$ {2:.2f})'.format(cls, mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs[cls], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col[cls], alpha=.5)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(field)
        ax.legend(loc="lower right")

        yield model, ax


@Experiment._record_sig
def learning_curve_depths(exp: AmpliconExperiment, field, groups=None,
                          train_depths=np.array([0.1, 0.325, 0.55, 0.775, 1.]),
                          cv=None, scoring=None, exploit_incremental_learning=False,
                          n_jobs=1, pre_dispatch='all', verbose=0, shuffle=False,
                          random_state=None):
    '''Compute the learning curve with regarding to sequencing depths.'''
