'''
machine learning (:mod:`calour.training`)
=========================================

.. currentmodule:: calour.training

This module contains the functions related to machine learning.


Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   SortedStratifiedKFold
   RepeatedSortedStratifiedKFold

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   classify
   plot_cm
   plot_roc
   regress
   plot_scatter
   add_sample_metadata_as_features
'''

from logging import getLogger
import itertools

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import (train_test_split,
                                     RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.model_selection._split import check_cv, _RepeatedSplits
from sklearn.base import is_classifier, clone
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import interp, stats
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


class SortedStratifiedKFold(StratifiedKFold):
    '''Stratified K-Fold cross validator.

    Please see :class:`sklearn.model_selection.StratifiedKFold` for
    documentation for parameters, etc. It is very similar to that
    except this is for regression of numeric values.

    This implementation basically assigns a unique label (int here) to
    each consecutive `n_splits` values after y is sorted. Then rely on
    StratifiedKFold to split. The idea is borrowed from this `blog
    <http://scottclowe.com/2016-03-19-stratified-regression-partitions/>`_.

    See Also
    --------
    RepeatedSortedStratifiedKFold
    '''
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def _sort_partition(self, y):
        n = len(y)
        cats = np.empty(n, dtype='u4')
        div, mod = divmod(n, self.n_splits)
        cats[:n-mod] = np.repeat(range(div), self.n_splits)
        cats[n-mod:] = div + 1
        # run argsort twice to get the rank of each y value
        return cats[np.argsort(np.argsort(y))]

    def split(self, X, y, groups=None):
        y_cat = self._sort_partition(y)
        return super().split(X, y_cat, groups)


class RepeatedSortedStratifiedKFold(_RepeatedSplits):
    '''Repeated Stratified K-Fold cross validator.

    Please see :class:`sklearn.model_selection.RepeatedStratifiedKFold` for
    documentation for parameters, etc. It is very similar to that
    except this is for regression of numeric values.

    See Also
    --------
    SortedStratifiedKFold
    '''
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(SortedStratifiedKFold, n_repeats, random_state, n_splits=n_splits)


def regress(exp: Experiment, field, estimator, cv=RepeatedSortedStratifiedKFold(3, 1), params=None):
    '''Evaluate regression during cross validation.

    Parameters
    ----------
    field : str
        column name in the sample metadata, which contains the variable we want to predict.
    estimator : estimator object implementing `fit` and `predict`
        scikit-learn estimator. e.g. :class:`sklearn.ensemble.RandomForestRegressor`
    cv : int, cross-validation generator or an iterable
        similar to the `cv` parameter in :class:`sklearn.model_selection.GridSearchCV`
    params : dict of string to sequence, or sequence of such
        For example, the output of
        :class:`sklearn.model_selection.ParameterGrid` or
        :class:`sklearn.model_selection.ParameterSampler`. By default,
        it uses whatever default parameters of the `estimator` set in
        `scikit-learn`

    Yields
    ------
    pandas.DataFrame
        The result of prediction per sample for a given parameter set. It contains the
        following columns:

        - Y_TRUE: the true value for the samples
        - SAMPLE: sample IDs
        - CV: which split of the cross validation
        - Y_PRED: the predicted value for the samples
    '''
    X = exp.data
    y = exp.sample_metadata[field]
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if params is None:
        # use sklearn default param values for the given estimator
        params = [{}]

    for param in params:
        logger.debug('run regression with parameters: %r' % param)
        dfs = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            # deep copy the model by clone to avoid the impact from last iteration of fit.
            model = clone(estimator)
            model = model.set_params(**param)
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            df = pd.DataFrame({'Y_PRED': pred,
                               'Y_TRUE': y[test].values,
                               'SAMPLE': y[test].index.values,
                               'CV': i})
            dfs.append(df)
        yield pd.concat(dfs, axis=0).reset_index(drop=True)


def plot_scatter(result, title='', cmap=None, cor=stats.pearsonr, ax=None):
    '''Plot prediction vs. observation for regression.

    Parameters
    ----------
    result : pandas.DataFrame
        data frame containing predictions per sample (in row). It must have a column of
        true class named "Y_TRUE". It must have a column of predicted class named "Y_PRED".
        It typically takes the output of :func:`classify`.
    title : str
        plot title
    cmap : str or matplotlib.colors.ListedColormap
        str to indicate the colormap name. Default is "Blues" colormap.
        For all available colormaps in matplotlib: https://matplotlib.org/users/colormaps.html
    cor : Callable or None
        a correlation function that takes predicted y and observed y as inputs and returns
        correlation coefficient and p-value. If None, don't compute and label correlation
        on the plot.
    ax : matplotlib.axes.Axes or None (default), optional
        The axes where the confusion matrix is plotted. None (default) to create a new figure and
        axes to plot the confusion matrix

    Returns
    -------
    matplotlib.axes.Axes
        The axes for the confusion matrix

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.tab10
    if ax is None:
        fig, ax = plt.subplots()
    # ax.axis('equal')
    for i, (grp, df) in enumerate(result.groupby('CV')):
        ax.scatter(df['Y_TRUE'], df['Y_PRED'],
                   color=cmap.colors[i],
                   label='CV {}'.format(grp))

    m1 = result[['Y_TRUE', 'Y_PRED']].min()
    m2 = result[['Y_TRUE', 'Y_PRED']].max()
    ax.plot([m1, m2], [m1, m2], color='black', alpha=.3)
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.set_title(title)
    ax.legend(loc="lower right")
    if cor is not None:
        r, p = cor(result['Y_TRUE'], result['Y_PRED'])
        ax.annotate("r={0:.2f} p-value={1:.3f}".format(r, p), xy=(.1, .95), xycoords=ax.transAxes)
    return ax


def classify(exp: Experiment, field, estimator, cv=RepeatedStratifiedKFold(3, 1),
             predict='predict_proba', params=None):
    '''Evaluate classification during cross validation.

    Parameters
    ----------
    field : str
        column name in the sample metadata, which contains the classes we want to predict.
    estimator : estimator object implementing `fit` and `predict`
        scikit-learn estimator. e.g. :class:`sklearn.ensemble.RandomForestClassifer`
    cv : int, cross-validation generator or an iterable
        similar to the `cv` parameter in :class:`sklearn.model_selection.GridSearchCV`
    predict : {'predict', 'predict_proba'}
        the function used to predict the validation sets. Some estimators
        have both functions to predict class or predict the probablity of each class
        for a sample. For example, see :class:`sklearn.ensemble.RandomForestClassifier`
    params : dict of string to sequence, or sequence of such
        For example, the output of
        :class:`sklearn.model_selection.ParameterGrid` or
        :class:`sklearn.model_selection.ParameterSampler`. By default,
        it uses whatever default parameters of the `estimator` set in
        `scikit-learn`

    Yields
    ------
    pandas.DataFrame
        The result of prediction per sample for a given parameter set. It contains the
        following columns:

        - Y_TRUE: the true class for the samples
        - SAMPLE: sample IDs
        - CV: which split of the cross validation
        - Y_PRED: the predicted class for the samples (if "predict")
        - mutliple columns with each contain probabilities predicted as each class (if "predict_proba")

    '''
    X = exp.data
    y = exp.sample_metadata[field]
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if params is None:
        # use sklearn default param values for the given estimator
        params = [{}]

    for param in params:
        logger.debug('run classification with parameters: %r' % param)
        dfs = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            # deep copy the model by clone to avoid the impact from last iteration of fit.
            model = clone(estimator)
            model = model.set_params(**param)
            model.fit(X[train], y[train])
            pred = getattr(model, predict)(X[test])
            if pred.ndim > 1:
                df = pd.DataFrame(pred, columns=model.classes_)
            else:
                df = pd.DataFrame(pred, columns=['Y_PRED'])
            df['Y_TRUE'] = y[test].values
            df['SAMPLE'] = y[test].index.values
            df['CV'] = i
            dfs.append(df)
        yield pd.concat(dfs, axis=0).reset_index(drop=True)


def plot_cm(result, normalize=False, title='confusion matrix', cmap=None, ax=None):
    '''Plot confusion matrix

    Parameters
    ----------
    result : pandas.DataFrame
        data frame containing predictions per sample (in row). It must have a column of
        true class named "Y_TRUE". It must have a column of predicted class named "Y_PRED"
        or multiple columns of predicted probabilities for each class. It typically takes
        the output of :func:`classify`.
    normalize : bool
        normalize the confusion matrix or not
    title : str
        plot title
    cmap : str or matplotlib.colors.ListedColormap
        str to indicate the colormap name. Default is "Blues" colormap.
        For all available colormaps in matplotlib: https://matplotlib.org/users/colormaps.html
    ax : matplotlib.axes.Axes or None (default), optional
        The axes where the confusion matrix is plotted. None (default) to create a new figure and
        axes to plot the confusion matrix

    Returns
    -------
    matplotlib.axes.Axes
        The axes for the confusion matrix

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.Blues
    classes = result['Y_TRUE'].unique()
    cm = _compute_cm(result, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.debug("Normalized confusion matrix")
    else:
        logger.debug('Confusion matrix, without normalization')
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    img = ax.imshow(cm, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(img)
    tick_marks = np.arange(len(classes))
    ax.tick_params(rotation=45)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Observation')
    ax.set_xlabel('Prediction')
    fig.tight_layout()
    return ax


def _compute_cm(result, labels, **kwargs):
    if 'Y_PRED' not in result.columns:
        idx = np.argmax(result[labels].values, axis=1)
        y_pred = [labels[i] for i in idx]
    else:
        y_pred = result['Y_PRED']
    return confusion_matrix(result['Y_TRUE'], y_pred, labels=labels, **kwargs)


def plot_roc(result, pos_label=None, title='ROC', cmap=None, ax=None):
    '''Plot ROC curve.

    Parameters
    ----------
    result : pandas.DataFrame
        data frame containing predictions per sample (in row). It must have a column of
        true class named "Y_TRUE" and multiple columns of predicted probabilities for each class.
        It typically takes the output of :func:`classify`.
    pos_label : str, optional
        the interested class if it is a binary classification. For
        example, for "Health" vs. "IBD" classification, you would want
        to set it to "IBD". This is ignored if it is not a binary
        classification. This value is passed to
        :func:`sklearn.metrics.roc_curve`
    title : str
        plot title
    cmap : str or matplotlib.colors.ListedColormap
        str to indicate the colormap name. Default is "Dark2" colormap.
        For all available colormaps in matplotlib: https://matplotlib.org/users/colormaps.html
    ax : matplotlib.axes.Axes or None (default), optional
        The axes where to plot. None (default) to create a new figure and
        axes to plot

    Returns
    -------
    matplotlib.axes.Axes
        The axes for the ROC

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.Dark2
    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('equal')
    ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='black', label='Luck', alpha=.5)

    classes = result['Y_TRUE'].unique()
    # if this is a binary classification, we only need to set one class as positive
    # and just plot ROC for the positive class
    if len(classes) == 2:
        if pos_label is None:
            # by default use whatever the first class is as positive
            classes = classes[:1]
        else:
            classes = [pos_label]
    col = dict(zip(classes, itertools.cycle(cmap.colors)))

    mean_fpr = np.linspace(0, 1, 100)
    for cls in classes:
        tprs = []
        aucs = []
        for grp, df in result.groupby('CV'):
            y_true = df['Y_TRUE'].values == cls
            fpr, tpr, thresholds = roc_curve(y_true, df[cls])
            mean_tpr = interp(mean_fpr, fpr, tpr)
            tprs.append(mean_tpr)
            tprs[-1][0] = 0.0
            roc_auc = auc(mean_fpr, mean_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=col[cls],
                label='{0} ({1:.2f} $\pm$ {2:.2f})'.format(cls, mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col[cls], alpha=.5)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")

    return ax


@Experiment._record_sig
def learning_curve_depths(exp: AmpliconExperiment, field, groups=None,
                          train_depths=np.array([0.1, 0.325, 0.55, 0.775, 1.]),
                          cv=None, scoring=None, exploit_incremental_learning=False,
                          n_jobs=1, pre_dispatch='all', verbose=0, shuffle=False,
                          random_state=None):
    '''Compute the learning curve with regarding to sequencing depths.'''
