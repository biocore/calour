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
   plot_prc
   regress
   plot_scatter
   add_sample_metadata_as_features
'''

from logging import getLogger
import warnings
import itertools

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import (train_test_split,
                                     RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.model_selection._split import check_cv, _RepeatedSplits
from sklearn.base import is_classifier, clone
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix
from scipy import stats
from scipy.sparse import hstack
import pandas as pd
import numpy as np

from .experiment import Experiment
from .amplicon_experiment import AmpliconExperiment


logger = getLogger(__name__)


def add_sample_metadata_as_features(exp: Experiment, fields, sparse=None, inplace=False) -> Experiment:
    '''Add covariates from sample metadata to the data table as features for machine learning.

    This will convert the columns of categorical strings using one-hot
    encoding scheme and add them into the data table as new features.

    .. note:: This is only for numeric and/or nominal covariates in
       sample metadata. If you want to add a ordinal column as a
       feature, use :func:`pandas.Series.map` to convert ordinal column to
       numeric column first.

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
    new.feature_metadata = pd.concat([pd.DataFrame(index=vec.get_feature_names()), new.feature_metadata], sort=False)
    return new


def split_train_test(exp: Experiment,
                     # the following parameters are passed to `sklearn.model_selection.train_test_split`
                     test_size, train_size=None, stratify=None, shuffle=False, random_state=None):
    '''Split experiment into train experiment and test experiment.

    Parameters
    ----------
    test_size, train_size, shuffle, random_seed :
        They are passed to :func:`sklearn.model_selection.train_test_split`.
        Please check documentation there.
    stratify : str or array-like or None
        If it is array-like or None, it is directly passed to
        :func:`sklearn.model_selection.train_test_split`. If it is a
        str, it must be a valid column name in the
        `Experiment.sample_metadata` and this column will be passed to
        the function for stratified split.

    Returns
    -------
    tuple of 2 Experiment objects
        train, test
    '''
    if isinstance(stratify, str):
        stratify = exp.sample_metadata[stratify]
    train_idx, test_idx = train_test_split(
        range(exp.shape[0]), test_size=test_size, train_size=train_size,
        stratify=stratify, random_state=random_state)

    return exp.reorder(train_idx), exp.reorder(test_idx)


class SortedStratifiedKFold(StratifiedKFold):
    '''Stratified K-Fold cross validator.

    Please see :class:`sklearn.model_selection.StratifiedKFold` for
    documentation for parameters, etc. It is very similar to that class
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
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

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

    def _make_test_folds(self, X, y=None):
        '''The sole purpose of this function is to suppress the specific unintended warning from sklearn.'''
        with warnings.catch_warnings():
            # suppress specific warnings
            warnings.filterwarnings("ignore", message="The least populated class in y has only 1 members, which is less than n_splits=")
            return super()._make_test_folds(X, y)


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
        super().__init__(SortedStratifiedKFold, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def regress(exp: Experiment, field, estimator, cv=RepeatedSortedStratifiedKFold(n_splits=3, n_repeats=1), params=None):
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


def plot_scatter(result, title='', cmap=None, cor=stats.pearsonr, cv=False, ax=None, **kwargs):
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
    cv : boolean
        Whether to color the plot by different folds of cross validation. You need to have
        'CV' column in the input `result` data frame.
    ax : matplotlib.axes.Axes or None (default), optional
        The axes where the confusion matrix is plotted. None (default) to create a new figure and
        axes to plot the confusion matrix
    kwargs : dict
        keyword arguments passing to :func:`matplotlib.pyplot.scatter`

    Returns
    -------
    matplotlib.axes.Axes
        The axes for the confusion matrix

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.tab10
    colors = itertools.cycle(cmap.colors)
    if ax is None:
        fig, ax = plt.subplots()
    # ax.axis('equal')
    if cv is True:
        for c, (grp, df) in zip(colors, result.groupby('CV')):
            ax.scatter(df['Y_TRUE'], df['Y_PRED'],
                       color=c,
                       label='CV {}'.format(grp),
                       **kwargs)
        ax.legend(loc="lower right")
    else:
        ax.scatter(result['Y_TRUE'], result['Y_PRED'], **kwargs)

    m1 = result[['Y_TRUE', 'Y_PRED']].min()
    m2 = result[['Y_TRUE', 'Y_PRED']].max()
    ax.plot([m1, m2], [m1, m2], color='black', alpha=.3)
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.set_title(title)

    if cor is not None:
        r, p = cor(result['Y_TRUE'], result['Y_PRED'])
        ax.annotate("r={0:.2f} p-value={1:.3f}".format(r, p), xy=(.1, .95), xycoords=ax.transAxes)
    return ax


def classify(exp: Experiment, fields, estimator, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=1),
             predict='predict_proba', params=None):
    '''Evaluate classification during cross validation.

    Parameters
    ----------
    fields : str or list of str
        column name(s) in the sample metadata, which contains the classes we want to predict.
        If it is a list of str, this function does multi-task (aka multioutput-multiclass)
        classification and you must provide an `estimator` of multi-task classifier. See
        `http://scikit-learn.org/stable/modules/multiclass.html` for more information.
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
    y = exp.sample_metadata[fields]
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


def plot_cm(result, normalize=False, title='confusion matrix', cmap=None, ax=None, classes=None, **kwargs):
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
    classes : list
        The list of the labels you want to include in the plot in the order specified in the list.
    kwargs : dict
        keyword argument passing to :func:`matplotlib.pyplot.imshow`. For example, you can pass
        `vmin=0, vmax=1` as keyword arguments to manually define color range (especially useful
        when you set `normalize=True`)

    Returns
    -------
    matplotlib.axes.Axes
        The axes for the confusion matrix

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.Blues
    # if labels is given, use it (and its order); otherwise:
    if classes is None:
        classes = np.unique(result['Y_TRUE'].values)
        classes.sort()

    cm = _compute_cm(result, labels=classes)
    accuracy = cm.trace() / cm.sum()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.debug("Normalized confusion matrix")
    else:
        logger.debug('Confusion matrix, without normalization')
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    img = ax.imshow(cm, cmap=cmap, **kwargs)
    ax.set_title('{0}\naccuracy: {1:.1%}'.format(title, accuracy))
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
    # plot a diagonal line
    ax.plot([0, 1], [1, 0], alpha=0.3, color='gray', linewidth=1, transform=ax.transAxes)

    ax.set_ylabel('Observation')
    ax.set_xlabel('Prediction')
    # disable grid explicitly because seaborn default style plot grid
    # lines, which is very disturbing
    ax.grid(False)
    fig.tight_layout()
    return ax


def _compute_cm(result, labels, **kwargs):
    if 'Y_PRED' not in result.columns:
        idx = np.argmax(result[labels].values, axis=1)
        y_pred = [labels[i] for i in idx]
    else:
        y_pred = result['Y_PRED']
    return confusion_matrix(result['Y_TRUE'], y_pred, labels=labels, **kwargs)


def plot_prc(result, classes=None, title='precision-recall curve', cmap=None, ax=None):
    '''Plot precision-recall curve.

    Parameters
    ----------
    result : pandas.DataFrame
        data frame containing predictions per sample (in row). It must have a column of
        true class named "Y_TRUE" and multiple columns of predicted probabilities for each class.
        It typically takes the output of :func:`classify`.
    classes: list
        The list of the labels you want to include in the plot in the order specified in the list.
        If it is a binary classification (eg "Health" vs. "IBD"), you would want
        to set it to `["IBD"]` and don't plot for "Health" because it is equivalent.
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
        The axes for the curve

    See Also
    --------
    plot_roc
    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.Dark2
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_aspect('equal')

    if classes is None:
        classes = np.unique(result['Y_TRUE'].values)
        classes.sort()

    col = dict(zip(classes, itertools.cycle(cmap.colors)))

    lines = []
    labels = []
    for cls in classes:
        aucs = []
        for grp, df in result.groupby('CV'):
            y_true = df['Y_TRUE'].values == cls
            auc = average_precision_score(y_true, df[cls])
            aucs.append(auc)
        precision, recall, thresholds = precision_recall_curve(
            result['Y_TRUE'].values == cls, result[cls])
        line, = ax.step(recall, precision, color=col[cls], lw=2, alpha=.5)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        lines.append(line)
        labels.append('{0} ({1:.2f} $\\pm$ {2:.2f})'.format(cls, mean_auc, std_auc))

    # plot iso-f1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.25)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.8, y[45] + 0.02))
    lines.append(l)
    labels.append('iso-f1 curves')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(lines, labels, loc='lower left', fancybox=True, framealpha=0.5)
    return ax


def _interpolate_precision_recall(x, recall, precision):
    '''https://stackoverflow.com/a/39862264/489564'''
    increasing_max_precision = np.maximum.accumulate(precision[::-1])
    y = np.zeros(x.shape)
    for xpi, ypi in zip(reversed(recall), increasing_max_precision):
        y[x <= xpi] = ypi

    return y


def plot_roc(result, classes=None, title='ROC', cv=True, cmap=None, ax=None):
    '''Plot ROC curve.

    .. note:: You may want to consider using precision-recall curve
       (:func:`plot_prc`) instead of ROC curve. If your model needs to
       perform equally well on the negative class as the positive
       class, you would use the ROC AUC. For example, for classifying
       images between cats and dogs, if you would like the model to
       perform well on the cats as well as on the dogs, then you can
       use ROC and ROC is more intuitive to understand. If you have
       *imbalanced* classes OR you don't care about the negative class
       at all, then you should use precision-recall curve. Take cancer
       diagnosis as an example, you tends to have way more negative
       samples than positive cancer samples and want to make sure you
       positive predictions are correct (precision) and don't miss any
       cancer (recall or aka sensitivity), you should use
       precision-recall curve. If the classes are balanced, ROC
       usually works fine even in this scenario. [1]_.

    Parameters
    ----------
    result : pandas.DataFrame
        data frame containing predictions per sample (in row). It must have a column of
        true class named "Y_TRUE" and multiple columns of predicted probabilities for each class.
        It typically takes the output of :func:`classify`.
    classes: list
        The list of the labels you want to include in the plot in the order specified in the list.
        If it is a binary classification (eg "Health" vs. "IBD"), you would want
        to set it to `["IBD"]` and don't plot for "Health" because it is equivalent.
    title : str
        plot title
    cv : boolean
        Whether to plot ROC curve shade by different folds of cross validation. You need to have
        'CV' column in the input `result` data frame.
    cmap : str or matplotlib.colors.ListedColormap
        str to indicate the colormap name. Default is "Dark2" colormap.
        For all available colormaps in matplotlib: https://matplotlib.org/users/colormaps.html
    ax : matplotlib.axes.Axes or None (default), optional
        The axes where to plot. None (default) to create a new figure and
        axes to plot

    Returns
    -------
    tuple of matplotlib.axes.Axes and float
        The axes for the ROC and the AUC value

    References
    ----------
    .. [1] Saito T and Rehmsmeier M (2015) The Precision-Recall Plot
       Is More Informative than the ROC Plot When Evaluating Binary
       Classifiers on Imbalanced Datasets. PLoS One, 10.

    '''
    from matplotlib import pyplot as plt
    if cmap is None:
        cmap = plt.cm.Dark2
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='black', label='Luck', alpha=.5)

    if classes is None:
        classes = np.unique(result['Y_TRUE'].values)
        classes.sort()

    col = dict(zip(classes, itertools.cycle(cmap.colors)))

    mean_fpr = np.linspace(0, 1, 100)
    for cls in classes:
        tprs = []
        aucs = []
        if cv is True:
            for grp, df in result.groupby('CV'):
                y_true = df['Y_TRUE'].values == cls
                fpr, tpr, thresholds = roc_curve(y_true.astype(int), df[cls])
                if np.isnan(fpr[-1]) or np.isnan(tpr[-1]):
                    logger.warning(
                        'The cross validation fold %r is skipped because the true positive rate or '
                        'false positive rate computation failed. This is likely because you '
                        'have either no true positive or no negative samples in this '
                        'cross validation for the class %r' % (grp, cls))
                    continue
                mean_tpr = np.interp(mean_fpr, fpr, tpr)
                tprs.append(mean_tpr)
                tprs[-1][0] = 0.0
                roc_auc = auc(mean_fpr, mean_tpr)
                aucs.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color=col[cls],
                    label='{0} ({1:.2f} $\\pm$ {2:.2f})'.format(cls, mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col[cls], alpha=.5)
        else:
            y_true = result['Y_TRUE'].values == cls
            fpr, tpr, thresholds = roc_curve(y_true.astype(int), result[cls])
            if np.isnan(fpr[-1]) or np.isnan(tpr[-1]):
                logger.warning(
                    'The class %r is skipped because the true positive rate or '
                    'false positive rate computation failed. This is likely because you '
                    'have either no true positive or no negative samples for this class' % cls)
            roc_auc = auc(fpr, tpr)
            # prepend zero because, if not, the curve may start in the middle of the plot.
            ax.plot(np.insert(fpr, 0, 0), np.insert(tpr, 0, 0), label='{0} ({1:.2f})'.format(cls, roc_auc), lw=2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")

    return ax, mean_auc if cv else roc_auc


def plot_calibration(y_true, y_prob, bins=10):
    from sklearn.calibration import calibration_curve
    from matplotlib import pyplot as plt

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=bins)

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set_ylabel("Fraction of young samples")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title('reliability curve')

    ax2.hist(y_prob, range=(0, 1), bins=bins, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    return fig


def learning_curve_depths(exp: AmpliconExperiment, field, groups=None,
                          train_depths=np.array([0.1, 0.325, 0.55, 0.775, 1.]),
                          cv=None, scoring=None, exploit_incremental_learning=False,
                          n_jobs=1, pre_dispatch='all', verbose=0, shuffle=False,
                          random_state=None):
    '''Compute the learning curve with regarding to sequencing depths.'''
