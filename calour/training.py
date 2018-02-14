'''
machine learning (:mod:`calour.training`)
=========================================

.. currentmodule:: calour.training

This module contains the functions related to machine learning.

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   onehot_encode_features
'''


from logging import getLogger

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
import pandas as pd
import numpy as np

from .experiment import Experiment


logger = getLogger(__name__)


@Experiment._record_sig
def onehot_encode_features(exp: Experiment, fields, sparse=None, inplace=False):
    '''Add covariates from sample metadata to the data table as features for machine learning.

    This will convert the columns of categorical strings using one-hot encoding scheme and add them
    into the data table as new features.

    Examples
    --------
    >>> exp = Experiment(np.array([[1,2], [3, 4]]), sparse=False,
    ...                  sample_metadata=pd.DataFrame({'category': ['A', 'B'],
    ...                                                'ph': [6.6, 7.7]},
    ...                                               index=['s1', 's2']),
    ...                  feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['otu1', 'otu2']))
    >>> exp
    Experiment
    ----------
    data dimension: 2 samples, 2 features

    Let's add the columns of `category` and `ph` as features into data table:

    >>> new = exp.onehot_encode_features(['category', 'ph'])
    >>> new
    Experiment
    ----------
    data dimension: 2 samples, 5 features
    >>> new.feature_metadata
               motile
    category=A    NaN
    category=B    NaN
    ph            NaN
    otu1            y
    otu2            n
    >>> new.data
    array([[1. , 0. , 6.6, 1. , 2. ],
           [0. , 1. , 7.7, 3. , 4. ]])

    Parameters
    ----------
    fields : list of str
        the column names in the sample metadata. These columns will be
        converted to one-hot numeric code and then concatenated to the
        data table
    sparse : bool or ``None`` (optional)
        use sparse or dense data matrix. When it is ``None``, it will follow
        the same sparsity of the current data table in the :class:`.Experiment` object
    inplace : bool
        change the :class:`.Experiment` object in place or return a copy of changed.

    Returns
    -------
    :class:`.Experiment`

    See Also
    --------
    :class:`sklearn.preprocessing.OneHotEncoder`
    '''
    # '''Add covariates from sample metadata to the data table as features for machine learning.

    # This will convert the columns of categorical strings using one-hot encoding scheme and add them
    # into the data table as new features.

    # Examples
    # --------
    # >>> exp = Experiment(np.array([[1,2], [3, 4]]), sparse=False,
    # ...                  sample_metadata=pd.DataFrame({'category': ['A', 'B'],
    # ...                                                'ph': [6.6, 7.7]},
    # ...                                               index=['s1', 's2']),
    # ...                  feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['otu1', 'otu2']))
    # >>> exp
    # Experiment
    # ----------
    # data dimension: 2 samples, 2 features
    # sample IDs: Index(['s1', 's2'], dtype='object')
    # feature IDs: Index(['otu1', 'otu2'], dtype='object')

    # Let's add the columns of `category` and `ph` as features into data table:

    # >>> new = exp.onehot_encode_features(['category', 'ph'])
    # >>> new
    # Experiment
    # ----------
    # data dimension: 2 samples, 5 features
    # sample IDs: Index(['s1', 's2'], dtype='object')
    # feature IDs: Index(['category=A', 'category=B', 'ph', 'otu1', 'otu2'], dtype='object')
    # >>> new.feature_metadata
    #            motile
    # category=A    NaN
    # category=B    NaN
    # ph            NaN
    # otu1            y
    # otu2            n
    # >>> new.data
    # array([[1. , 0. , 6.6, 1. , 2. ],
    #        [0. , 1. , 7.7, 3. , 4. ]])

    # Parameters
    # ----------
    # fields : list of str
    #     the column names in the sample metadata. These columns will be
    #     converted to one-hot numeric code and then concatenated to the
    #     data table
    # sparse : bool or ``None`` (optional)
    #     use sparse or dense data matrix. When it is ``None``, it will follow
    #     the same sparsity of the current data table in the :class:`.Experiment` object
    # inplace : bool
    #     change the :class:`.Experiment` object in place or return a copy of changed.

    # Returns
    # -------
    # :class:`.Experiment`

    # See Also
    # --------
    # :class:`sklearn.preprocessing.OneHotEncoder`
    # '''
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
        new.data = hstack((encoded, new.data))
    else:
        new.data = np.concatenate([encoded, new.data], axis=1)
    # the order in the concatenation should be consistent with the data table
    new.feature_metadata = pd.concat([pd.DataFrame(index=vec.get_feature_names()), new.feature_metadata])
    return new
