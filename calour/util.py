'''
utilities (:mod:`calour.util`)
==============================

.. currentmodule:: calour.util

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   join_fields
   compute_prevalence
   register_functions
   set_log_level
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import os
import hashlib
import inspect
import re
import configparser
import warnings
from types import FunctionType
from functools import wraps, update_wrapper
from importlib import import_module
from collections.abc import Sequence
from logging import getLogger
from numbers import Real
from pkg_resources import resource_filename

import numpy as np
import scipy


logger = getLogger(__name__)


def join_fields(df, field1, field2, new_field=None, sep='_', pad=None):
    '''Join two fields into a single new field

    Parameters
    ----------
    df : pandas.DataFrame
    field1 : str
        Name of the first field to join. The value in this column can be any data type.
    field2 : str
        Name of the field to join. The value in this column can be any data type.
    new_field : str, default=None
        name of the new (joined) field. Default to name it as field1 + sep + field2
    sep : str, optional
        The separator between the values of the two fields when joining
    pad : str, default=None
        Padding char. Align and pad the text in field1 and field2 before joining. Default to join without padding.

    Returns
    -------
    pandas.DataFrame
        the original data frame with new joined field.

    Examples
    --------
    >>> import pandas as pd
    >>> pd.set_option('display.max_colwidth', None)
    >>> df = pd.DataFrame([['dog', 'bone'], ['monkey', 'banana']], columns=['animal', 'food'])
    >>> # pandas display on Mac is problematic with ellipsis, skip it for now.
    >>> join_fields(df, 'animal', 'food')                            #doctest: +SKIP
       animal    food    animal_food
    0     dog    bone       dog_bone
    1  monkey  banana  monkey_banana
    >>> join_fields(df, 'animal', 'food', new_field='new', pad='-')  #doctest: +SKIP
       animal    food    animal_food            new
    0     dog    bone       dog_bone  dog---_--bone
    1  monkey  banana  monkey_banana  monkey_banana
    '''
    logger.debug('joining fields %s and %s into %s' % (field1, field2, new_field))

    # validate the data
    if field1 not in df.columns:
        raise ValueError('field %s not in the data frame' % field1)
    if field2 not in df.columns:
        raise ValueError('field %s not in the data frame' % field2)

    # get the new column name
    if new_field is None:
        new_field = field1 + sep + field2

    if new_field in df.columns:
        raise ValueError('new field name %s already exists in df. Please use different new_field value' % new_field)

    col1 = df[field1].astype(str)
    max1 = col1.str.len().max()
    col2 = df[field2].astype(str)
    max2 = col2.str.len().max()

    if pad is not None:
        col1 = col1.str.pad(width=max1, side='right', fillchar=pad)
        col2 = col2.str.pad(width=max2, side='left', fillchar=pad)

    df[new_field] = col1 + sep + col2

    return df


def compute_prevalence(abundance):
    '''Return the prevalence at each abundance cutoffs.

    Each sample that has the feature above the cutoff (exclusive) will
    be counted.

    Parameters
    ----------
    abundance : 1d array-like of numeric
        The abundance of a feature across samples.

    Returns
    -------
    np.ndarray
        1d sorted array that contains the unique abundance values in the input array.
    np.ndarray
        same size with the 1st 1d array. Each value in the array is
        the feature prevalence defined as its abundance > each unique
        value in the 1st array.

    Examples
    --------
    >>> abund = [0, 1, 0, 2, 4]
    >>> x, y = compute_prevalence(abund)
    >>> x   #doctest: +SKIP
    array([0, 1, 2, 4])
    >>> y   #doctest: +SKIP
    array([0.6, 0.4, 0.2, 0.])

    '''
    # unique values are sorted
    cutoffs, counts = np.unique(abundance, return_counts=True)
    cum_counts = np.cumsum(counts)
    prevalences = 1 - cum_counts / counts.sum()
    return cutoffs, prevalences


def _transition_index(obj):
    '''Return the transition index and current value of the list.

    Examples
    -------
    >>> obj = ['a', 'a', 'b']
    >>> list(_transition_index(obj))
    [(2, 'a'), (3, 'b')]
    >>> obj = ['a', 'a', 'b', 1, 2, None, None]
    >>> list(_transition_index(obj))
    [(2, 'a'), (3, 'b'), (4, 1), (5, 2), (7, None)]

    Parameters
    ----------
    obj : Iterable of arbitrary objects

    Yields
    ------
    tuple of (int, arbitrary)
        the transition index, the item value
    '''
    it = enumerate(obj)
    i, item = next(it)
    item = str(type(item)), item
    for i, current in it:
        current = str(type(current)), current
        if item != current:
            yield i, item[1]
            item = current
    yield i + 1, item[1]


def _convert_axis_name(func):
    '''Convert str value of axis to 0/1.

    This allows the decorated function with ``axis`` parameter to
    accept "sample"/"s" and "feature"/"f" as value for ``axis``
    parameter.

    This should be always the closest decorator to the function if
    you have multiple decorators for this function.

    '''
    conversion = {'sample': 0,
                  's': 0,
                  'samples': 0,
                  'feature': 1,
                  'f': 1,
                  'features': 1}

    @wraps(func)
    def inner(*args, **kwargs):
        sig = inspect.signature(func)
        ba = sig.bind(*args, **kwargs)
        param = ba.arguments
        v = param.get('axis', None)
        if v is None:
            return func(*args, **kwargs)
        if isinstance(v, str):
            param['axis'] = conversion[v.lower()]
        elif v not in {0, 1}:
            raise ValueError('unknown axis `%r`' % v)

        return func(*ba.args, **ba.kwargs)
    return inner


def _get_taxonomy_string(exp, sep=';', remove_underscore=True, to_lower=False):
    '''Get a nice taxonomy string.

    Convert the taxonomy list stored (from biom.read_table) to a single string per feature.

    Parameters
    ----------
    exp : Experiment
        with the taxonomy entry in the feature_metadata
    sep : str, optional
        the output separator to use between the taxonomic levels
    remove_underscore : bool, optional
        True (default) to remove the entries like 'g__' and missing values
        False to keep them
    to_lower : bool, optional
        False (default) to keep case
        True to convert to lowercase

    Returns
    -------
    taxonomy : list of str
        list of taxonomy string per feature
    '''
    # test if we have taxonomy in the feature metadata
    logger.debug('getting taxonomy string')
    if 'taxonomy' not in exp.feature_metadata.columns:
        raise ValueError('No taxonomy field in experiment')

    # if it is not a list - just return it
    if not isinstance(exp.feature_metadata['taxonomy'][0], list):
        return list(exp.feature_metadata['taxonomy'].values)

    if not remove_underscore:
        taxonomy = [sep.join(x) for x in exp.feature_metadata['taxonomy']]
    else:
        taxonomy = []
        for ctax in exp.feature_metadata['taxonomy']:
            taxstr = ''
            for clevel in ctax:
                clevel = clevel.strip()
                if len(clevel) > 3:
                    if clevel[1:3] == '__':
                        clevel = clevel[3:]
                    taxstr += clevel + sep
            if len(taxstr) == 0:
                taxstr = 'na'
            taxonomy.append(taxstr)

    if to_lower:
        taxonomy = [x.lower() for x in taxonomy]
    return taxonomy


def get_file_md5(f, encoding='utf-8'):
    '''get the md5 of the text file.

    Parameters
    ----------
    f : str
        name of the file to calculate md5 on
    encoding : str or None, optional
        encoding of the text file (see python str.encode() ). None to use 'utf-8'

    Returns
    -------
    flmd5: str
        the md5 of the file f
    '''
    logger.debug('getting file md5 for file %s' % f)
    if f is None:
        return None
    with open(f, 'rb') as fl:
        flmd5 = hashlib.md5()
        chunk_size = 4096
        for chunk in iter(lambda: fl.read(chunk_size), b""):
            flmd5.update(chunk)
        flmd5 = flmd5.hexdigest()
        logger.debug('md5 of %s: %s' % (f, flmd5))
        return flmd5


def get_data_md5(data):
    '''Calculate the md5 of a dense/sparse matrix

    Calculat matrix md5 based on row by row order

    Parameters
    ----------
    data : dense or sparse matrix

    Returns
    -------
    datmd5 : str
        the md5 of the data
    '''
    logger.debug('caculating data md5')
    if scipy.sparse.issparse(data):
        # if sparse need to convert to numpy array
        data = data.toarray()
    # convert to string of raw data since hashlib.md5 does not take numpy array as input
    datmd5 = hashlib.md5(data.tobytes())
    datmd5 = datmd5.hexdigest()
    logger.debug('data md5 is: %s' % datmd5)
    return datmd5


def get_config_file():
    '''Get the calour config file location

    If the environment CALOUR_CONFIG_FILE is set, take the config file from it
    otherwise return CALOUR_PACKAGE_LOCATION/calour/calour.config

    Returns
    -------
    config_file_name : str
        the full path to the calour config file
    '''
    if 'CALOUR_CONFIG_FILE' in os.environ:
        config_file_name = os.environ['CALOUR_CONFIG_FILE']
        logger.debug('Using calour config file %s from CALOUR_CONFIG_FILE variable' % config_file_name)
    else:
        config_file_name = resource_filename(__package__, 'calour.config')
    return config_file_name


def set_config_value(key, value, section='DEFAULT', config_file_name=None):
    '''Set the value in the calour config file

    Parameters
    ----------
    key : str
        the key to get the value for
    value : str
        the value to store
    section : str, optional
        the section to get the value from
    config_file_name : str, optional
        the full path to the config file or None to use default config file
    '''
    if config_file_name is None:
        config_file_name = get_config_file()

    config = configparser.ConfigParser()
    config.read(config_file_name)
    if section not in config:
        config.add_section(section)
    config.set(section, key, value)
    with open(config_file_name, 'w') as config_file:
        config.write(config_file)
    logger.debug('wrote key %s value %s to config file' % (key, value))


def get_config_sections(config_file_name=None):
    '''Get a list of the sections in the config file

    Parameters
    ----------
     config_file_name : str, optional
        the full path to the config file or None to use default config file

    Returns
    -------
    list of str
        List of the sections in the config file
    '''
    if config_file_name is None:
        config_file_name = get_config_file()
    logger.debug('getting sections from config file %s' % config_file_name)
    config = configparser.ConfigParser()
    config.read(config_file_name)
    return config.sections()


def get_config_value(key, fallback=None, section='DEFAULT', config_file_name=None):
    '''Get the value from the calour config file

    Parameters
    ----------
    key : str
        the key to get the value for
    fallback : str, optional
        the fallback value if the key/section/file does not exist
    section : str, optional
        the section to get the value from
    config_file_name : str, optional
        the full path to the config file or None to use default config file

    Returns
    -------
    value : str
        value of the key or fallback if file/section/key does not exist
    '''
    if config_file_name is None:
        config_file_name = get_config_file()

    config = configparser.ConfigParser()
    config.read(config_file_name)

    if section not in config:
        logger.debug('section %s not in config file %s' % (section, config_file_name))
        return fallback

    if key not in config[section]:
        logger.debug('key %s not in config file %s section %s' % (key, config_file_name, section))
        return fallback

    value = config[section][key]
    return value


def set_log_level(level):
    '''Set the debug level for calour

    You can see the logging levels at:
    https://docs.python.org/3.5/library/logging.html#levels

    Parameters
    ----------
    level : int or str
        10 for debug, 20 for info, 30 for warn, etc.
        It is passing to :func:`logging.Logger.setLevel`

    '''
    clog = getLogger('calour')
    clog.setLevel(level)


def _to_list(x):
    '''if x is non iterable or string, convert to iterable.

    See the expected behavior in the examples below.

    Examples
    --------
    >>> _to_list('a')
    ['a']
    >>> _to_list({})
    [{}]
    >>> _to_list(['a'])
    ['a']
    >>> _to_list(set(['a']))
    [{'a'}]
    '''
    if isinstance(x, str):
        return [x]
    if isinstance(x, Sequence):
        return x
    return [x]


def _argsort(values, reverse=False):
    '''Sort a sequence of values of heterogeneous variable types.

    This is useful to overcome the problem when using numpy.argsort on a pandas
    series values with missing values or different data types.

    Examples
    --------
    >>> l = [10, 'b', np.nan, 2.5, 'a']
    >>> idx = _argsort(l)
    >>> idx
    [3, 0, 2, 4, 1]
    >>> l_sorted = [l[i] for i in idx]
    >>> l_sorted
    [2.5, 10, nan, 'a', 'b']
    >>> l_sorted_reverse = [l[i] for i in _argsort(l, True)]
    >>> l_sorted_reverse
    ['b', 'a', nan, 10, 2.5]

    Parameters
    ----------
    values : iterable
        the values to sort

    Returns
    -------
    list of ints
        the positions of the sorted values

    '''
    pairs = []
    for cval in values:
        if isinstance(cval, Real):
            if np.isnan(cval):
                cval = np.inf
            else:
                cval = float(cval)
        pairs.append((str(type(cval)), cval))

    # # convert all numbers to float otherwise int will be sorted different place
    # values = [float(x) if isinstance(x, Real) else x for x in values]
    # # make values ordered by type and sort inside each var type
    # values = [(str(type(x)), x) if not np.isnan(x) else (str(type(x)), np.inf) for x in values]
    # return sorted(range(len(values)), key=values.__getitem__)
    return sorted(range(len(pairs)), key=pairs.__getitem__, reverse=reverse)


def _clone_function(f):
    '''Make a copy of a function'''
    # based on http://stackoverflow.com/a/13503277/2289509
    new_f = FunctionType(f.__code__, f.__globals__,
                         name=f.__name__,
                         argdefs=f.__defaults__,
                         closure=f.__closure__)
    new_f = update_wrapper(new_f, f)
    new_f.__kwdefaults__ = f.__kwdefaults__
    return new_f


def register_functions(clss, modules=None):
    '''Search and modify functions in the modules.

    This searches all the functions defined in the given
    ``modules`` and modify functions as following:

    1. for each public function with ``axis`` parameter, decorate it with
       :func:`._convert_axis_name` to convert "s" and "f" to 0 or
       1 for the ``axis`` parameter.

    2. for each public function that accepts its 1st argument of type
       defined in ``clss``, register it as a class method to the class
       type of its 1st argument.

    3. for each public function that accepts its 1st argument of type
       defined in ``clss`` **and** returns value of the same type,
       also decorate it with :meth:`.Experiment._record_sig`.

    Parameters
    ----------
    clss : tuple of ``class`` objects
        The class that functions will .
    modules : iterable of str, default=None
        The module names where functions are defined. ``None`` means all public
        modules in `calour`.

    '''

    # pattern to recognize the Parameters section
    p = re.compile(r"(\n +Parameters\n +-+ *)")

    if modules is None:
        modules = ['calour.' + i for i in
                   ['io', 'sorting', 'filtering', 'analysis', 'training', 'transforming',
                    'heatmap.heatmap', 'plotting', 'manipulation', 'database', 'export_html']]

    for module_name in modules:
        module = import_module(module_name)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fn, f in functions:
            sig = inspect.signature(f)
            params = sig.parameters

            # ski private function
            if fn.startswith('_'):
                continue

            if 'axis' in params.keys():
                f = _convert_axis_name(f)

            for _, param in params.items():
                cls = param.annotation
                if cls in clss:
                    # make a copy of the function because we want
                    # to update the docstring of the original
                    # function but not that of the registered
                    # version
                    if hasattr(cls, fn):
                        # python can't distinguish defined and
                        # imported functions. If a function is defined
                        # in a module and imported in another, without
                        # this check, it will get processed twice.
                        continue
                    if sig.return_annotation is cls:
                        setattr(cls, fn, cls._record_sig(_clone_function(f)))
                    else:
                        setattr(cls, fn, _clone_function(f))
                    updated = ('\n    .. note:: This function is also available as a class method :meth:`.{0}.{1}`\n'
                               '\\1'
                               '\n    exp : {0}'
                               '\n        Input {0} object.'
                               '\n')

                    # use `or` in case f.__doc__ is None
                    f.__doc__ = p.sub(updated.format(cls.__name__, fn), f.__doc__ or '')
                # only check the first func parameter
                break


def deprecated(message):
    '''Deprecation decorator.

    Parameters
    ----------
    message : str
        the message to print together with deprecation warning.
    '''
    def deprecated_decorator(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator


def format_docstring(*args, **kwargs):
    '''Format the docstring of the decorated function.'''
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    return dec
