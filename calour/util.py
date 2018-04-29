'''
utilities (:mod:`calour.util`)
==============================

.. currentmodule:: calour.util

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

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


def compute_prevalence(abundance):
    '''Return the prevalence at each abundance cutoffs.

    Each sample that has the OTU above the cutoff (exclusive) will
    be counted.

    Parameters
    ----------
    abundance : iterable of numeric
        The abundance of a species across samples.

    Examples
    --------
    >>> abund = [0, 0, 1, 2, 4]
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


def _transition_index(l):
    '''Return the transition index and current value of the list.

    Examples
    -------
    >>> l = ['a', 'a', 'b']
    >>> list(_transition_index(l))
    [(2, 'a'), (3, 'b')]
    >>> l = ['a', 'a', 'b', 1, 2, None, None]
    >>> list(_transition_index(l))
    [(2, 'a'), (3, 'b'), (4, 1), (5, 2), (7, None)]

    Parameters
    ----------
    l : Iterable of arbitrary objects

    Yields
    ------
    tuple of (int, arbitrary)
        the transition index, the item value
    '''
    it = enumerate(l)
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

    This allows the decorated function with ``axis`` parameter
    to accept "sample" and "feature" as value for ``axis`` parameter.

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

    Convert the taxonomy list stored (from biom.read_table) to a single string per feature

    Parameters
    ----------
    exp : Experiment
        with the taxonomy entry in the feature_metadata
    sep : str, optional
        the output separator to use between the taxonomic levels
    remove_underscore : bool, optional
        True (default) to remove the 'g__' entries and missing values
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


def _argsort(values):
    '''Sort a sequence of values of heterogeneous variable types.

    Used to overcome the problem when using numpy.argsort on a pandas
    series values with missing values

    Examples
    --------
    >>> l = [10, 'b', np.nan, 2.5, 'a']
    >>> idx = _argsort(l)
    >>> idx
    [3, 0, 2, 4, 1]
    >>> l_sorted = [l[i] for i in idx]
    >>> l_sorted
    [2.5, 10, nan, 'a', 'b']

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
    return sorted(range(len(pairs)), key=pairs.__getitem__)


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


def register_functions(cls, modules=None):
    '''Dynamically register functions to the class as methods.

    Parameters
    ----------
    cls : ``class`` object
        The class that the functions will be added to
    modules : iterable of str, optional
        The module names where the functions are defined. ``None`` means all public
        modules in `calour`.
    '''
    # pattern to recognize the Parameters section
    p = re.compile(r"(\n +Parameters\n +-+ *)")
    if modules is None:
        modules = ['calour.' + i for i in
                   ['io', 'sorting', 'filtering', 'analysis', 'training', 'transforming',
                    'heatmap.heatmap', 'plotting', 'manipulation', 'database']]
    for module_name in modules:
        module = import_module(module_name)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fn, f in functions:
            # skip private functions
            if not fn.startswith('_'):
                params = inspect.signature(f).parameters
                if params:
                    # if the func accepts parameters, ie params is not empty
                    first = next(iter(params.values()))
                    if first.annotation is cls:
                        # make a copy of the function because we want
                        # to update the docstring of the original
                        # function but not that of the registered
                        # version
                        setattr(cls, fn, _clone_function(f))
                        updated = ('\n    .. note:: This function is also available as a class method :meth:`.{0}.{1}`\n'
                                   '\\1'
                                   '\n    exp : {0}'
                                   '\n        Input experiment object.')

                        if not f.__doc__:
                            f.__doc__ = ''

                        f.__doc__ = p.sub(updated.format(cls.__name__, fn), f.__doc__)
