'''
database access functions (:mod:`calour.database`)
==================================================

.. currentmodule:: calour.database

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   add_terms_to_features
   enrichment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from abc import ABC
import importlib

from .util import get_config_value, get_config_file, get_config_sections
from .experiment import Experiment
from packaging import version

logger = getLogger(__name__)


def _get_database_class(dbname, exp=None, config_file_name=None):
    '''Get the database class for the given database name

    Uses the calour config file (calour.config) keys

    Parameters
    ----------
    dbname : str
        the database name. common options are:
            'dbbact' : the amplicon sequence manual annotation database
            'spongeworld' : the sponge microbiome database
            'redbiome' : the qiita automatic amplicon sequence database
        Names are listed in the calour.config file as section names
    config_file_name: str or None, optional
        None (default) to use the default calour condig file.
        str to use the file names str as the conig file


    Returns
    -------
    calour.database.Database
    A ``Database`` class for the requested dbname
    '''
    class_name = get_config_value('class_name', section=dbname, config_file_name=config_file_name)
    module_name = get_config_value('module_name', section=dbname, config_file_name=config_file_name)
    min_version = version.parse(get_config_value('min_version', section=dbname, config_file_name=config_file_name, fallback='0.0'))
    module_website = get_config_value('website', section=dbname, config_file_name=config_file_name, fallback='NA')

    if class_name is not None and module_name is not None:
        try:
            # import the database module
            db_module = importlib.import_module(module_name)
        except ImportError:
            module_installation = get_config_value('installation', section=dbname, config_file_name=config_file_name)
            logger.warning('Database interface %s not installed.\nSkipping.\n'
                           'You can install the database using:\n%s\n'
                           'For details see: %s' % (module_name, module_installation, module_website))
            return None
        # get the class
        DBClass = getattr(db_module, class_name)
        cdb = DBClass(exp)
        # test if database version is compatible
        if min_version > version.parse('0.0'):
            db_version = version.parse(str(cdb.version()))
            logger.debug('database: %s , version installed: %s , minimal version: %s' % (dbname, db_version, min_version))
            if db_version < min_version:
                logger.warning('Please update %s database module. Current version (%s) not supported (minimal version %s).\nFor details see %s' % (dbname, db_version.public, min_version.public, module_website))
        return cdb
    # not found, so print available database names
    databases = []
    sections = get_config_sections()
    for csection in sections:
        class_name = get_config_value('class_name', section=csection, config_file_name=config_file_name)
        module_name = get_config_value('class_name', section=csection, config_file_name=config_file_name)
        if class_name is not None and module_name is not None:
            databases.append(csection)
    if len(databases) == 0:
        logger.warning('calour config file %s does not contain any database sections. Skipping' % get_config_file())
        return None
    logger.warning('Database %s not found in config file (%s).\nSkipping.\n'
                   'Current databases in config file: %s' % (dbname, get_config_file(), databases))
    return None


def add_terms_to_features(exp: Experiment, dbname, use_term_list=None, field_name='common_term', term_type=None, ignore_exp=None, **kwargs):
    '''Add a field to the feature metadata, with most common term for each feature

    Create a new feature_metadata field, with the most common term (out of term_list) for each feature in experiment.
    It adds annotations in-place.

    Parameters
    ----------
    use_term_list : list of str or None, optional
        Use only terms appearing in this list
        None (default) to use all terms
    field_name : str, optional
        Name of feature_metadata field to store the annotatiosn.
    term_type : str or None, optional
        type of the annotation summary to get from the database (db specific)
        None to get default type
    ignore_exp : list of int or None, optional
        list of experiments to ignore when adding the terms
    **kwargs: database specific additional parameters (see database interface get_feature_terms() function for specific terms)

    Returns
    -------
    Experiment
        with feature_metadata field containing the most common database term for each feature
    '''
    logger.debug('Adding terms to features for database %s' % dbname)
    db = _get_database_class(dbname, exp)
    features = exp.feature_metadata.index.values
    logger.debug('found %d features' % len(features))

    # get the per feature term scores
    term_list = db.get_feature_terms(features, exp=exp, term_type=term_type, ignore_exp=ignore_exp, **kwargs)
    logger.debug('got %d terms from database' % len(term_list))

    # find the most enriched term (out of the list) for each feature
    feature_terms = []
    for cfeature in features:
        if cfeature not in term_list:
            feature_terms.append('na')
            continue
        if len(term_list[cfeature]) == 0:
            feature_terms.append('na')
            continue
        if use_term_list is None:
            bterm = max(term_list[cfeature], key=term_list[cfeature].get)
        else:
            bterm = 'other'
            bscore = 0
            for cterm in use_term_list:
                if cterm not in term_list[cfeature]:
                    continue
                cscore = term_list[cfeature][cterm]
                if cscore > bscore:
                    bscore = cscore
                    bterm = cterm
        feature_terms.append(bterm)
    exp.feature_metadata[field_name] = feature_terms
    return exp


def enrichment(exp: Experiment, features, dbname, *args, **kwargs):
    '''Get the list of enriched annotation terms in features compared to all other features in exp.

    Uses the database specific enrichment analysis method.

    Parameters
    ----------
    features : list of str
        The features to test for enrichment (compared to all other features in exp)
    dbname : str
        the database to use for the annotation terms and enrichment analysis
    *args : tuple
    **kwargs : dict
        Additional database specific parameters (see per-database module documentation for .enrichment() method)

    Returns
    -------
        pandas.DataFrame with  info about significantly enriched terms. The columns include:
            * 'feature' : the feature ID (str)
            * 'pval' : the p-value for the enrichment (float)
            * 'odif' : the effect size (float)
            * 'observed' : the number of observations of this term in group1 (int)
            * 'expected' : the expected (based on all features) number of observations of this term in group1 (float)
            * 'frac_group1' : fraction of total terms in group 1 which are the specific term (float)
            * 'frac_group2' : fraction of total terms in group 2 which are the specific term (float)
            * 'num_group1' : number of total terms in group 1 which are the specific term (float)
            * 'num_group2' : number of total terms in group 2 which are the specific term (float)
            * 'description' : the term (str)

        numpy.Array where rows are features (ordered like the dataframe), columns are terms, and value is score
            for term in feature

        pandas.DataFrame with info about the features used. columns:
            * 'group' : int, the group (1/2) to which the feature belongs
            * 'sequence': str
    '''
    db = _get_database_class(dbname, exp=exp)
    if not db.can_do_enrichment:
        raise ValueError('database %s does not support enrichment analysis' % dbname)
    return db.enrichment(exp, features, *args, **kwargs)


class Database(ABC):
    def __init__(self, exp=None, database_name=None, methods=['get', 'annotate', 'enrichment']):
        '''Initialize the database interface

        Parameters
        ----------
        exp : Experiment or None, optional
            The experiment link for the database (if needed)
        database_name : str, optional
            name of the database (for showing errors, etc.)
        methods : list of str, optional
            The integration level this database interface supports.
            'get' if database interface supports get_seq_annotation_strings()
            'annotate' if database interface supports add_annotation()
            'enrichment' if database interface supports get_feature_terms()
        '''
        self.database_name = database_name
        self._methods = set(methods)
        self._exp = exp

    @property
    def annotatable(self):
        '''True if the database supports adding annotations via the add_annotation() function
        '''
        return 'annotate' in self._methods

    @property
    def can_do_enrichment(self):
        '''True if the database supports getting a dict of terms per feature via the get_feature_terms() function
        '''
        return 'enrichment' in self._methods

    def get_seq_annotation_strings(self, feature):
        '''Get nice string summaries of annotations for a given sequence

        Parameters
        ----------
        feature : str
            the feature ID to query the database about

        Returns
        -------
        list of [(dict,str)] (annotationdetails,annotationsummary)
            a list of:
                annotationdetails : dict
                    'annotationid' : int, the annotation id in the database
                    'annotationtype : str
                    ...
                annotationsummary : str
                    a short user readble summary of the annotation.
                    This will be displayed for the user
        '''
        logger.debug('Generic function for get_annotation_strings')
        return []

    def get_annotation_website(self, annotation):
        '''Get the database website address of information about the annotation.
        Used for the Jupyter GUI when clicking on an annotation in the list
        (to open in a new browser tab)

        Parameters
        ----------
        annotation : dict
            keys/values are database specific (supplied by the database interface when calling get_annotation_strings() ).
            These keys/values can be used by the database interface here to determine which website address to return.

        Returns
        -------
        str or None
            The webaddress of the html page with details about the annotation,
            or None if not available
        '''
        logger.debug('Generic function for get_annotation_website')
        return None

    def show_annotation_info(self, annotation):
        '''Show details about the annotation.
        This should use a database interface created GUI to show more details about the annotation.
        Called from the qt5 heatmap GUI when double clicking on a database annotation.
        A common GUI can be a new browser window with details about the annotation.

        Parameters
        ----------
        annotation : dict
            keys/values are database specific (supplied by the database interface when calling get_annotation_strings() ).
            These keys/values can be used by the database interface here to determine which website address to return.
        '''
        logger.debug('Generic function for show annotation info')
        return

    def add_annotation(self, features, exp):
        '''Add an entry to the database about a set of features.
        This is an optional function for databases that support manual annotations (level L4).
        supporting this option is indicated by the "annotate" method in __init__()
        It is called from the qt5 heatmap GUI when pressing the "Annotate" button.
        All GUI should be supplied by the database interface.

        Parameters
        ----------
        features : list of str
            the features to add to the database
        exp : Experiment
            the experiment where the features are coming from

        Returns
        -------
        err : str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for add_annotations')
        raise NotImplementedError

    def update_annotation(self, annotation, exp=None):
        '''Update an existing annotation
        This is an optional function for databases that support manual annotations (level L4).
        supporting this option is indicated by the "annotate" method in __init__().
        It is called when right clicking on an annotation in the qt5 GUI and selecting "update".
        All GUI should be supplied by the database interface.

        Parameters
        ----------
        annotation : dict
            The annotation to update (keys/values are database specific)
        exp : Experiment, optional
            The calour experiment from which the annotation is coming from
        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for update_annotation')
        raise NotImplementedError

    def delete_annotation(self, annotation_details):
        '''Delete an annotation from the database (if allowed). All features associated with this annotation
        lose this annotation.
        This is an optional function for databases that support manual annotations (level L4).
        supporting this option is indicated by the "annotate" method in __init__()
        It is called when right clicking on an annotation in the qt5 GUI and selecting "delete".

        Parameters
        ----------
        annotation_details : dict
            The details about the annotation to delete (annotationdetails from get_seq_annotation_strings() )
            Should contain a unique identifier for the annotation (created/used by the database)

        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for delete_annotation')
        return 'Not implemented'

    def remove_feature_from_annotation(self, features, annotation_details):
        '''remove a feature from the annotation in the database (if allowed). If after the removal the annotation contains no features,
        it will be removed from the database. Otherwise, the annotation remains for the features not removed from it.
        This is an optional function for databases that support manual annotations (level L4).
        supporting this option is indicated by the "annotate" method in __init__()
        It is called when right clicking on an annotation in the qt5 GUI and selecting "remove feature".

        Parameters
        ----------
        features : list of str
            The feature ids to remove
        annotation_details : dict
            The details about the annotation to delete (annotationdetails from get_seq_annotation_strings() )
            Should contain a unique identifier for the annotation (created/used by the database)

        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for remove_features_from_annotation')
        return 'Not implemented'

    def get_feature_terms(self, features, exp=None):
        '''Get list of terms per feature

        Parameters
        ----------
        features : list of str
            the features to get the terms for
        exp : Experiment, optional
            not None to store results inthe exp (to save time for multiple queries)

        Returns
        -------
        feature_terms : dict of term scores associated with each feature.
            Key is the feature (str), and the value is a dict of score for each term (i.e. key is the term str, value is the score for this term in this feature)
        '''
        logger.debug('Generic function for get_feature_terms')
        return {}

    def enrichment(self, exp, features, *args, **kwargs):
        '''Get the list of enriched terms in features compared to all features in exp.
        This is an optional function for databases that support enrichment analysis (level L3).

        Parameters
        ----------
        exp : Experiment
            The experiment to compare the features to
        features : list of str
            The features (from exp) to test for enrichment
        *args : tuple
        **kwargs : dict
            Additional database specific parameters

        Returns
        -------
        pandas.DataFrame
            Its columns include:

            feature : str. the feature

            pval : float. the p-value for the enrichment

            odif : float. the effect size for the enrichment

            term : str. the enriched term
        '''
        logger.debug('Generic function for enrichment')
        return None

    def show_term_details(self, term, exp, features, *args, **kwargs):
        '''
        Show details about the specific term in the database and in what features it appears.
        This is an optional function, and is called when a user double clicks
        an enriched term in the qt5 enrichment analysis (for an integration level L3 database).
        It shows details why this term was denoted as enriched. This is a database specific implementation.

        Parameters
        ----------
        term : str
            The term to get the details for
        exp : Experiment
            The calour experiment for showing the term details in
        features: list of str
            The features in the experiment for which to show the term details

        Returns
        -------
        '''
        logger.debug('Generic function for term details')
        return None
