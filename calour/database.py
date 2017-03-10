from logging import getLogger
from abc import ABC
import importlib

from .util import get_config_value, get_config_file, get_config_sections

logger = getLogger(__name__)


def _get_database_class(dbname, config_file_name=None):
    '''Get the database class for the given database name

    Uses the calour config file (calour.config) keys

    Parameters
    ----------
    dbname : str
        the database name. common options are:
            'dbbact' : the amplicon sequence manual annotation database
            'spongeworld' : the sponge microbiome database
            'redbiome' : the qiita automatic amplicon sequence database
    config_file_name: str or None (optional)
        None (default) to use the default calour condig file.
        str to use the file names str as the conig file


    Returns
    -------
    calour.database.Database
    A ``Database`` class for the requested dbname
    '''
    class_name = get_config_value('class_name', section=dbname, config_file_name=config_file_name)
    module_name = get_config_value('module_name', section=dbname, config_file_name=config_file_name)
    if class_name is not None and module_name is not None:
        # import the database module
        db_module = importlib.import_module(module_name)
        # get the class
        DBClass = getattr(db_module, class_name)
        cdb = DBClass()
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
        raise ValueError('calour config file %s does not contain any database sections.' % get_config_file())
    raise ValueError('Database %s not found in config file (%s).\n'
                     'Currently contains the databases: %s' % (dbname, get_config_file(), databases))


class Database(ABC):
    def __init__(self, database_name='generic', methods=['get', 'annotate', 'feature_terms']):
        '''Initialize the database interface

        Parameters
        ----------
        database_name : str (optional)
            name of the database
        methods : list of str (optional)
            'get' if database interface supports get_seq_annotation_strings()
            'annotate' if database interface supports add_annotation()
            'enrichment' if database interface supports get_feature_terms()
        '''
        self._database_name = database_name
        self._methods = set(methods)

    def get_name(self):
        '''Get the name of the database.
        Used for displaying when no annotations are found

        Returns
        -------
        dbname : str
            nice name of the database
        '''
        return self._database_name

    @property
    def annotatable(self):
        '''True if the database supports adding annotations via the add_annotation() function
        '''
        return 'annotate' in self._methods

    @property
    def can_get_feature_terms(self):
        '''True if the database supports getting a dict of terms per feature via the get_feature_terms() function
        '''
        return 'feature_terms' in self._methods

    def get_seq_annotation_strings(self, sequence):
        '''Get nice string summaries of annotations for a given sequence

        Parameters
        ----------
        sequence : str
            the DNA sequence to query the annotation strings about

        Returns
        -------
        shortdesc : list of (dict,str) (annotationdetails,annotationsummary)
            a list of:
                annotationdetails : dict
                    'annotationid' : int, the annotation id in the database
                    'annotationtype : str
                    ...
                annotationsummary : str
                    a short summary of the annotation
        '''
        logger.debug('Generic function for get_annotation_strings')
        return []

    def show_annotation_info(self, annotation):
        '''Show details about the annotation

        Parameters
        ----------
        annotation : dict
            keys/values are database specific.
            E.g. See dbBact REST API /annotations/get_annotation for keys / values
        '''
        # open in a new tab, if possible
        logger.debug('Generic function for show annotation info')
        return

    def add_annotation(self, features, exp):
        '''Add an entry to the database about a set of features

        Parameters
        ----------
        features : list of str
            the features to add to the database
        exp : calour.Experiment
            the experiment where the features are coming from

        Returns
        -------
        err : str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for add_annotations')
        raise NotImplementedError

    def upadte_annotation(self, annotation, exp=None):
        '''Update an existing annotation

        Parameters
        ----------
        annotation : dict
            The annotation to update (keys/values are database specific)
        exp : ``Experiment`` (optional)
            The calour experiment from which the annotation is coming from
        Returns
        -------
        str
            empty if ok, otherwise the error encountered
        '''
        logger.debug('Generic function for update_annotation')
        raise NotImplementedError

    def delete_annotation(self, annotation_details):
        '''Delete an annotation from the database (if allowed)

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
        '''remove a feature from the annotation in the database (if allowed)

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
        exp : calour.Experiment (optional)
            not None to store results inthe exp (to save time for multiple queries)

        Returns
        -------
        feature_terms : dict of list of str/int
            key is the feature, list contains all terms associated with the feature
        '''
        logger.debug('Generic function for get_feature_terms')
        return {}
