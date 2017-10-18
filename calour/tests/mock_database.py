# mock database for calour gui unit testing
# simulates a simple database for querying sequences

from logging import getLogger

from calour.database import Database


logger = getLogger(__name__)


class MockDatabase(Database):
    def __init__(self, exp=None):
        '''Initialize the database interface

        Parameters
        ----------
        database_name : str, optional
            name of the database
        methods : list of str, optional
            'get' if database interface supports get_seq_annotation_strings()
            'annotate' if database interface supports add_annotation()
            'enrichment' if database interface supports get_feature_terms()
        '''
        super().__init__(exp=exp, database_name='mock_db', methods=['get', 'annotate', 'enrichment'])
        self.dbinfo = {
                       'TACGTATGTCACAAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCCGTGGATTAAGCGTGTTGTGAAATGTAGACGCTCAACGTCTGAATCGCAGCGCGAACTGGTTCACTTGAGTATGCACAACGTAGGCGGAATTCGTCG': ['seq1', 'nice'],
                       'TACATAGGTCGCAAGCGTTATCCGGAATTATTGGGCGTAAAGCGTTCGTAGGCTGTTTATTAAGTCTGGAGTCAAATCCCAGGGCTCAACCCTGGCTCGCTTTGGATACTGGTAAACTAGAGTTAGATAGAGGTAAGCAGAATTCCATGT': ['seq2'],
                       'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG': ['seq3', 'nice']
        }

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
        annotations = []
        if sequence not in self.dbinfo:
            return []
        for cstr in self.dbinfo[sequence]:
            cannotation = ({'annotationid': 1, 'annotationtype': 'other'}, cstr)
            annotations.append(cannotation)
        return annotations

    def get_feature_terms(self, features, exp=None):
        '''Get list of terms per feature

        Parameters
        ----------
        features : list of str
            the features to get the terms for
        exp : :class:`.Experiment`, optional
            not None to store results inthe exp (to save time for multiple queries)

        Returns
        -------
        feature_terms : dict of list of str/int
            key is the feature, list contains all terms associated with the feature
        '''
        feature_terms = {}
        for cseq, cterms in self.dbinfo.items():
            feature_terms[cseq] = []
            for cstr in cterms:
                feature_terms[cseq].append(cstr)
        return feature_terms
