import requests
from logging import getLogger


logger = getLogger(__name__)


class BactDB:
    def __init__(self):
        # Web address of the bact server
        self.dburl = 'http://amnonim.webfactional.com/scdb_main'

    def getseqannotations(self, sequence):
        '''Get the annotations for a sequence

        Parameters:
        -----------
        sequence : str (ACGT)

        Returns:
        --------
        curs : list of list of (curation dict,list of [Type,Value] of curation details)
        '''
        rdata = {}
        rdata['sequence'] = sequence
        res = requests.get(self.dburl + '/sequences/get_annotations', json=rdata)
        if res.status_code != 200:
            logger.warn('error getting annotations for sequence %s' % sequence)
            return []
        annotations = res.json()['annotations']
        logger.debug('Found %d annotations for sequence %s' % (len(annotations), sequence))
        return annotations

    def getannotationstrings(self, sequence):
        '''Get nice string summaries of annotations for a given sequence

        Parameters:
        -----------
        sequence : str (ACGT)
            the sequence to query the annotation strings about

        Returns:
        --------
        shortdesc : list of (dict,str) (annotationdetails,annotationsummary)
            a list of:
                annotationdetails : dict
                    'annotationid' : int, the annotation id in the database
                    'annotationtype : str
                    ...
                annotationsummary : str
                    a short summary of the annotation
        '''
        shortdesc = []
        annotations = self.getseqannotations(sequence)
        for cann in annotations:
            annotationdetails = cann
            cdesc = ''
            if cann['description']:
                cdesc += cann['description']+' ('
            if cann['annotationtype'] == 'diffexp':
                chigh = []
                clow = []
                call = []
                for cdet in cann['details']:
                    if cdet[0] == 'all':
                        call.append(cdet[1])
                        continue
                    if cdet[0] == 'low':
                        clow.append(cdet[1])
                        continue
                    if cdet[0] == 'high':
                        chigh.append(cdet[1])
                        continue
                cdesc += ' high in '
                for cval in chigh:
                    cdesc += cval+' '
                cdesc += ' compared to '
                for cval in clow:
                    cdesc += cval+' '
                cdesc += ' in '
                for cval in call:
                    cdesc += cval+' '
            elif cann['annotationtype'] == 'isa':
                cdesc += ' is a '
                for cdet in cann['details']:
                    cdesc += 'cdet,'
            elif cann['annotationtype'] == 'contamination':
                cdesc += 'contamination'
            else:
                cdesc += cann['annotationtype']+' '
                for cdet in cann['details']:
                    cdesc = cdesc + ' ' + cdet[1] + ','
            shortdesc.append((annotationdetails, cdesc))
        return shortdesc
