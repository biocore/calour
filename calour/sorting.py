# calour functions for sorting samples/observations
# functions should call reorder_samples() / reorder_obs()


def sort_taxonomy(exp, inplace=False):
	'''
	sort the observations based on the taxonomy
	'''


def cluster_obs(exp, minreads=0, inplace=False):
	'''
	reorder observations using clustering
	first filter away observations with <minreads
	'''


def cluster_samples(exp, inplace=False):
	'''
	reorder samples by clustering similar samples
	'''


def sort_samples(exp, field, numeric=False, inplace=False):
	'''
	sort samples based on values in field
	(do we need numeric or is it automatic in pandas?)
	'''


def sort_freq(exp, field=None, values=None, exclude=False, logscale=True, reverse=False, inplace=False):
	'''
	sort observatios based on mean frequency in group (field,values)
	if field is None, sort based on mean frequency in all experiment
	by default, use mean of log2(samples)
	'''


def sort_obs_center_mass(exp,field=None, numeric=True, uselog=True,inplace=False):
	'''
	sort observations based on center of mass after sorting samples by field (or None not to pre sort)
	'''


def sort_seqs_first(exp, seqs, inplace=False):
	'''
	reorder observations by first putting the observations in seqs and then the others
	'''


def reverse_obs(exp, inplace=False):
	'''
	reverse the order of the observations
	'''


def sort_samples_by_seqs(exp, seqs, reverse=False, inplace=False):
	'''
	sort the samples based on the frequencies of sequences in seqs
	'''
