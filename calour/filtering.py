# calour functions for filtering samples/observations
# functions should call reorder_samples() / reorder_obs()


def filter_samples(exp, field, values, exclude=False, inplace=False):
	'''
	filter samples from the study keeping only samples where the value in field
	is in values
	if exclude is true - remove the matching samples instead of filtering them
	'''

	# should call reorder_samples()


def filter_orig_reads(exp, minreads=10000, inplace=False):
	'''
	filter samples that have less than minreads total reads
	'''


def filter_taxonomy(exp, taxonomy, exact=False, exclude=False, inplace=False):
	'''
	filter keeping only observations with taxonomy string matching taxonomy
	if exact=True, look for partial match instead of identity
	'''


def filter_obs_min_reads(exp, minreads=10, exclude=False, inplace=False):
	'''
	keep only observations with at least minreads total over all samples
	'''


def filter_obs_mean_reads(exp, meanreads=0.001, exclude=False, inplace=False):
	'''
	keep only observations with at least minreads mean in all samples
	'''


def filter_obs_presence(exp, frac_pres=0.5, threshold=0, exclude=False, inplace=False):
	'''
	keep only observations present (number reads > threshold) in at least frac_pres of samples
	'''


def filter_obs(exp, values, exclude=False, inplace=False):
	'''
	keep only observations with id (usually sequence) in values list
	'''


def filter_seqs_from_fasta(exp, filename, exclude=False, inplace=False):
	'''
	keep only observations who'se id (sequence) is in the fasta file filename
	'''
