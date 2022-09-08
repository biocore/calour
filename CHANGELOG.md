# calour changelog
## Version 2022.9.8

## Version 2023.4.22

Bug Fixes:
* update the Experiment.plot() function so works also with matplotlib > 3.7

## Version 2022.9.8

Other changes:
* Add type hints to function return values
* Create Experiment.info dict and populate with md5s if not provided when creating a new experiment

Other changes:
* Add type hints to function return values

## Version 2022.7.1
Incompatible changes:
* Change default join_metadata_fields() inplace parameter to False
* In plot_diff_abundance_enrichment(), plot_enrichment(), Replaced enriched_exp_color parameter with labels_kwargs, numbers_kwargs, to enable better control of the barplot labels

Bug Fixes:
* Fix join_metadata_fields() to use axis='s' by default
* Fix join_experiments() to update exp.info['sample_metadata_md5'] and exp.info['data_md5']
* Fix join_experiments() to make field parameter optional, enable joining when field already exists in the experiment, and update the doc
* Fix join_experiments_featurewise() to make field parameter optional, enable joining when field already exists in the experiment, and update the doc
* Fix join_metadata() to use axis='s' by default
* Remove normalization check from filter_by_data()
* Fix heatmap() to copy the colormap to avoid matplotlib depracation warning (modifying the state of a globally registered colormap)
* Fix experiment read functions to show by default only the summary and first 5 samples without data/without metadata

## Version 2020.8.6

Incompatible changes:
* Change random_seed to work with numpy.random.default_rng. This may cause different random numbers compared to the old versions using numpy.random.seed().
* Change parameter names in some functions
* Rename filter_abundance() to filter_sum(abundance)
* Other backwards incompatible function API changes and code refactoring.

New features:
* Add RatioExperiment for working with ratios between two groups of features
* Add random_seed option to tranforming.permute_data()
* Add bad_color parameter to heatmap() and derivative functions
* Add more methods for MS1Experiment
* Add q-values (correted p-values) to dsfdr and derivative functions. This is manifested in a new feature_metadata field ("qval") for results of diff_abundance() / correlation()
* improved GUI for qt5 heatmap database enrichment results.
* Add `read_qiime2()` function to enable reading of qiime2 feature tables artifacts with the associated representative sequences and taxonomy artifacts (without the need to install qiime2)
* Add `Experiment.validate()`.
* Change default color scale in heatmap plot to linear scale for `Experiment` and log scale for `AmpliconExperiment` and `MS1Experiment`.
* Move to pytest for unit tests and doctests.
* Add new mechanism to register a function to a class as a method automatically. In order for a function to be registerred to a class, it must be a public function and has type hint of the class type for its first function parameter and return value.
* Clean and improve API documentation.

Other changes:
* make scikit-bio an optional dependency (needed only when processing dendrograms)

## Version 2019.5.1

* Add `reverse` parameter to the sorting functions.
* Fix minor documentation formatting issues
* Update installation instruction with conda install from conda-forge
* Change the column names added to `Experiment.feature_metadata` after running `Experiment.correlation` or `Experiment.diff_abundance`

## Version 2018.10.1

* Add notebook tutorial for `calour.training` module for classification and regression.
* Add notebook tutorial for metabolome data analysis
* Add plot functions in `calour.training` module
* Fix a bug in `Expriment.aggregate_by_metadata` when the number in the data table is overflow the int type.
* Add CONTRIBUTING.md as guidelines


## Version 2018.5.2

* Add export_html() function to save heatmap as an interactive html heatmap


## Version 2018.5.1

* In `calour.training` module, added functions to do regression and classification and to visualize their results. `SortedStratifiedKFold` and `RepeatedSortedStratifiedKFold` are aslo added for stratified cross validation for regression. They are recommended.
