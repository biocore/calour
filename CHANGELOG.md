# calour changelog

## Next version

* Add `read_qiime2()` function to enable reading of qiime2 feature tables artifacts with the associated rep-seqs and taxonomy artifacts.

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
