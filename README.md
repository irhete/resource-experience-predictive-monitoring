# Resource experience in predictive business process monitoring
This repository contains scripts for modeling resource experience as inter-case features for outcome-oriented predictive business process monitoring.

## Requirements   
The code is written in Python 3.6. Although not tested, it should work with any version of Python 3. The following Python libraries are required to run the code: 

* sklearn
* numpy
* pandas
* xgboost
* jupyter-notebook

## Usage
#### Data format
The scripts assume that each input dataset is a CSV file, each row representing one event in a trace, wherein each event is associated with at least the following attributes (as columns in the CSV file): the case id, activity type, resource id, timestamp, class label. As additional columns, any number of event and case attributes is accepted that are used to enhance the predictive power of the classifier. The relevant columns for each dataset should be specified in the script `dataset_confs.py`.

The input log is temporally split into data for training (80% of cases) and evaluating (20% of cases) the predictive models.

#### 1. Extracting features related to resource experience.
For each dataset, run:

`python extract_resource_features.py <dataset_name> <output_dir>`  

* _dataset_name_ - the name of the dataset, should correspond to the settings specified in `dataset_confs.py`.
* _output_dir_ - name of the output directory to write the datasets enriched with resource information.

#### 2. Hyperparameter optimization (optional)
The hyperparameters of random forest are tuned using cross-validation with grid search, i.e. for each dataset and method, different values of max_features are tested and the configuration that yields the highest AUC is chosen.

2.1. Testing different parameter configurations.

In order to launch cross-validation experiments, run:

`python optimize_params_rf.py <dataset_name> <results_dir>`  

* _dataset_name_ - the name of the dataset, should correspond to the settings specified in `dataset_confs.py`.
* _results_dir_ - name of the output directory to write the cross-validation results.

2.2. Selecting best parameters.

After the experiments launched via `random_search.py` have finished, the best parameters can be extracted using the following jupyter notebook: `extract_best_params.ipynb` 

#### 3. Training and evaluating the (final) models

After extracting the best parameters, the final models can be trained and applied using:

`python train_evaluate_final.py <dataset_name> <cls_method> <results_dir> <params_dir> <optimal_params_filename>` 

The arguments to the scripts are: 

* _dataset_name_ - the name of the dataset, should correspond to the settings specified in `dataset_confs.py`. Additional suffixes can be used, for instance:
1. traffic_fines_1 - baseline, uses only columns that are not related to resource experience.
2. traffic_fines_1_exp - uses (additionally) all columns related to resource experience.
3. traffic_fines_1_act_freqs - uses basic columns + frequencies of activities a resource has performed.
4. traffic_fines_1_act_freqs_norm - uses basic columns + normalized frequencies of activities a resource has performed.
5. *_no_res (e.g. traffic_fines_1_act_freqs_no_res) - excludes the resource id column.
* _cls_method_ - rf or xgb.
* _results_dir_ - the name of the directory where the predictions will be written.
* _optimal_params_filename_ - the name of the file where the optimal parameters have been saved. Use "None" if fixed parameters should be used (hard-coded in the script).

#### 4. Analyis and plotting of the final results

Open `resource_experience_analysis.R` in RStudio and run the script line by line.
