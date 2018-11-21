import scipy.sparse
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from time import time
import pickle
import os
from sys import argv
import itertools
import xgboost as xgb

import EncoderFactory
import ClassifierFactory
import BucketFactory
from DatasetManager import DatasetManager

dataset_name = argv[1]
cls_method = argv[2]
results_dir = argv[3]
optimal_params_filename = argv[4]

bucket_encoding = "agg"
bucket_method = "single"
cls_encoding = "agg"

def calculate_activity_freqs(gr, normalize=False):
    gr[act_freq_cols] = gr[act_freq_cols].cumsum()
    if normalize:
        gr[act_freq_cols] = (gr[act_freq_cols].T / gr[act_freq_cols].sum(axis=1)).T
    return(gr)

method_name = "%s_%s"%(bucket_method, cls_encoding)

if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))
    os.makedirs(os.path.join(results_dir, "final_results"))
    os.makedirs(os.path.join(results_dir, "feature_importances"))
    os.makedirs(os.path.join(results_dir, "detailed_results"))
    
if optimal_params_filename != "None":
    with open(os.path.join(optimal_params_filename), "rb") as fin:
        best_params = pickle.load(fin)
else:
    best_params = {}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}
    
train_ratio = 0.8
val_ratio = 0.2 # this is used for XGBoost early stopping
random_state = 22
fillna = True

methods = encoding_dict[cls_encoding]

outfile = os.path.join(results_dir, "final_results/%s_%s_%s.csv"%(cls_method, method_name, dataset_name)) 
detailed_results_file = os.path.join(results_dir, "detailed_results/detailed_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_name)) 
feature_importance_file = os.path.join(results_dir, "feature_importances/%s_%s_%s.csv"%(cls_method, method_name, dataset_name)) 
    
    
##### MAIN PART ######    
detailed_results = pd.DataFrame()
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score"))
    
    dataset_name2 = dataset_name[:]
    dataset_name = dataset_name.replace("_act_freqs_norm", "").replace("_act_freqs", "").replace("_no_res", "")
    dataset_manager = DatasetManager(dataset_name)

    # read the data
    data = dataset_manager.read_dataset().sort_values(dataset_manager.timestamp_col, ascending=True, kind="mergesort")

    if "act_freqs_norm" in dataset_name2:
        data = pd.concat([data, pd.get_dummies(data[dataset_manager.activity_col], prefix="act_freq")], axis=1)
        act_freq_cols = [col for col in data.columns if col.startswith("act_freq")]
        data = data.groupby(dataset_manager.resource_col).apply(calculate_activity_freqs, normalize=True)

    elif "act_freqs" in dataset_name2:
        data = pd.concat([data, pd.get_dummies(data[dataset_manager.activity_col], prefix="act_freq")], axis=1)
        act_freq_cols = [col for col in data.columns if col.startswith("act_freq")]
        data = data.groupby(dataset_manager.resource_col).apply(calculate_activity_freqs)

    if "no_res" in dataset_name2:
        dataset_manager.dynamic_cat_cols = [col for col in dataset_manager.dynamic_cat_cols if col != dataset_manager.resource_col]

    # split data into train and test
    train, test = dataset_manager.split_data_strict(data, train_ratio)
    if cls_method != "rf":
        train, val = dataset_manager.split_val(train, val_ratio)

    # consider prefix lengths until 90% of positive cases have finished
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    del data

    # create prefix logs
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

    # extract a val set for XGBoost early stopping 
    if cls_method != "rf":
        dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)

    print(dt_train_prefixes.shape)
    print(dt_test_prefixes.shape)

    print(dt_test_prefixes[dataset_manager.resource_col].value_counts())

    # extract arguments
    bucketer_args = {'encoding_method':bucket_encoding, 
                     'case_id_col':dataset_manager.case_id_col, 
                     'cat_cols':[dataset_manager.activity_col], 
                     'num_cols':[], 
                     'n_clusters':None, 
                     'random_state':random_state}

    cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                        'static_cat_cols':dataset_manager.static_cat_cols,
                        'static_num_cols':dataset_manager.static_num_cols, 
                        'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                        'fillna':fillna}


    # Bucketing prefixes based on control flow -- doesn't do anything if we use bucket_method == "single"
    print("Bucketing prefixes...")
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    bucket_assignments_test = bucketer.fit_predict(dt_test_prefixes)

    feature_combiners = {}
    classifiers = {}

    # train and fit pipeline for each bucket
    for bucket in set(bucket_assignments_train):
        print("Fitting pipeline for bucket %s..."%bucket)

        # set optimal params for this bucket
        if dataset_name in best_params:
            cls_args = best_params[dataset_name]
        elif cls_method == "rf":
            cls_args = {"n_estimators": 500, "max_features": 0.25}
        else:
            cls_args =  {'n_estimators': 1000,
                         'learning_rate': 0.02,
                         'subsample': 0.5,
                         'max_depth': 8,
                         'colsample_bytree': 0.48,
                         'min_child_weight': 2}


        # select relevant cases
        relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket)
        y_train = dataset_manager.get_label_numeric(dt_train_bucket)
        if len(set(y_train)) < 2:
            break

        encoders = [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods]
        if "act_freqs" in dataset_name2:
            cls_encoder_args2 = {'case_id_col':dataset_manager.case_id_col, 
                        'static_cat_cols':[],
                        'static_num_cols':[], 
                        'dynamic_cat_cols':[],
                        'dynamic_num_cols':act_freq_cols, 
                        'fillna':fillna}
            encoders.append(("act_freqs_last", EncoderFactory.get_encoder("last", **cls_encoder_args2)))
        feature_combiners[bucket] = FeatureUnion(encoders)

        X_train = feature_combiners[bucket].fit_transform(dt_train_bucket)

        # for XGBoost we use early stopping -- can be changed
        if cls_method == "rf":
            cls_args['random_state'] = random_state
            classifiers[bucket] = RandomForestClassifier(**cls_args)
            classifiers[bucket].fit(X_train, y_train)
        else:
            bucket_assignments_val = bucketer.fit_predict(dt_val_prefixes)
            relevant_cases_bucket = dataset_manager.get_indexes(dt_val_prefixes)[bucket_assignments_val == bucket]
            dt_val_bucket = dataset_manager.get_relevant_data_by_indexes(dt_val_prefixes, relevant_cases_bucket)
            y_val = dataset_manager.get_label_numeric(dt_val_bucket)
            if len(set(y_val)) < 2:
                break
            X_val = feature_combiners[bucket].fit_transform(dt_val_bucket)
            classifiers[bucket] = xgb.XGBClassifier(objective='binary:logistic', **cls_args)
            eval_set = [(X_val, y_val)]
            classifiers[bucket].fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)

        # write feature importances to file
        columns = []
        for encoder in feature_combiners[bucket].transformer_list:
            columns.extend(list(encoder[1].columns))
        importances = pd.DataFrame({"importance":classifiers[bucket].feature_importances_, "feature":columns})
        importances.to_csv(feature_importance_file, sep=";", index=False)


    prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

    # test separately for each prefix length
    for nr_events in range(min_prefix_length, max_prefix_length+1):
        print("Predicting for %s events..."%nr_events)

        # select only cases that are at least of length nr_events
        relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

        if len(relevant_cases_nr_events) == 0:
            break

        dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
        del relevant_cases_nr_events

        start = time()
        # get predicted cluster for each test case
        bucket_assignments_test = bucketer.predict(dt_test_nr_events)

        # use appropriate classifier for each bucket of test cases
        # for evaluation, collect predictions from different buckets together
        preds = []
        test_y = []
        for bucket in set(bucket_assignments_test):
            relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events, relevant_cases_bucket) # one row per event

            if len(relevant_cases_bucket) == 0:
                continue

            elif bucket not in classifiers:
                # use the general class ratio (in training set) as prediction 
                preds_bucket = [dataset_manager.get_class_ratio(train)] * len(relevant_cases_bucket)

            else:
                # make actual predictions
                X_test = feature_combiners[bucket].transform(dt_test_bucket)
                preds_pos_label_idx = np.where(classifiers[bucket].classes_ == 1)[0][0]
                preds_bucket = classifiers[bucket].predict_proba(X_test)[:,preds_pos_label_idx]

            preds.extend(preds_bucket)

            # extract actual label values
            test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
            test_y.extend(test_y_bucket)

            case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
            resources = list(dt_test_bucket.groupby(dataset_manager.case_id_col).last()[dataset_manager.resource_col])
            current_results = pd.DataFrame({"dataset": dataset_name, "cls": cls_method, "params": method_name, "nr_events": nr_events, "predicted": preds_bucket, "actual": test_y_bucket, "case_id": case_ids, "resource": resources})
            detailed_results = pd.concat([detailed_results, current_results], axis=0)

        if len(set(test_y)) < 2:
            auc = None
        else:
            auc = roc_auc_score(test_y, preds)
        prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

        fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "auc", auc))
        fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "precision", prec))
        fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "recall", rec))
        fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "fscore", fscore))

    print("\n")

detailed_results.to_csv(detailed_results_file, sep=";", index=False)
