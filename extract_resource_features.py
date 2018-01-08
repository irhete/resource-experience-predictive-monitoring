import numpy as np
import time
import pandas as pd
from scipy import stats
from sys import argv
import dataset_confs
import os

def ent(data, col):
    p_data = 1.0 * data[col].value_counts() / len(data) # calculates the probabilities
    entropy = stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

def get_prev_resource_and_event_nr(group):
    group[handoff_col] = group[resource_col].shift(1)
    group["event_nr"] = range(1, len(group)+1)
    return(group)

def extract_experience(gr):
    group = gr.copy()
    group["n_tasks"] = 0
    group["n_tasks_recent"] = 0
    group["n_cases"] = 0
    group["n_cases_recent"] = 0
    group["n_acts"] = 0
    group["n_acts_recent"] = 0
    group["n_handoffs"] = 0
    group["n_handoffs_recent"] = 0
    group["ent_act"] = 0
    group["ent_act_recent"] = 0
    group["ent_case"] = 0
    group["ent_case_recent"] = 0
    group["ent_handoff"] = 0
    group["ent_handoff_recent"] = 0
    group["polarity_case"] = 0
    group["polarity_case_recent"] = 0
    group["polarity_tasks"] = 0
    group["polarity_tasks_recent"] = 0
    group["ratio_act_case"] = 0
    group["ratio_act_case_recent"] = 0
    
    group["n_current_case"] = 0
    group["n_current_case_recent"] = 0
    group["n_current_act"] = 0
    group["n_current_act_recent"] = 0
    group["n_current_handoff"] = 0
    group["n_current_handoff_recent"] = 0
    group["ratio_current_case"] = 0
    group["ratio_current_case_recent"] = 0
    group["ratio_current_act"] = 0
    group["ratio_current_act_recent"] = 0
    group["ratio_current_handoff"] = 0
    group["ratio_current_handoff_recent"] = 0
    group["polarity_current_act"] = 0
    group["polarity_current_act_recent"] = 0
    group["polarity_current_handoff"] = 0
    group["polarity_current_handoff_recent"] = 0
    
    start_idx = 0
    start_time = group.iloc[0][timestamp_col]
    idx = 0
    
    for _, row in group.iterrows():
        if row["event_nr"] > max_events:
            idx += 1
            continue
        
        while (row[timestamp_col] - start_time).days > recent_days:
            start_idx += 1
            start_time = group.iloc[start_idx][timestamp_col]
            
        all_prev_exp = group.iloc[:(idx+1)]
        recent_prev_exp = group.iloc[start_idx:(idx+1)]
        idx += 1
        
        n_tasks = len(all_prev_exp)
        n_tasks_recent = len(recent_prev_exp)
        n_cases = len(all_prev_exp[case_id_col].unique())
        n_cases_recent = len(recent_prev_exp[case_id_col].unique())
        n_acts = len(all_prev_exp[activity_col].unique())
        n_acts_recent = len(recent_prev_exp[activity_col].unique())
        n_handoffs = len(all_prev_exp[handoff_col].unique())
        n_handoffs_recent = len(recent_prev_exp[handoff_col].unique())
        ent_act = ent(all_prev_exp, activity_col)
        ent_act_recent = ent(recent_prev_exp, activity_col)
        ent_case = ent(all_prev_exp, case_id_col)
        ent_case_recent = ent(recent_prev_exp, case_id_col)
        ent_handoff = ent(all_prev_exp, handoff_col)
        ent_handoff_recent = ent(recent_prev_exp, handoff_col)
        
        tmp = all_prev_exp[all_prev_exp.event_nr==1][label_col].value_counts()
        polarity_case = (0 if pos_label not in tmp else tmp[pos_label]) / n_cases
        tmp = recent_prev_exp[recent_prev_exp.event_nr==1][label_col].value_counts()
        polarity_case_recent = (0 if pos_label not in tmp else tmp[pos_label]) / n_tasks_recent
        tmp = all_prev_exp[label_col].value_counts()
        polarity_tasks = (0 if pos_label not in tmp else tmp[pos_label]) / n_tasks
        tmp = recent_prev_exp[label_col].value_counts()
        polarity_tasks_recent = (0 if pos_label not in tmp else tmp[pos_label]) / n_tasks_recent
        
        ratio_act_case = n_tasks / n_cases
        ratio_act_case_recent = n_tasks_recent / n_cases_recent
        
        n_current_case = len(all_prev_exp[all_prev_exp[case_id_col] == row[case_id_col]])
        n_current_case_recent = len(recent_prev_exp[recent_prev_exp[case_id_col] == row[case_id_col]])
        n_current_act = len(all_prev_exp[all_prev_exp[activity_col] == row[activity_col]])
        n_current_act_recent = len(recent_prev_exp[recent_prev_exp[activity_col] == row[activity_col]])
        n_current_handoff = len(all_prev_exp[all_prev_exp[handoff_col] == row[handoff_col]])
        n_current_handoff_recent = len(recent_prev_exp[recent_prev_exp[handoff_col] == row[handoff_col]])
        ratio_current_case = n_current_case / n_tasks
        ratio_current_case_recent = n_current_case_recent / n_tasks_recent
        ratio_current_act = n_current_act / n_tasks
        ratio_current_act_recent = n_current_act_recent / n_tasks_recent
        ratio_current_handoff = n_current_handoff / n_tasks
        ratio_current_handoff_recent = n_current_handoff_recent / n_tasks_recent
        
        tmp = all_prev_exp[all_prev_exp[activity_col] == row[activity_col]].drop_duplicates(subset=[case_id_col])[label_col].value_counts()
        polarity_current_act = (0 if pos_label not in tmp else tmp[pos_label]) / n_current_act
        tmp = recent_prev_exp[recent_prev_exp[activity_col] == row[activity_col]].drop_duplicates(subset=[case_id_col])[label_col].value_counts()
        polarity_current_act_recent = (0 if pos_label not in tmp else tmp[pos_label]) / n_current_act_recent
        tmp = all_prev_exp[all_prev_exp[handoff_col] == row[handoff_col]].drop_duplicates(subset=[case_id_col])[label_col].value_counts()
        polarity_current_handoff = (0 if pos_label not in tmp else tmp[pos_label]) / n_current_handoff
        tmp = recent_prev_exp[recent_prev_exp[handoff_col] == row[handoff_col]].drop_duplicates(subset=[case_id_col])[label_col].value_counts()
        polarity_current_handoff_recent = (0 if pos_label not in tmp else tmp[pos_label]) / n_current_handoff_recent
        
        group = group.set_value(idx, "n_tasks", n_tasks)
        group = group.set_value(idx, "n_tasks_recent", n_tasks_recent)
        group = group.set_value(idx, "n_cases", n_cases)
        group = group.set_value(idx, "n_cases_recent", n_cases_recent)
        group = group.set_value(idx, "n_acts", n_acts)
        group = group.set_value(idx, "n_acts_recent", n_acts_recent)
        group = group.set_value(idx, "n_handoffs", n_acts)
        group = group.set_value(idx, "n_handoffs_recent", n_acts_recent)
        group = group.set_value(idx, "ent_act", ent_act)
        group = group.set_value(idx, "ent_act_recent", ent_act_recent)
        group = group.set_value(idx, "ent_case", ent_case)
        group = group.set_value(idx, "ent_case_recent", ent_case_recent)
        group = group.set_value(idx, "ent_handoff", ent_handoff)
        group = group.set_value(idx, "ent_handoff_recent", ent_handoff_recent)
        group = group.set_value(idx, "polarity_case", polarity_case)
        group = group.set_value(idx, "polarity_case_recent", polarity_case_recent)
        group = group.set_value(idx, "polarity_tasks", polarity_tasks)
        group = group.set_value(idx, "polarity_tasks_recent", polarity_tasks_recent)
        group = group.set_value(idx, "ratio_act_case", ratio_act_case)
        group = group.set_value(idx, "ratio_act_case_recent", ratio_act_case_recent)
        
        group = group.set_value(idx, "n_current_case", n_current_case)
        group = group.set_value(idx, "n_current_case_recent", n_current_case_recent)
        group = group.set_value(idx, "n_current_act", n_current_act)
        group = group.set_value(idx, "n_current_act_recent", n_current_act_recent)
        group = group.set_value(idx, "n_current_handoff", n_current_handoff)
        group = group.set_value(idx, "n_current_handoff_recent", n_current_handoff_recent)
        group = group.set_value(idx, "ratio_current_case", ratio_current_case)
        group = group.set_value(idx, "ratio_current_case_recent", ratio_current_case_recent)
        group = group.set_value(idx, "ratio_current_act", ratio_current_act)
        group = group.set_value(idx, "ratio_current_act_recent", ratio_current_act_recent)
        group = group.set_value(idx, "ratio_current_handoff", ratio_current_handoff)
        group = group.set_value(idx, "ratio_current_handoff_recent", ratio_current_handoff_recent)
        group = group.set_value(idx, "polarity_current_act", polarity_current_act)
        group = group.set_value(idx, "polarity_current_act_recent", polarity_current_act_recent)
        group = group.set_value(idx, "polarity_current_handoff", polarity_current_handoff)
        group = group.set_value(idx, "polarity_current_handoff_recent", polarity_current_handoff_recent)
    
    return group


dataset = argv[1]
output_dir = argv[2]

if "hospital_billing" in dataset:
    max_events = 10
elif "bpic2017" in dataset:
    max_events = 20
else:
    max_events = 40

timestamp_col = dataset_confs.timestamp_col[dataset]
case_id_col = dataset_confs.case_id_col[dataset]
activity_col = dataset_confs.activity_col[dataset]
handoff_col = "prev_resource"
resource_col = dataset_confs.resource_col[dataset]
label_col = dataset_confs.label_col[dataset]
pos_label = dataset_confs.pos_label[dataset]
filename = dataset_confs.filename[dataset]

recent_days = 90

data = pd.read_csv(filename, sep=";")
data[timestamp_col] = pd.to_datetime(data[timestamp_col])

data = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col).apply(get_prev_resource_and_event_nr)
data[handoff_col] = data[handoff_col].fillna("first")

start = time.time()
data = data.sort_values(timestamp_col,
                               ascending=True,
                               kind="mergesort").groupby(resource_col).apply(extract_experience)
print(time.time() - start)

data.to_csv(os.path.join(output_dir, os.path.basename(filename)), sep=";", index=False)
