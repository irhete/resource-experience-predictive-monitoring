import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "data_with_exp"

#### Traffic fines settings ####

ds = "traffic_fines_1"
for suffix in ["", "_exp"]:
    dataset = ds + suffix

    filename[dataset] = os.path.join(logs_dir, "traffic_fines_1.csv")

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
    static_cat_cols[dataset] = ["article", "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]
        
        
#### BPIC2017 settings ####

bpic2017_dict = {"bpic2017_cancelled": "BPIC17_O_Cancelled.csv",
                 "bpic2017_accepted": "BPIC17_O_Accepted.csv",
                 "bpic2017_refused": "BPIC17_O_Refused.csv"
                }

for ds, fname in bpic2017_dict.items():
    for suffix in ["", "_exp"]:
        dataset = ds + suffix

        filename[dataset] = os.path.join(logs_dir, fname)

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = 'org:resource'
        timestamp_col[dataset] = 'time:timestamp'
        label_col[dataset] = "label"
        neg_label[dataset] = "regular"
        pos_label[dataset] = "deviant"

        # features for classifier
        dynamic_cat_cols[dataset] = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition',
                                    "Accepted", "Selected"] 
        static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
        dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',  "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
        static_num_cols[dataset] = ['RequestedAmount']
    
            
#### BPIC2012 settings ####
bpic2012_dict = {"bpic2012_cancelled": "bpic2012_O_CANCELLED-COMPLETE.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv"
                }

for ds, fname in bpic2012_dict.items():
    for suffix in ["", "_exp"]:
        dataset = ds + suffix

        filename[dataset] = os.path.join(logs_dir, fname)

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "Resource"
        timestamp_col[dataset] = "Complete Timestamp"
        label_col[dataset] = "label"
        neg_label[dataset] = "regular"
        pos_label[dataset] = "deviant"

        # features for classifier
        dynamic_cat_cols[dataset] = ["Activity", "Resource"]
        static_cat_cols[dataset] = []
        dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
        static_num_cols[dataset] = ['AMOUNT_REQ']

#### BPIC2015 settings ####
for municipality in range(1,6):
    for formula in range(2,3):
        for suffix in ["", "_exp"]:
            dataset = "bpic2015_%s%s"%(municipality, suffix)
        
            filename[dataset] = os.path.join(logs_dir, "BPIC15_%s_f2.csv"%(municipality))

            case_id_col[dataset] = "Case ID"
            activity_col[dataset] = "Activity"
            resource_col[dataset] = "org:resource"
            timestamp_col[dataset] = "time:timestamp"
            label_col[dataset] = "label"
            pos_label[dataset] = "deviant"
            neg_label[dataset] = "regular"

            # features for classifier
            dynamic_cat_cols[dataset] = ["Activity", "monitoringResource", "question", "org:resource"]
            static_cat_cols[dataset] = ["Responsible_actor"]
            dynamic_num_cols[dataset] = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
            static_num_cols[dataset] = ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw', 'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming', 'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap', 'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)', 'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']

            if municipality in [3,5]:
                static_num_cols[dataset].append('Flora en Fauna')
            if municipality in [1,2,3,5]:
                static_num_cols[dataset].append('Brandveilig gebruik (melding)')
                static_num_cols[dataset].append('Milieu (melding)')

# add resource experience columns
for dataset in dynamic_num_cols.keys():
    if "_exp" in dataset:
        dynamic_num_cols[dataset] += ['n_tasks',
                                      'n_tasks_recent', 'n_cases', 'n_cases_recent', 'n_acts',
                                      'n_acts_recent', 'n_handoffs', 'n_handoffs_recent', 'ent_act',
                                      'ent_act_recent', 'ent_case', 'ent_case_recent', 'ent_handoff',
                                      'ent_handoff_recent', 'ratio_act_case',
                                      'ratio_act_case_recent', 'n_current_case', 'n_current_case_recent',
                                      'n_current_act', 'n_current_act_recent', 'n_current_handoff',
                                      'n_current_handoff_recent', 'ratio_current_case',
                                      'ratio_current_case_recent', 'ratio_current_act',
                                      'ratio_current_act_recent', 'ratio_current_handoff',
                                      'ratio_current_handoff_recent', 'busyness', 'busyness_recent']
        # comment out polarity columns if necessary
        dynamic_num_cols[dataset] += ['polarity_case', 'polarity_case_recent',
                                      'polarity_tasks', 'polarity_tasks_recent', 'polarity_current_act',
                                      'polarity_current_act_recent', 'polarity_current_handoff',
                                      'polarity_current_handoff_recent']
        # if commented out, resource name column is used
        #dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != resource_col[dataset]]