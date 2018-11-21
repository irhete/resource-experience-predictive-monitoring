library(ggplot2)

# XGBoost
setwd("/home/irene/Repos/resource-experience-predictive-monitoring/results_xgb/final_results/")
data <- data.frame()
files <- list.files()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  if (grepl("act_freqs_norm_no_res", file)) {
    tmp$exp <- "act_freqs_norm_no_res"
  } else if (grepl("act_freqs_norm", file)) {
    tmp$exp <- "act_freqs_norm"
  } else if (grepl("act_freqs", file)) {
    tmp$exp <- "act_freqs"
  } else if (grepl("exp", file)) {
    tmp$exp <- "all_exp"
  } else {
    tmp$exp <- "baseline"
  }
  data <- rbind(data, tmp)
}
data$score <- as.numeric(as.character(data$score))

# Random forest
setwd("/home/irene/Repos/resource-experience-predictive-monitoring/results_rf/final_results/")
files <- list.files()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  if (grepl("act_freqs_norm_no_res", file)) {
    tmp$exp <- "act_freqs_norm_no_res"
  } else if (grepl("act_freqs_norm", file)) {
    tmp$exp <- "act_freqs_norm"
  } else if (grepl("act_freqs", file)) {
    tmp$exp <- "act_freqs"
  } else if (grepl("exp", file)) {
    tmp$exp <- "all_exp"
  } else {
    tmp$exp <- "baseline"
  }
  data <- rbind(data, tmp)
}
setwd("../../results_rf_wo_polarity/final_results/")
files <- list.files()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  tmp$exp <- "wo_pol"
  data <- rbind(data, tmp)
}
setwd("../../results_rf_wo_polarity_wo_resource/final_results/")
files <- list.files()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  tmp$exp <- "wo_pol_res"
  data <- rbind(data, tmp)
}
data$score <- as.numeric(data$score)
data$dataset <- gsub("_exp", "", data$dataset)

data$exp[data$exp=="wo_pol"] <- "no_polarity"
data$exp[data$exp=="wo_pol_res"] <- "no_polarity_no_res"

head(data)

# plot results XGBoost
png("/home/irene/Dropbox/resource-experience-predictive-monitoring/results_xgboost.png", height=900, width=1300)
ggplot(subset(data, metric=="auc" & cls=="results_xgb"), aes(x=nr_events, y=score, color=exp)) + geom_point() + geom_line() +
  theme_bw(base_size=32) + theme(legend.position="top", legend.title=element_text("Resource experience")) + facet_wrap(.~dataset, scales="free", ncol=4)
dev.off()

# plot results random forest
png("/home/irene/Dropbox/resource-experience-predictive-monitoring/results_rf.png", height=900, width=1300)
ggplot(subset(data, metric=="auc" & cls=="rf"), aes(x=nr_events, y=score, color=exp)) + geom_point() + geom_line() +
  theme_bw(base_size=32) + theme(legend.position="top", legend.title=element_text("Resource experience")) + facet_wrap(.~dataset, scales="free", ncol=4)
dev.off()


### Feature importances ###
dt_imp <- read.table("../../results_xgb/feature_importances/results_xgb_single_agg_bpic2012_declined_exp.csv", sep=";", header=T, quote='"', comment.char="")
head(dt_imp)
head(dt_imp[order(-dt_imp$importance),], 20)



### Check if resource experience features help with new resources (i.e. unseen in training data) ###
library(plyr)
library(pROC)
library(ggplot2)

setwd("/home/irene/Repos/resource-experience-predictive-monitoring/results_xgb/")

# Choose one of the following 6 configurations (i.e. read data for one dataset):

# 1. 
data <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2015_1.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2015_1_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('10716070', '11744364', '9264148')

# 2.
data <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_declined.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_declined_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('11299', '11300', '11302', '11309', '11319', '11339')

# 3.
data <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_accepted.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_accepted_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('11299', '11300', '11302', '11309', '11319', '11339')

# 4.
data <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_cancelled.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2012_cancelled_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('11299', '11300', '11302', '11309', '11319', '11339')

# 5.
data <- read.table("detailed_results/detailed_results_xgb_single_agg_traffic_fines_1.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_traffic_fines_1_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('61',
             '62',
             '63',
             '64',
             '65',
             '66',
             '858',
             '859',
             '860',
             '861',
             '862',
             '864',
             '865',
             '866',
             '867',
             '868',
             '869',
             '870')

# 6.
data <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2017_refused.csv", sep=";", header=T)
data$method <- "base"
tmp <- read.table("detailed_results/detailed_results_xgb_single_agg_bpic2017_refused_act_freqs_norm_no_res.csv", sep=";", header=T)
tmp$method <- "act_freqs_norm_no_res"
data <- rbind(data, tmp)
new_res <- c('User_105',
             'User_134',
             'User_135',
             'User_137',
             'User_140',
             'User_69',
             'User_72',
             'User_81',
             'User_88',
             'User_94',
             'User_98')

head(data)
table(data$resource)

# AUC over all resources
aucs <- ddply(data, .(cls, dataset, nr_events, params, method), function(x) auc(x$actual, x$predicted))
ggplot(aucs, aes(x=nr_events, y=V1, color=method)) + geom_point() + geom_line() + theme_bw()

# AUC, given only resources that exist in the training set
aucs <- ddply(subset(data, !(resource %in% new_res)), .(cls, dataset, nr_events, params, method), function(x) ifelse(length(unique(x$actual)) > 1, auc(x$actual, x$predicted), 0.5))
ggplot(aucs, aes(x=nr_events, y=V1, color=method)) + geom_point() + geom_line() + theme_bw()

# AUC, given only "new" resources that do not exist in the training set
aucs <- ddply(subset(data, resource %in% new_res), .(cls, dataset, nr_events, params, method), function(x) ifelse(length(unique(x$actual)) > 1, auc(x$actual, x$predicted), 0.5))
ggplot(aucs, aes(x=nr_events, y=V1, color=method)) + geom_point() + geom_line() + theme_bw()

