library(batchtools)
library(mlr)

setwd("/home/probst/Benchmarking/benchmark-mlr-openml/results")
options(batchtools.progress = FALSE)
regis = loadRegistry("benchmark-mlr-openml")

min(getJobStatus()$started, na.rm = T)
max(getJobStatus()$done, na.rm = T)

res_classif_load = reduceResultsList(ids = 1:187, fun = function(r) as.list(r), reg = regis)
measure.names = c("accuracy", "balanced error rate", "multiclass auc", "multiclass brier score", "logarithmic loss", "time for training")
measure.names.short = c("acc", "ber", "multiclass.au1u", "multiclass.brier", "logloss", "timetrain")
algo.names = substr(names(res_classif_load[[1]]$results$data), 9, 100)
head(data.frame(res_classif_load[[1]]))

res_classif = data.table()
for(i in which(sapply(res_classif_load, function(x) !is.null(x)))){
  print(i)
  res_classif_i = cbind(data.table(t(sapply(res_classif_load[[i]]$results$data, "[[", 5))), did = i,  algo = 1:27)
  res_classif = rbind(res_classif, res_classif_i)
}

colnames(res_classif)[c(1:2, 4:7)] = measure.names.short
res_classif[, sum(is.na(acc)), by = algo]

NA_is_smallest = function(x, na.last = FALSE) {
  ind = which(is.na(x)) 
  x = rank(x, na.last = na.last)  
  x[ind] =  mean(x[ind])
  x
}

revert = function(x, y) {
  if (x %in% c("acc", "multiclass.au1u")){
    return(rev(y))
  } else {
    return(y)}
}

# multiclass.au1u muss NA sein, wenn die anderen auch NA
res_classif[is.na(res_classif$acc), multiclass.au1u := NA]
res_classif[is.na(res_classif$acc), timetrain := NA]

rank_analysis_with_na = function(res_classif) {
  res_classif_rank = res_classif[, list(algo, acc = NA_is_smallest(acc),
                                        ber = NA_is_smallest(ber, na.last = TRUE),
                                        multiclass.au1u = NA_is_smallest(multiclass.au1u),
                                        multiclass.brier = NA_is_smallest(multiclass.brier, na.last = TRUE),
                                        logloss = NA_is_smallest(logloss, na.last = TRUE),
                                        timetrain = NA_is_smallest(timetrain, na.last = TRUE)), by = did]
  
  res_classif_rank_mean = res_classif_rank[, list(acc = mean(acc), ber = mean(ber), 
                                                  multiclass.au1u = mean(multiclass.au1u), 
                                                  multiclass.brier = mean(multiclass.brier),
                                                  logloss = mean(logloss), 
                                                  timetrain = mean(timetrain)), by = algo]
  
  for (i in colnames(res_classif_rank_mean)[2:7]){
    setkeyv(res_classif_rank_mean, cols = i)
    measure.name = measure.names[i == measure.names.short]
    
    plot_data = data.frame(learner = revert(i, algo.names[res_classif_rank_mean$algo]), 
                           average_rank = revert(i, unlist(res_classif_rank_mean[, i, with = F])))
    plot_data$learner = factor(plot_data$learner, levels = plot_data$learner)
    print(ggplot(plot_data, aes(x = learner, y = average_rank)) + geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
            ggtitle(paste0("Comparison of ", measure.name, " of 27 multiclass classification learners")) + ylab(paste0("Mean of ",measure.name, " rank on ", nrow(res_classif)/nrow(res_classif_rank_mean) ," classification datasets")) + xlab("multiclass classification learner"))
  }
}


pdf("best_algo_classif_with_na_rank.pdf",width=12,height=9)
rank_analysis_with_na(res_classif)
dev.off()

rank_analysis_with_na(res_classif[did %in% which(tasks[1:187,]$NumberOfClasses == 2)])
rank_analysis_with_na(res_classif[did %in% which(tasks[1:187,]$NumberOfClasses != 2)])



# ohne na's
sum_nas = res_classif[, list(acc = sum(is.na(acc)), multiclass.brier = sum(is.na(multiclass.brier)), 
                             logloss = sum(is.na(logloss))), by = did]
dids_ok = sum_nas[acc <= 0 & logloss <= 0 & multiclass.brier <=0,]$did

res_classif[is.na(res_classif$logloss)]$acc
res_classif[is.na(res_classif$multiclass.brier)]$acc

res_classif_na = res_classif[did %in% dids_ok,]
res_classif_na[, sum(is.na(acc)), by = did]
res_classif_na[, sum(is.na(acc)), by = algo]
res_classif_na = res_classif_na[algo %in% which(res_classif_na[, sum(is.na(acc)), by = algo]$V1 == 0)]

# mean
mean_analysis = function(res_classif_na) {
  res_classif_na_mean = res_classif_na[, list(acc = mean(acc), ber = mean(ber), 
                                                 multiclass.au1u = mean(multiclass.au1u), 
                                                 multiclass.brier = mean(multiclass.brier),
                                                 logloss = mean(logloss),
                                                 timetrain = mean(timetrain)), by = algo]
   
  for (i in colnames(res_classif_rank_mean)[2:7]){
    print(i)
    setkeyv(res_classif_na_mean, cols = i)
    measure.name = measure.names[i == measure.names.short]
    
    plot_data = data.frame(learner = revert(i, algo.names[res_classif_na_mean$algo]), 
                           average_rank = revert(i, unlist(res_classif_na_mean[, i, with = F])))
    plot_data$learner = factor(plot_data$learner, levels = plot_data$learner)
    print(ggplot(plot_data, aes(x = learner, y = average_rank)) + geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_cartesian(ylim=range(plot_data$average_rank)) +
            ggtitle(paste0("Comparison of ", measure.name, " of 27 multiclass classification learners")) + ylab(paste0("Mean of ",measure.name, " rank on 187 classification datasets")) + xlab("learner"))
  }
}

pdf("best_algo_classif_mean.pdf",width=12,height=9)
mean_analysis(res_classif_na)
dev.off()
# binary classification problems
mean_analysis(res_classif_na[did %in% which(tasks[1:187,]$NumberOfClasses == 2)])
# multiclass classification problems
mean_analysis(res_classif_na[did %in% which(tasks[1:187,]$NumberOfClasses != 2)])


# ranks
rank_analysis = function(res_classif_na) {
  res_classif_na_rank = res_classif_na[, list(algo, acc = rank(acc),
                                              ber = rank(ber),
                                              multiclass.au1u = rank(multiclass.au1u),
                                              multiclass.brier = rank(multiclass.brier),
                                              logloss = rank(logloss), 
                                              timetrain = rank(timetrain)), by = did]
  res_classif_na_rank = res_classif_na_rank[, list(acc = mean(acc), ber = mean(ber), 
                                                   multiclass.au1u = mean(multiclass.au1u), 
                                                   multiclass.brier = mean(multiclass.brier),
                                                   logloss = mean(logloss),
                                                   timetrain = mean(timetrain)), by = algo]
  
  for (i in colnames(res_classif_rank_mean)[2:7]){
    print(i)
    setkeyv(res_classif_na_rank, cols = i)
    measure.name = measure.names[i == measure.names.short]
    
    plot_data = data.frame(learner = revert(i, algo.names[res_classif_na_rank$algo]), 
                           average_rank = revert(i, unlist(res_classif_na_rank[, i, with = F])))
    plot_data$learner = factor(plot_data$learner, levels = plot_data$learner)
    print(ggplot(plot_data, aes(x = learner, y = average_rank)) + geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_cartesian(ylim=range(plot_data$average_rank)) +
            ggtitle(paste0("Comparison of ", measure.name, " of 27 multiclass classification learners")) + ylab(paste0("Mean of ",measure.name, " rank on 187 classification datasets")) + xlab("learner"))
  }
}

pdf("best_algo_classif_rank.pdf",width=12,height=9)
rank_analysis(res_classif_na)
dev.off()
# binary classification problems
rank_analysis(res_classif_na[did %in% which(tasks[1:187,]$NumberOfClasses == 2)])
# multiclass classification problems
rank_analysis(res_classif_na[did %in% which(tasks[1:187,]$NumberOfClasses != 2)])


########################################################################################################
# Regression

res_regr_load = reduceResultsList(ids = 188:298, fun = function(r) as.list(r), reg = regis)
measure.names = c("mse", "mae", "medae", "medse", "timetrain")
measure.names.short = c("mse", "mae", "medae", "medse", "timetrain")
algo.names = substr(names(res_regr_load[[1]]$results$data), 6, 100)

res_regr = data.table()
for(i in which(sapply(res_regr_load, function(x) !is.null(x)))) {
  print(i)
  res_regr_i = cbind(data.table(t(sapply(res_regr_load[[i]]$results$data, "[[", 5))), did = i,  algo = 1:31)
  res_regr = rbind(res_regr, res_regr_i)
}

colnames(res_regr)[1:5] = measure.names.short
res_regr[, sum(is.na(mse)), by = algo]
res_regr_rank = res_regr[, list(algo, mse = NA_is_smallest(mse, na.last = TRUE), mae = NA_is_smallest(mae, na.last = TRUE), medae = NA_is_smallest(medae, na.last = TRUE), medse = NA_is_smallest(medse, na.last = TRUE), timetrain = NA_is_smallest(timetrain, na.last = TRUE)), by = did]
res_regr_rank_mean = res_regr_rank[, list(mse = mean(mse), mae = mean(mae), medae = mean(medae), medse = mean(medse), timetrain = mean(timetrain)), by = algo]

pdf("best_algo_regr_with_na_rank.pdf",width=12,height=9)
for (i in colnames(res_regr_rank)[3:7]){
  setkeyv(res_regr_rank_mean, cols = i)
  measure.name = measure.names[i == measure.names.short]
  
  plot_data = data.frame(learner = revert(i, algo.names[res_regr_rank_mean$algo]), 
                         average_rank = revert(i, unlist(res_regr_rank_mean[, i, with = F])))
  plot_data$learner = factor(plot_data$learner, levels = plot_data$learner)
  print(ggplot(plot_data, aes(x = learner, y = average_rank)) + geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
          ggtitle(paste0("Comparison of ", measure.name, " of 31 regression learners")) + ylab(paste0("Mean of ",measure.name, " rank on ", nrow(res_regr)/nrow(res_regr_rank_mean) ," regression datasets")) + xlab("regression learner"))
}
dev.off()

# (fast) ohne na's
sum_nas = result_regr[, sum(is.na(mse.test.mean)), by = did]
dids_ok = sum_nas[V1 <= 2,]$did

res_regr_na = res_regr[did %in% dids_ok,]
res_regr_na[, sum(is.na(mse)), by = did]
res_regr_na[, sum(is.na(mse)), by = algo]

res_regr[, sum(is.na(mse)), by = algo]
res_regr_na_rank = res_regr_na[, list(algo, mse = NA_is_smallest(mse, na.last = TRUE), mae = NA_is_smallest(mae, na.last = TRUE), medae = NA_is_smallest(medae, na.last = TRUE), medse = NA_is_smallest(medse, na.last = TRUE), timetrain = NA_is_smallest(timetrain, na.last = TRUE)), by = did]
res_regr_na_rank_mean = res_regr_rank[, list(mse = mean(mse), mae = mean(mae), medae = mean(medae), medse = mean(medse), timetrain = mean(timetrain)), by = algo]

pdf("best_algo_regr_rank.pdf",width=12,height=9)
for (i in colnames(res_regr_na_rank)[3:7]){
  setkeyv(res_regr_na_rank_mean, cols = i)
  measure.name = measure.names[i == measure.names.short]
  
  plot_data = data.frame(learner = revert(i, algo.names[res_regr_na_rank_mean$algo]), 
                         average_rank = revert(i, unlist(res_regr_na_rank_mean[, i, with = F])))
  plot_data$learner = factor(plot_data$learner, levels = plot_data$learner)
  print(ggplot(plot_data, aes(x = learner, y = average_rank)) + geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
          ggtitle(paste0("Comparison of ", measure.name, " of 31 regression learners")) + ylab(paste0("Mean of ", measure.name, " rank on ", nrow(res_regr_na)/nrow(res_regr_na_rank_mean) ," regression datasets")) + xlab("regression learner"))
}
dev.off()




