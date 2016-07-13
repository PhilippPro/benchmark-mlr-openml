library(batchtools)
library(mlr)

setwd("/home/probst/Benchmarking/benchmark-mlr-openml/results")
options(batchtools.progress = FALSE)
regis = loadRegistry("benchmark-mlr-openml")

min(getJobStatus()$started, na.rm = T)
max(getJobStatus()$done, na.rm = T)

res_classif_load = reduceResultsList(ids = 1:187, fun = function(r) as.list(r), reg = regis)

head(data.frame(res_classif_load[[i]]))

result_classif = data.table()
for(i in which(sapply(res_classif_load, function(x) !is.null(x)))){
  print(i)
  result_classif_i = cbind(data.table(t(sapply(res_classif_load[[i]]$results$data, "[[", 5))), did = i,  algo = 1:27)
  result_classif = rbind(result_classif, result_classif_i)
}

result_classif[, sum(is.na(acc.test.mean)), by = algo]
res_classif = result_classif[, list(algo, rank = rank(acc.test.mean, na.last = "keep")), by = did]
barplot(sort(res_classif[, mean(rank, na.rm=T), by = algo]$V1, decreasing = T), 
        names.arg = substr(names(res_classif_load[[i]]$results$data)[order(res_classif[, mean(rank,na.rm=T), by = algo]$V1, decreasing = T, na.last = NA)],9,100), 
        col = "blue", las = 2, main = "Wer hat den Längsten?", ylab = "Average Rank")


# ohne na's
sum_nas = result_classif[, sum(is.na(acc.test.mean)), by = did]
dids_ok = sum_nas[V1 <= 0,]$did

result_classif_na = result_classif[did %in% dids_ok,]
result_classif_na[, sum(is.na(acc.test.mean)), by = did]
result_classif_na[, sum(is.na(acc.test.mean)), by = algo]
result_classif_na = result_classif_na[algo %in% which(result_classif_na[, sum(is.na(acc.test.mean)), by = algo]$V1 == 0)]

# mean
res_classif_na_mean = result_classif_na[, list(acc = mean(acc.test.mean, na.rm=T), ber = mean(ber.test.mean, na.rm=T), 
                                               multiclass.au1u = mean(multiclass.au1u.test.mean, na.rm=T), 
                                               multiclass.brier = mean(multiclass.brier.test.mean, na.rm=T),
                                               logloss = mean(logloss.test.mean, na.rm=T)), by = algo]

pdf("best_algo_classif_mean.pdf",width=12,height=9)
for (i in colnames(res_classif_na_mean)[2:6]){
  setkeyv(res_classif_na_mean, cols = i)
  par(mar=c(10, 4, 4, 1))
  barplot(revert(i, unlist(res_classif_na_mean[, i, with = F])), 
          names.arg = substr(names(res_classif_load[[1]]$results$data)[revert(i, res_classif_na_mean$algo)],9,100), ylim = range(unlist(res_classif_na_mean[, i, with = F])), xpd = FALSE,
          col = "blue", las = 2, main = paste0("Wer hat den Längsten? (", i, ")"), ylab = paste0("Mean of ", i ," of the 27 classification learners on 77 clean datasets"))
}
dev.off()

# ranks

res_classif_na = result_classif_na[, list(algo, acc = rank(acc.test.mean, na.last = "keep"),
                                                     ber = rank(ber.test.mean, na.last = "keep"),
                                                     multiclass.au1u = rank(multiclass.au1u.test.mean, na.last = "keep"),
                                                     multiclass.brier = rank(multiclass.brier.test.mean, na.last = "keep"),
                                                     logloss = rank(logloss.test.mean, na.last = "keep")), by = did]
res_classif_na_rank = res_classif_na[, list(acc = mean(acc, na.rm=T), ber = mean(ber, na.rm=T), 
                                           multiclass.au1u = mean(multiclass.au1u, na.rm=T), 
                                           multiclass.brier = mean(multiclass.brier, na.rm=T),
                                           logloss = mean(logloss , na.rm=T)), by = algo]

revert = function(x, y) {
  if (x %in% c("acc", "multiclass.au1u")){
  return(rev(y))
    } else {
    return(y)}
}

pdf("best_algo_classif_rank.pdf",width=12,height=9)
for (i in colnames(res_classif_na_rank)[2:6]){
setkeyv(res_classif_na_rank, cols = i)
par(mar=c(10, 4, 4, 1))
barplot(revert(i, unlist(res_classif_na_rank[, i, with = F])), 
        names.arg = substr(names(res_classif_load[[1]]$results$data)[revert(i, res_classif_na_rank$algo)],9,100), 
        col = "blue", las = 2, main = paste0("Wer hat den Längsten? (", i, ")"), ylab = paste0("Average Rank (", i ,") of the 27 classification learners on 77 clean datasets"))
}
dev.off()


########################################################################################################
# Regression

res_regr_load = reduceResultsList(ids = 188:298, fun = function(r) as.list(r), reg = regis)

head(data.frame(res_regr_load[[1]]))

result_regr = data.table()
for(i in which(sapply(res_regr_load, function(x) !is.null(x)))){
  print(i)
  result_regr_i = cbind(data.table(t(sapply(res_regr_load[[i]]$results$data, "[[", 5))), did = i,  algo = 1:31)
  result_regr = rbind(result_regr, result_regr_i)
}

result_regr[, sum(is.na(mse.test.mean)), by = algo]
res_regr = result_regr[, list(algo, rank = rank(mse.test.mean, na.last = "keep")), by = did]
barplot(sort(res_regr[, mean(rank, na.rm=T), by = algo]$V1, decreasing = T), 
        names.arg = substr(names(res_regr_load[[i]]$results$data)[order(res_regr[, mean(rank,na.rm=T), by = algo]$V1, decreasing = T, na.last = NA)],6,100), 
        col = "blue", las = 2, main = "Wer hat den Längsten?", ylab = "Average Rank")


# ohne na's
sum_nas = result_regr[, sum(is.na(mse.test.mean)), by = did]
dids_ok = sum_nas[V1 <= 4,]$did

result_regr_na = result_regr[did %in% dids_ok,]
result_regr_na[, sum(is.na(mse.test.mean)), by = did]
result_regr_na[, sum(is.na(mse.test.mean)), by = algo]
#result_regr_na = result_regr_na[algo %in% which(result_regr_na[, sum(is.na(mse.test.mean)), by = algo]$V1 == 0)]

res_regr_na = result_regr_na[, list(algo, mse = rank(mse.test.mean, na.last = "keep"),
                                          mae = rank(mae.test.mean, na.last = "keep"),
                                          medae = rank(medae.test.mean, na.last = "keep"),
                                          medse = rank(medse.test.mean, na.last = "keep")), by = did]

res_regr_na_rank = res_regr_na[,list(mse = mean(mse, na.rm=T), mae = mean(mae, na.rm=T), 
                                     medae = mean(medae, na.rm=T), 
                                           medse  = mean(medse , na.rm=T)), by = algo]

pdf("best_algo_regr.pdf",width=12,height=9)
for (i in colnames(res_regr_na_rank)[2:5]){
  setkeyv(res_regr_na_rank, cols = i)
  par(mar=c(10, 4, 4, 1))
  barplot(unlist(res_regr_na_rank[, i, with = F]), 
          names.arg = substr(names(res_regr_load[[1]]$results$data)[res_regr_na_rank$algo],6,100), 
          col = "blue", las = 2, main = paste0("Wer hat den Längsten? (", i, ")"), ylab = paste0("Average Rank (", i ,") of the 31 regression learners on x clean datasets"))
}
dev.off()


