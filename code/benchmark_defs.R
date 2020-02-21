load("C:/Arbeit/R/Github/benchmark-mlr-openml/results/clas.RData")
load("C:/Arbeit/R/Github/benchmark-mlr-openml/results/reg.RData")
tasks = rbind(clas_small, reg_small)

OMLDATASETS = tasks$did[!(tasks$did %in% c(1054, 1071, 1065))] # Cannot guess task.type from data! for these 3

MEASURES = function(x) switch(x, "classif" = list(acc, ber, mmce, multiclass.au1u, multiclass.brier, logloss, timetrain), "regr" = list(mse, mae, medae, medse, timetrain))

