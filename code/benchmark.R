library(mlr)
library(batchtools)
library(plyr)

dir = "/home/probst/Benchmarking/benchmark-mlr-openml"
setwd(paste0(dir,"/results"))
source(paste0(dir,"/code/benchmark_defs.R"))

unlink("benchmark-mlr-openml", recursive = TRUE)
regis = makeExperimentRegistry("benchmark-mlr-openml", 
                               packages = c("mlr", "OpenML", "methods"), 
                               source = paste0(dir, "/code/benchmark_defs.R"),
                               work.dir = paste0(dir, "/results"),
                               conf.file = paste0(dir,"/code/.batchtools.conf.R")
)
regis$cluster.functions = makeClusterFunctionsMulticore() 

# add selected OML datasets as problems
for (did in OMLDATASETS) {
  data = list(did = did)
  addProblem(name = as.character(did), data = data)
}

# add one generic 'algo' that evals the RF in hyperpar space
addAlgorithm("eval", fun = function(job, data, instance, lrn.id, ...) {
  par.vals = list(...)
  oml.dset = getOMLDataSet(data$did)             
  task = convertOMLDataSetToMlr(oml.dset)
  type = getTaskType(task)
  #par.vals = par.vals[!(is.na(par.vals))]
  #par.vals = CONVERTPARVAL(par.vals, task, lrn.id)
  lrn.id = paste0(type, ".", lrn.id)
  lrn = switch(type, "classif" = makeLearner(lrn.id, predict.type = "prob"), "regr" = makeLearner(lrn.id))
  #lrn = setHyperPars(lrn, par.vals = par.vals)
  measures = MEASURES(type)
  rdesc = makeResampleDesc("RepCV", folds = 5, reps = 2, stratify = FALSE)
  resample(lrn, task, rdesc, measures)$aggr
  })

# Take all available learners in mlr that can do classification and regression
set.seed(124)
ades = data.frame()
for (lid in LEARNERIDS) {
  ps = makeMyParamSet(lid, task = NULL)
  des.size = DESSIZE(ps)
  d = generateDesign(des.size, ps)
  d = cbind(lrn.id = lid, d, stringsAsFactors = FALSE)
  ades = rbind.fill(ades, d)
}
addExperiments(algo.designs = list(eval = ades))

