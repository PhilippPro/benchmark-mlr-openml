library(mlr)
library(batchtools)
library(plyr)

dir = "C:/Arbeit/R/Github/benchmark-mlr-openml"
source(paste0(dir,"/code/benchmark_defs.R"))
setwd(paste0(dir,"/results"))

unlink("benchmark-mlr-openml", recursive = TRUE)
regis = makeExperimentRegistry("benchmark-mlr-openml", 
                               packages = c("mlr", "OpenML", "methods"), 
                               source = paste0(dir, "/code/benchmark_defs.R"),
                               work.dir = paste0(dir, "/results"),
                               conf.file = paste0(dir,"/code/.batchtools.conf.R")
)
regis$cluster.functions = makeClusterFunctionsMulticore(ncpus = 6) 

# add selected OML datasets as problems
for (did in OMLDATASETS) {
  data = list(did = did)
  addProblem(name = as.character(did), data = data)
}

# add one generic 'algo' that evals the RF in hyperpar space
addAlgorithm("eval", fun = function(job, data, instance,  ...) {
  par.vals = list(...)
  oml.dset = getOMLDataSet(data$did)             
  task = convertOMLDataSetToMlr(oml.dset)
  type = getTaskType(task)
  # set here better defaults for each package?
  if(type == "classif") {
    learners = listLearners("classif", properties = c("multiclass", "prob", "factors"))$class
    # FIXME: In Abhängigkeit der Eigenschaft des Datensatzes definieren
    # classif.boosting braucht sehr lange
    # Fehler: lognet (zu wenig Beobachtungen pro Klasse), gbm, glmnet, ksvm, lda, qda, rda
    learners = learners[!(learners %in% c("classif.boosting"))]
    learners = lapply(learners, function(x) makeLearner(x, predict.type = "prob"))
    }
  if(type == "regr"){
    learners = listLearners("regr", properties = c("factors"))$class
    # regr.crs, regr.penalized.lasso, regr.penalized.ridge geht nicht!
    # regr.btgp, regr.btgpllm und regr.btlm brauchen sehr lange
    learners = learners[!(learners %in% c("regr.btgp", "regr.btgpllm","regr.btlm"))]
    learners = lapply(learners, makeLearner)
  }
 
  measures = MEASURES(type)
  rdesc = makeResampleDesc("RepCV", folds = 10, reps = 10, stratify = FALSE)
  configureMlr(on.learner.error = "warn", show.learner.output = FALSE)
  bmr = benchmark(learners, task, rdesc, measures, keep.pred = FALSE, models = FALSE, show.info = TRUE)
  bmr
})

set.seed(124)
ades = data.frame(c(1))
addExperiments(algo.designs = list(eval = ades))
summarizeExperiments()

ids = chunkIds(findNotDone(), chunk.size = 5)
submitJobs(ids)
# ranger braucht ca. 3 h für regression + classification
submitJobs(1)
submitJobs(100:110)
#submitJobs(ids, resources = list(chunk.ncpus = 9))
getStatus()
getErrorMessages()
