# Abzugsdatum: 28.01 / 29.01
options(java.parameters = "- Xmx1024m") # Should avoid java gc overhead
library(OpenML)

# saveOMLConfig(apikey = "6df93acdb13899f48fd6bd689b0dd7cf", arff.reader = "farff", overwrite=TRUE)
saveOMLConfig(apikey = "6df93acdb13899f48fd6bd689b0dd7cf", arff.reader = "RWeka", overwrite=TRUE)
datasets = listOMLDataSets() 
tasks = listOMLTasks()
tasktypes = listOMLTaskTypes()

clas = subset(tasks, task.type == "Supervised Classification") # 1860 Class.-Tasks

# nur Datensaetze ohne NA-Werte
clas = clas[!(clas$name %in% clas$name[clas$NumberOfMissingValues != 0]), ]

# Jeder Datensatz nur einmal
clas = clas[order(clas$did),]
logic = logical(nrow(clas))
logic = rep(TRUE, nrow(clas))
for(i in 2:nrow(clas))
  if(clas$did[i] == clas$did[i-1]) logic[i] = FALSE
clas = clas[logic,]

nans = character(nrow(clas))
# Nur Datensätze mit Categorical target
for(j in 1:nrow(clas)){
  print(j)
  task = getOMLTask(task.id = clas$task_id[j], verbosity=0)
  nans[j] = try(class(task$input$data.set$data[, task$input$data.set$target.features]))
  save(nans, file = "/home/probst/Random_Forest/Parameter_Tuning/Simulation/Classification/nans_clas.RData")
  gc()
}

load("/home/probst/Random_Forest/Parameter_Tuning/Simulation/Classification/nans_clas.RData")
clas = clas[which(nans == "factor"), ]

# doppelt vorkommende entfernen
doppelt = names(sort(table(clas$name)[table(clas$name) > 1]))
doppelt = clas[clas$name %in% doppelt,]
doppelt = doppelt[order(doppelt$name), ]
doppelt[100:121, c(1,3,5,7,9,10,11,14,15)]
raus = c(2075, 9900, 9901, 3833, 3876, 3877, 3564, 3860, 10095, 3854, 9898, 7295, 7293, 7291, 3846, 3567, 3878, 2072, 3020, 7593, 3951, 3578, 3874, 3851, 3875, 3868, 3888,
         3837, 3832, 3840, 3869, 3841, 3834, 3883, 3825, 3885, 3858, 3819, 3822, 9892, 3843, 3884, 3827, 3882, 9891, 3859, 3811, 3607, 3821, 3576, 3575, 
         3817, 3816, 3818, 3857, 9940, 3731, 3879, 9941, 9942, 3842, 3872, 3836, 3828)
clas = clas[!(clas$task_id %in% raus),]
clas = clas[order(clas$name), ]

# Friedman-, volcanoes- und trX-Datensaetze raus
clas = clas[substr(clas$name,1,9) != "volcanoes" & substr(clas$name,1,4) != "fri_" & substr(clas$name,1,3) != "tr1" & substr(clas$name,1,3) != "tr2" & substr(clas$name,1,3) != "tr3" & substr(clas$name,1,3) != "tr4", ]

nrow(clas) #  509 datasets
sum(clas$NumberOfFeatures) # 1168491 Features
hist(clas$NumberOfFeatures)
sum(clas[clas$NumberOfFeatures < 100,]$NumberOfFeatures)

# Datensaetze nach Groesse ordnen (n*p)
clas = clas[order(clas$NumberOfFeatures * clas$NumberOfInstances), ]

# Datensaetze mit mehr als 50 Kategorien in einer Variablen rausnehmen
more = logical(nrow(clas))
for(j in 1:nrow(clas)){
  print(j)
  task = try(getOMLTask(task.id = clas$task_id[j], verbosity=0))
  classen = sapply(task$input$data.set$data, class)
  indiz = which(classen == "character" | classen == "factor")
  if(any(apply(as.data.frame(task$input$data.set$data[, indiz]), 2, function(x) length(unique(x))) > 50))
    more[j] = TRUE
}
clas = clas[which(more == FALSE) ,]

# Offensichtlich gleiche Datensaetze entfernen
raus = c(9962, 3633, 40, 3692, 3660, 3794, 3770, 3771, 3772, 3053, 9936, 3736, 10098, 10099, 10100, 3903, 3672, 3681, 3839, 9943, 3688, 9972, 3618, 3764, 145, 202, 3925, 3895, 3990, 151, 
         220, 221, 224, 226, 225, 228, 227, 3506, 205, 207, 2262, 209, 150, 138, 210, 211, 152, 148, 146,137, 155, 132, 2136, 133, 216, 144, 201, 141, 140)
clas = clas[!(clas$task_id %in% raus),]

# Getrennte Analyse für kleine und große Datensätze
clas_small = clas[which(clas$NumberOfInstances < 1000 & clas$NumberOfFeatures < 1000 & clas$NumberOfClasses < 30),]
clas_big = clas[which(!(clas$NumberOfInstances < 1000 & clas$NumberOfFeatures < 1000 & clas$NumberOfClasses < 30)),]

save(clas, clas_small, clas_big, file="/home/probst/Random_Forest/RFSplitbias/results/clas.RData")







# Datensaetze abspeichern
#for(j in 1:nrow(clas)){
#  print(j)
#  task = getOMLTask(task.id = clas$task_id[j], verbosity=0)
#  save(task, file = paste("/home/probst/Random_Forest/Parameter_Tuning/Simulation/Classification/Datasets/", j, ".RData", sep=""))
#  gc()
#}

######################################################### Regression #########################################################

options(java.parameters = "- Xmx1024m") # Should avoid java gc overhead
library(OpenML)

saveOMLConfig(apikey = "6df93acdb13899f48fd6bd689b0dd7cf", arff.reader = "RWeka", overwrite=TRUE)
datasets = listOMLDataSets() 
length(datasets$name)
tasks = listOMLTasks()
tasktypes = listOMLTaskTypes()

tasktypes
reg = subset(tasks, task_type == "Supervised Regression") # 1854 Regr.-Tasks
# nur Datensaetze ohne NA-Werte
reg = reg[!(reg$name %in% reg$name[reg$NumberOfMissingValues != 0]), ]

# Jeder Datensatz nur einmal
reg = reg[order(reg$did),]
logic = logical(nrow(reg))
logic = rep(TRUE, nrow(reg))
for(i in 2:nrow(reg))
  if(reg$did[i] == reg$did[i-1]) logic[i] = FALSE
reg = reg[logic,]

nans = character(nrow(reg))
#Nur Datensätze mit numeric/integer target
for(j in 1:length(reg$task_id)){
  print(j)
  task = getOMLTask(task.id = reg$task_id[j], verbosity=0)
  nans[j] = try(class(task$input$data.set$data[, task$input$data.set$target.features]))
  save(nans, file = "/home/probst/Random_Forest/Parameter_Tuning/Simulation/Regression/nans_reg.RData")
  gc()
}

load("/home/probst/Random_Forest/Parameter_Tuning/Simulation/Regression/nans_reg.RData")
reg = reg[which(nans == "numeric" | nans == "integer"), ]

# doppelt vorkommende entfernen
doppelt = names(sort(table(reg$name)[table(reg$name) > 1]))
doppelt = reg[reg$name %in% doppelt, ]
doppelt = doppelt[order(doppelt$name), ]
doppelt[, c(1,3,5,7,9,10,11,14,15)]

raus = c(4892, 4883, 4876, 5025, 4874)

reg = reg[!(reg$task_id %in% raus),]

reg = reg[order(reg$name), ]

# Friedman-, volcanoes- und trX-Datensaetze raus
reg = reg[substr(reg$name,1,3) != "fri" & substr(reg$name,1,4) != "a3a" & substr(reg$name,1,3) != "a4a" & substr(reg$name,1,3) != "a5a" 
          & substr(reg$name,1,3) != "a6a" & substr(reg$name,1,3) != "a7a" & substr(reg$name,1,3) != "a8a" & substr(reg$name,1,3) != "a9a" 
          & substr(reg$name,1,9) != "autoPrice" & substr(reg$name,1,9) != "chscase_c" & substr(reg$name,1,4) != "QSAR" , ]

# Künstliche Drug-Datensätze raus (mit 1143 Numeric Features)
reg = reg[reg$NumberOfNumericFeatures != 1143, ]

nrow(reg) #  160 datasets
sum(reg$NumberOfFeatures) # 16810 Features
hist(reg$NumberOfFeatures)
sum(reg[reg$NumberOfFeatures < 100,]$NumberOfFeatures)

# Datensaetze nach Groesse ordnen (n*p)
reg = reg[order(reg$NumberOfFeatures * reg$NumberOfInstances), ]

# Nur 1 Feature = Target, das ist unsinnig
reg = reg[-121, ]

# Datensaetze mit mehr als 50 Kategorien in einer Variablen rausnehmen
more2 = logical(nrow(reg))
for(j in 1:nrow(reg)){
  print(j)
  task = try(getOMLTask(task.id = reg$task_id[j], verbosity=0))
  classen = sapply(task$input$data.set$data, class)
  indiz = which(classen == "character" | classen == "factor")
  if(any(apply(as.data.frame(task$input$data.set$data[, indiz]), 2, function(x) length(unique(x))) > 50))
    more2[j] = TRUE
}
reg = reg[more2 == FALSE ,]

# Offensichtlich gleiche Datensaetze entfernen
raus = c(4993, 5013, 5023, 2280, 2313)
reg = reg[!(reg$task_id %in% raus),]

# Getrennte Analyse für kleine und große Datensätze
reg_small = reg[which(reg$NumberOfInstances < 1000 & reg$NumberOfFeatures < 1000),]
reg_big = reg[which(!(reg$NumberOfInstances < 1000 & reg$NumberOfFeatures < 1000)),]

save(reg, reg_small, reg_big, file="/home/probst/Random_Forest/RFParset/results/reg.RData")

# Datensaetze abspeichern
#for(j in 1:nrow(reg)){
#  print(j)
#  task = getOMLTask(task.id = reg$task_id[j], verbosity=0)
#  save(task, file = paste("/home/probst/Random_Forest/Parameter_Tuning/Simulation/Regression/Datasets/", j, ".RData", sep=""))
#  gc()
#}
