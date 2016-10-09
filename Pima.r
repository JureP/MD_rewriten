
## source("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Koda\\Pima.r")
## funkcije:
source("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Koda/BackEnd/Funkcije.r")
library(mlbench)
library(dplyr)
library(caret)
data(PimaIndiansDiabetes)
library(FNN)

## podatki

osnovniPodatki <- na.omit(PimaIndiansDiabetes)
onsovniPodatki_response <- osnovniPodatki[, 9]
onsovniPodatki_FM <- osnovniPodatki[, -9]

set.seed(123)

## adjustable parameters 
nRepets <- 20 ## number of cross validations
nBL <- 15 ## number of baseLearners
K <- 5
Kp <- 5
metaALG <- "rf"
hC <- 0.9
thrshld <- 0.8

## parameters 
modelName <- "linearModel"
FileData <- 'Pima'
FolderData <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/Pima"
# FolderBL <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/Pima/02_BaseLearner"
# FolderOP <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/03_OutputProfile"

dir.create(FolderData, recursive = FALSE)
# dir.create(FolderBL, recursive = FALSE)
# dir.create(FolderOP, recursive = FALSE)
classesOfProblem <- levels(onsovniPodatki_response)


## data partition
Save_dataPartition(onsovniPodatki_FM, onsovniPodatki_response,
					k = 4, n = nRepets, File = FileData,
					Folder = FolderData)
					

					
## test of working (correct selection of sets)
# data <- readRDS("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\Pima\\Partition_1\\Pima_1.rds")

# for(i in 1:(length(data)/2)){
	# data[[2*i-1]] <- data[[2*i-1]][1:(190 - 2*i), ]
	# data[[2*i]] <- data[[2*i]][1:(190 - 2*i)]
# }

# saveRDS(data, "C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\Pima\\Partition_1\\Pima_1.rds")



for(i in 1:nRepets){
	print(i)
	## base learner training
	print("training base learner")
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	onsovniPodatki_FM <- data$Fold1_FM
	onsovniPodatki_response <- data$Fold1_Y
	FolderBL <- paste0(FolderDataPartition, "/02_BaseLearner")
	Save_baseLearner(onsovniPodatki_FM, onsovniPodatki_response,
					nModels = nBL, Folder = FolderBL, File = modelName)
	## base learner prediction	
	print("output profile")
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	namesBL <- paste0(modelName, "_", 1:nBL)
	load <- paste0(FolderBL, "/", namesBL, ".rds")
	models <- lapply(load, readRDS)
	names(models) <- namesBL
	for(fold in c("Fold1", "Fold2", "Fold3", "Fold4")){
		featureMatrix <- data[[paste0(fold, "_FM")]]
		Save_OutputProfile(featureMatrix, models, Folder = FolderOP, File = paste0("OP_", fold))
	}
	## meta learner: meta problem
	print("meta problem")
	metaSet <- data$Fold3_FM
	yMetaSet <- data$Fold3_Y
	neigbourSet <- data$Fold2_FM
	yNeigbourSet <- data$Fold2_Y
	namesOfSet_train <- c("Fold1", "Fold2", "Fold3")
	baseLearner <- namesBL
	setwd(FolderOP)
	OP_meta <- readRDS("OP_Fold3.rds")
	OP_neigbours <- readRDS("OP_Fold2.rds")
	FolderMetaProblem <- paste0(FolderDataPartition, "/04_MetaProblem")
	FolderNeigbour <- paste0(FolderMetaProblem, "/Neigbours")
	save_MetaProblem(metaSet, ## FM of meta set 
				yMetaSet, ## response vector of meta set 
								##[if false creates meta FM without meta response]
				neigbourSet, ## set from which neighbours are chosen
							## if neigbourSet = metaSet then neighbours of x from metaSet\x
				yNeigbourSet, ## response vector for neigbour set
				namesOfSet_train, ## name of sets (used to save stuff)
				baseLearner, ## names of base learners 
				OP_meta, ## output profile of meta set
					## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
				OP_neigbours, ## output profile of neigbour set
					## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
				K, ## number of neighbours on FM
				Kp, ## number of neigbours on OP
				FolderNeigbour, ## Folder where neigbours are saved
				FolderMetaProblem ## Folder where meta problem is saved
				)
	## meta learner: training
	print("training meta classifier")
	FolderMetaClassifier <- paste0(FolderDataPartition, "/05_MetaClassifier")
	setwd(FolderMetaProblem)
	for(mtPrblm in dir()[grepl( paste0(namesBL, collapse ="|"),dir())]){
		setwd(FolderMetaProblem)
		metaProblem <- readRDS(mtPrblm)
		OP_classMetaSet <- probToClass(OP_meta, namesBL, classesOfProblem)
		save_trainMetaClassifier(metaProblem = metaProblem, 
								OP_classMetaSet = OP_classMetaSet,
								metaALG = metaALG, hC = hC, 
								FolderMetaClassifier = FolderMetaClassifier)
	}
	## metaPrediction
		## meta problem 
		FolderPrediction <- paste0(FolderDataPartition, "/06_MetaPrediction")
		print("meta problem, predict")
		testSet <- data$Fold4_FM
		yTestSet <- data$Fold4_Y
		neigbourSet <- data$Fold3_FM
		yNeigbourSet <- data$Fold3_Y
		namesOfSet_test <- c("Fold1", "Fold3", "Fold4")
		baseLearner <- namesBL
		setwd(FolderOP)
		OP_test <- readRDS("OP_Fold4.rds")
		OP_neigbours <- readRDS("OP_Fold3.rds")
		FolderMetaProblem <- paste0(FolderPrediction, "/MetaProblem_prediction")
		FolderNeigbour <- paste0(FolderMetaProblem, "/Neigbours")
		save_MetaProblem(testSet, ## FM of meta set 
					yTestSet, ## response vector of meta set 
									##[if false creates meta FM without meta response]
					neigbourSet, ## set from which neighbours are chosen
								## if neigbourSet = metaSet then neighbours of x from metaSet\x
					yNeigbourSet, ## response vector for neigbour set
					namesOfSet_test, ## name of sets (used to save stuff)
					baseLearner, ## names of base learners 
					OP_test, ## output profile of meta set
						## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
					OP_neigbours, ## output profile of neigbour set
						## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
					K, ## number of neighbours on FM
					Kp, ## number of neigbours on OP
					FolderNeigbour, ## Folder where neigbours are saved
					FolderMetaProblem ## Folder where meta problem is saved
					)
	
	## meta problem for test set
	print("kompetentnost")
	setwd(FolderMetaProblem)
	loadMetaProblem <- dir()[grepl(paste0(namesBL, collapse ="\\[|"),dir())]
	parametriModela <- paste0("\\[trainBL]", namesOfSet_test[1],"\\[sosedSet]",namesOfSet_test[2] ,"\\[metaSet]",namesOfSet_test[3] ,"\\[K]",K ,"\\[Kp]",Kp ,"\\[Dist]euclid")
	loadMetaProblem <- loadMetaProblem[grepl(parametriModela,loadMetaProblem)]
	testMetaProblem <- lapply(loadMetaProblem, readRDS)
	names(testMetaProblem) <- sapply(1:length(testMetaProblem), FUN = function(x) testMetaProblem[[x]]$parameters$bl)
	## meta classifier for test
	setwd(FolderMetaClassifier)
	nameMetaClassifier <- namesBL ## c("linearModel_1" ,"linearModel_1" ,"linearModel_1" )
	testMetaClassifierFile <- dir(FolderMetaClassifier)[grepl(paste0(nameMetaClassifier, "\\[", collapse ="|"),dir(FolderMetaClassifier))]
	parametriModela2 <- paste0("\\[trainBL]", namesOfSet_train[1],"\\[sosedSet]",namesOfSet_train[2] ,"\\[metaSet]",namesOfSet_train[3] ,"\\[K]",K ,"\\[Kp]",Kp ,"\\[Dist]euclid\\[hC]", hC)
	testMetaClassifierFile <- testMetaClassifierFile[grepl(parametriModela2,testMetaClassifierFile)]
	testMetaClassifier <- lapply(testMetaClassifierFile, readRDS)
	names(testMetaClassifier) <- gsub(pattern = "(.*BL])(.*)(\\[t.*)", replacement = "\\2", x = testMetaClassifierFile)
	## which meta model is used for which meta problem
	metaModelMap <- nameMetaClassifier
	names(metaModelMap) <- namesBL
	
	
	FolderKompetentnost <- paste0(FolderPrediction, "/Kompetentnost")
	save_Kompetentnost(metaProblem = testMetaProblem, ## list of meta problems 
						metaClassifier = testMetaClassifier, ## list of meta classifiers
						mappingProblemClassifier = metaModelMap, ## named vector mapping meta problem to meta classifier
						Folder = FolderKompetentnost, 
						File = "Competence"
						)
	setwd(FolderKompetentnost)
	competence <- readRDS("Competence.rds")
	accuracy <- NULL
	# for(mdl in namesBL){
		# predict <- competence[, paste0(mdl, "_kompetentnost")]
		# probBorder <- seq(0.5,0.9, 0.025)
		# setwd(FolderMetaProblem)
		# obs <- readRDS(dir()[grepl(paste0(mdl, "\\[", collapse ="|") ,dir())])$metaClass
		# obs <- factor(ifelse(obs == "competent", 1, 0))
		# accuracy <- rbind(accuracy, calculateAccuracy(obs, predict, probBorder))
	# }
	# rownames(accuracy) <- namesBL
	setwd(FolderPrediction)
	saveRDS(accuracy, "accuracyMetaClassifier.rds")
	## ensemble 
	# setwd(FolderOP)
	# OP <- readRDS("OP_Fold4.rds")
	# predClass <- probToClass(OP, namesBL, classesOfProblem)
	# setwd(FolderKompetentnost)
	# competence <- readRDS("Competence.rds")
	# ensembleClass <- ensemblePrediction(method = "vote", predClass,	competence,	threshold = thrshld)
	# setwd(FolderPrediction)
	# add meta data about model
	# saveRDS(ensembleClass, "ensemblePrediction_vote.rds")	
	# check result
	# print("##############################################################################################")
	# print("##############################################################################################")
	# print(sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y))
	# print(confusionMatrix(data$Fold4_Y, ensembleClass)$overall[[1]])
	# print(max(sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y)))
	# cat ("Press [enter] to continue")
	# line <- readline()
}




## meta classifier original (ONE)
for(i in 1:nRepets){
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	print("training meta classifiere origianl")
	## putting together all meta problems of BL
	namesBL <- paste0(modelName, "_", 1:nBL)
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	FolderMetaProblem <- paste0(FolderDataPartition, "/04_MetaProblem")
	FolderMetaClassifier_original <- paste0(FolderDataPartition, "/05_MetaClassifier_original")
	setwd(FolderOP)
	outputProfile <- readRDS("OP_Fold3.rds")
	outputProfileName = "metaSet"
	namesOfSet_train <- c("Fold1", "Fold2", "Fold3")
	FolderSave <- paste0(FolderDataPartition, "/04_MetaProblem_original")
	save_mergeMetaProblems(outputProfile,
							outputProfileName,
							namesBL,
							namesOfSet_train,
							K,
							Kp,
							FolderMetaProblem,
							FolderSave
							)
	## training original meta classifier
	setwd(FolderSave)
	name <- paste0("\\[trainBL]", namesOfSet_train[1],"\\[sosedSet]",namesOfSet_train[2] ,"\\[metaSet]",namesOfSet_train[3] ,"\\[K]",K ,"\\[Kp]",Kp ,"\\[Dist]euclid")
	metaProblem <- readRDS(dir()[grepl(name, dir())])
	nameOP <- paste0("OF_", outputProfileName, "X", length(namesBL), ".rds")
	OP_meta <- readRDS(nameOP)
	OP_classMetaSet <- probToClass(OP_meta, namesBL, classesOfProblem)
	save_trainMetaClassifier(metaProblem = metaProblem, 
							OP_classMetaSet = OP_classMetaSet,
							metaALG = metaALG, hC = hC, 
							FolderMetaClassifier = FolderMetaClassifier_original)
							
	## predictions of original META-DES
	## meta problem for test set
	print("kompetentnost original")
	FolderMetaProblem_test <- paste0(FolderDataPartition, "/06_MetaPrediction/MetaProblem_prediction")
	setwd(FolderMetaProblem_test)
	namesOfSet_test <- c("Fold1", "Fold3", "Fold4")
	loadMetaProblem <- dir()[grepl(paste0(namesBL, collapse ="\\[|"),dir())]
	parametriModela <- paste0("\\[trainBL]", namesOfSet_test[1],"\\[sosedSet]",namesOfSet_test[2] ,"\\[metaSet]",namesOfSet_test[3] ,"\\[K]",K ,"\\[Kp]",Kp ,"\\[Dist]euclid")
	loadMetaProblem <- loadMetaProblem[grepl(parametriModela,loadMetaProblem)]
	testMetaProblem <- lapply(loadMetaProblem, readRDS)
	names(testMetaProblem) <- sapply(1:length(testMetaProblem), FUN = function(x) testMetaProblem[[x]]$parameters$bl)
	## meta classifier for test
	setwd(FolderMetaClassifier_original)
	nameMetaDesOriginal <- paste0("metaKlasifikator[BL]ALL[trainBL]", namesOfSet_train[1] ,"[sosedSet]", 
								namesOfSet_train[2] ,"[metaSet]", namesOfSet_train[3] ,"[K]",K ,"[Kp]", 
								Kp ,"[Dist]euclid[hC]", hC,".rds")
	testMetaClassifier <- list(ALL = readRDS(nameMetaDesOriginal))
	## which meta model is used for which meta problem
	metaModelMap <- rep("ALL", length(namesBL))
	names(metaModelMap) <- namesBL
		## kompetentnost
	FolderKompetentnost_original <- paste0(FolderDataPartition, "/06_MetaPrediction_original/Kompetentnost")
	save_Kompetentnost(metaProblem = testMetaProblem, ## list of meta problems 
						metaClassifier = testMetaClassifier, ## list of meta classifiers
						mappingProblemClassifier = metaModelMap, ## named vector mapping meta problem to meta classifier
						Folder = FolderKompetentnost_original, 
						File = "Competence_original"
						)
	## ensemble prediction original
	# setwd(FolderOP)
	# OP <- readRDS("OP_Fold4.rds")
	# predClass <- probToClass(OP, namesBL, classesOfProblem)
	# setwd(FolderKompetentnost_original)
	# competence <- readRDS("Competence_original.rds")
	# ensembleClass <- ensemblePrediction(method = "vote", predClass,	competence,	threshold = thrshld)	
	# saveRDS(ensembleClass, "ensemblePredictionOriginal_vote.rds")	
	# print(confusionMatrix(data$Fold4_Y, ensembleClass)$overall[[1]])
	# print("##############################################################################################")
	# print("##############################################################################################")

}


## add meta data to competence and ensemble prediction

accuracyOriginal <- NULL
accuracyInd <- NULL
accuracyBest <- NULL
accuracyMean <- NULL
for(i in 1:nRepets){
	print(paste("Data", i))
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	FolderEnsemblePredictions <- paste0(FolderDataPartition, "/07_Result") 
	dir.create(FolderEnsemblePredictions, recursive = TRUE)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	print("meta-des original prediction and accuracy")
	namesBL <- paste0(modelName, "_", 1:nBL)
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	setwd(FolderOP)
	OP <- readRDS("OP_Fold4.rds")
	predClass <- probToClass(OP, namesBL, classesOfProblem)
	FolderKompetentnost_original <- paste0(FolderDataPartition, "/06_MetaPrediction_original/Kompetentnost")
	setwd(FolderKompetentnost_original)
	competence_original <- readRDS("Competence_original.rds")
	ensembleClass <- ensemblePrediction(method = "vote", predClass,	competence_original, threshold = thrshld)	
	FolderMETADES_original <- paste0(FolderEnsemblePredictions, "/MetaDes_original") 
	dir.create(FolderMETADES_original)
	setwd(FolderMETADES_original)
	saveRDS(ensembleClass, "ensemblePredictionOriginal_vote.rds")	
	a1 <- confusionMatrix(data$Fold4_Y, ensembleClass)$overall[[1]]
	accuracyOriginal <- c(accuracyOriginal, a1)
	print(a1)
	print("##############################################################################################")
	print("##############################################################################################")
	print("BL predictions and meta-des individual")
	FolderKompetentnost <- paste0(FolderDataPartition, "/06_MetaPrediction/Kompetentnost")
	setwd(FolderKompetentnost)
	competence <- readRDS("Competence.rds")
	ensembleClass_ind <- ensemblePrediction(method = "vote", predClass,	competence,	threshold = thrshld)
	FolderMETADES_individual <- paste0(FolderEnsemblePredictions, "/MetaDes_individual") 
	setwd(FolderPrediction)
	## add meta data about model
	saveRDS(ensembleClass_ind, "ensemblePrediction_vote.rds")	
	## compare predictions
	print("META_DES individual")
	a2 <- confusionMatrix(data$Fold4_Y, ensembleClass_ind)$overall[[1]]
	accuracyInd <- c(accuracyInd, a2)
	print(a2)
	print("Best BL")
	a3 <- max(sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y))
	accuracyBest <- c(accuracyBest, a3)
	print(a3)
	print("Mean of BL")	
	a4 <- mean(sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y))
	accuracyMean <- c(accuracyMean, a4)
	print(a4)
	print(sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y))
	print("##############################################################################################")
	print("##############################################################################################")
}
nameData <- paste0("Data_", 1:nRepets)
names(accuracyOriginal) <- nameData
names(accuracyInd) <- nameData
names(accuracyBest) <- nameData
names(accuracyMean) <- nameData
##
print(accuracyOriginal)
print(accuracyInd)
print(accuracyBest)
print(accuracyMean)
##
mean(accuracyOriginal)
mean(accuracyInd)
mean(accuracyBest)
mean(accuracyMean)



## knora

## ??

## meta-des.H

## greedy forward search

## best BL on valid set

## best BL

## oracle 



