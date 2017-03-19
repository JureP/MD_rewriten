
## source("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Koda\\Pima.r")
## funkcije:
source("C:/Users/jurep/Documents/Koda/Magisterska/MetaDes_Test/Funckije.r")
source("C:/Users/jurep/Documents/Koda/Magisterska/MetaDes_Test/Funkcije_FGS.r")
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
nBL <- 20 ## number of baseLearners
K <- 5
Kp <- 5
metaALG <- "rf"
hC <- 0.75
thrshld <- 0.8

## parameters 
modelName <- "linearModel"
namesBL <- paste0(modelName, "_", 1:nBL)
FileData <- 'Pima'
FolderData <- "C:/Users/jurep/Documents/Koda/Magisterska/MetaDes_Test/Podatki/Pima"
dir.create(FolderData, recursive = FALSE)
# FolderBL <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/Pima/02_BaseLearner"
# FolderOP <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/03_OutputProfile"

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
	## namesBL <- paste0(modelName, "_", 1:nBL)
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



## comparison of model accuracy
accuracyOriginal <- NULL
accuracyInd <- NULL
accuracyBest <- NULL
accuracyBestValid <- NULL
accuracyMean <- NULL
for(i in 1:nRepets){
	print(paste("Data", i))
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	FolderEnsemblePredictions <- paste0(FolderDataPartition, "/07_Result") 
	dir.create(FolderEnsemblePredictions, recursive = TRUE)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	print("meta-des original prediction and accuracy")
	## namesBL <- paste0(modelName, "_", 1:nBL)
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	setwd(FolderOP)
	OP <- readRDS("OP_Fold4.rds")
	predClass <- probToClass(OP, namesBL, classesOfProblem)
	FolderKompetentnost_original <- paste0(FolderDataPartition, "/06_MetaPrediction_original/Kompetentnost")
	setwd(FolderKompetentnost_original)
	competence_original <- readRDS("Competence_original.rds")
	ensembleClass <- ensemblePrediction(#method = "vote",
										method = "weighted",
										OutputProfile = OP,
										namesBL,
										classesOfProblem = levels(data$Fold1_Y),
										competence = competence_original,
										threshold = thrshld)	
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
	ensembleClass_ind <- ensemblePrediction(#method = "vote", 
											method = "weighted",	
											OutputProfile = OP,
											namesBL,
											classesOfProblem = levels(data$Fold1_Y),
											competence = competence,
											threshold = thrshld)	
	FolderMETADES_individual <- paste0(FolderEnsemblePredictions, "/MetaDes_individual") 
	setwd(FolderPrediction)
	## add meta data about model
	saveRDS(ensembleClass_ind, "ensemblePrediction_vote.rds")	
	
	## selecting best model on validation set
	## namesBL <- paste0(modelName, "_", 1:nBL)
	## BLoneValidSet <- rbind(data$Fold2_FM, data$Fold3_FM)
	yBLoneValidSet <- factor(c(as.character(data$Fold2_Y), as.character(data$Fold3_Y)))
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")	
	setwd(FolderOP)
	OP_validBL <- rbind(readRDS("OP_Fold2.rds"), readRDS("OP_Fold3.rds"))
	predClassVALID <- probToClass(OP_validBL, namesBL, classesOfProblem)	
	blAccuracy <- sapply(predClassVALID, function(x, y) confusionMatrix(x, y)$overall[[1]], yBLoneValidSet)
	bestBLOnValid <- which.max(blAccuracy)	
	
	
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
	print("Best BL on valid")
	a5 <- sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], data$Fold4_Y)[bestBLOnValid]
	accuracyBestValid <- c(accuracyBestValid, a5)
	print(a5)
	print("##############################################################################################")
	print("##############################################################################################")
}
nameData <- paste0("Data_", 1:nRepets)
names(accuracyOriginal) <- nameData
names(accuracyInd) <- nameData
names(accuracyBest) <- nameData
names(accuracyBestValid) <- nameData
names(accuracyMean) <- nameData
##
## originalni meta des
print(accuracyOriginal)
## meta des s individualnim meta klasifikatorjem
print(accuracyInd)
## 
## accuracy of best BL
print(accuracyBest)
## accuracy of best BL on the validation set
print(accuracyBestValid)
print(accuracyMean)
##
mean(accuracyOriginal)
mean(accuracyInd)
mean(accuracyBest)
mean(accuracyBestValid)
mean(accuracyMean)
## variance
var(accuracyOriginal)
var(accuracyInd)
var(accuracyBest)
var(accuracyBestValid)
var(accuracyMean)




## LA_OLA (ACCURACY ON FM NEIGBOURS)
## do a parameter tunning on a different set
## do a parameter tunning on a different set

for(i in 1:nRepets){
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	FolderPrediction <- paste0(FolderDataPartition, "/06_MetaPrediction")
	FolderMetaProblem <- paste0(FolderPrediction, "/MetaProblem_prediction")
	FolderNeigbour <- paste0(FolderMetaProblem, "/Neigbours")
	
	setwd(FolderOP)
	OP_test <- readRDS("OP_Fold4.rds")
	OP_neigbours <- readRDS("OP_Fold3.rds")

	setwd(FolderNeigbour)
	namesOfSet <- c("Fold1", "Fold3", "Fold4")
	fileSosedje <- paste0('matrikaSosedje[trainBL]',namesOfSet[1] ,'[neigbourSet]',namesOfSet[2], 
						'[metaSet]', namesOfSet[3], '[K]',K, '[Dist]euclid.rds')
	
	FolderLA_OLA <- paste0(FolderDataPartition, "/08_ComparisonModels/LA_OLA")
	
	
	LocClasAcc(OP_test, ## Matrix of base learner prediction
				OP_neigbours, ## Matrix of neigours in neigbour set
				yNeigbourSet, ## class of cases in neighbour set
				matrikaSosedje, ## kNN sosedje glede na feature matrix
				namesBL, ## list of meta classifiers
				FolderLA_OLA, ##
				File = "Competence_LAOLA", ##
				seedV = 123 ## seed used
							)
	
	setwd(Folder)
	Competence_LAOLA <- readRDS("Competence_LAOLA.rds")
				
	a <- ensemblePrediction(method = "weighted", 
							OutputProfile = OP_test, 
							namesBL,
							classesOfProblem, 
							competence = Competence_LAOLA, 
							threshold = 0)

}
## ACCURACY ON OP NEIGHBOUR



## Meta learning

accuracyMetaLearner <- NULL
for(i in 1:nRepets){
	print(paste("meta learning, partition", i, "out of", nRepets))
	metaALG <- 'rf'
	
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	FolderComparisonMethods <- paste0(FolderDataPartition, "/08_ComparisonModels")
	dir.create(FolderComparisonMethods)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	## base-learner library
	## X output profile ## TO BE ADOPTED FOR USE WITH MULTIPLE CLASSES
	## USE probabilityToClass in modelSelection
	## adopt bagging so that is selects all the probablities from one base learner and sends it to the next stage
	## Y respose vector
	## test this only on two class problems
	yBLoneValidSet <- factor(c(as.character(data$Fold2_Y), as.character(data$Fold3_Y)))
	setwd(FolderOP)
	OP_validBL <- rbind(readRDS("OP_Fold2.rds"), readRDS("OP_Fold3.rds"))
	
	metaModel <- train(OP_validBL, yBLoneValidSet, method = metaALG)

	## predict 

	setwd(FolderOP)
	OP_test <- readRDS("OP_Fold4.rds")
	Ytest <- data$Fold4_Y
	predictionMetaaLearner <- predict(metaModel, OP_test)
	
	aMetaLearner <- confusionMatrix(Ytest, predictionMetaaLearner)$overall[[1]]
	accuracyMetaLearner <- c(accuracyMetaLearner, aMetaLearner)

}

## meta-des.H

## greedy forward search ####################################################################
#############################################################################################
#############################################################################################


rbind(accuracyInd, accuracyOriginal, accuracyMetaLearner, accuracyBest)



accuracyFGS <- NULL
for(i in 1:nRepets){
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	FolderComparisonMethods <- paste0(FolderDataPartition, "/08_ComparisonModels")
	dir.create(FolderComparisonMethods)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	## base-learner library
	## X output profile ## TO BE ADOPTED FOR USE WITH MULTIPLE CLASSES
	## USE probabilityToClass in modelSelection
	## adopt bagging so that is selects all the probablities from one base learner and sends it to the next stage
	## Y respose vector
	## test this only on two class problems
	setwd(FolderOP)
	X <- readRDS("OP_Fold3.rds")
	Y <- data$Fold3_Y
	## izbira parametrov (gelje EnsembleSelection_funtion.r (razen Platt scaling))
	scaled = FALSE ## ali uporabi Platt scaling
	stIter <- 200 ## stevilo iteracij
	bagF <- 0.3 ## bagging fraction
	stPodIter <- 5 ## st iteracij na nakljucno izbrani podmnozici
	izbKrit <- cizbKrit <- c('accu', 'precision','p/rF', 'rmse')
	dataBag <- 0.7

	## namesBL <- paste0(modelName, "_", 1:nBL)
# bagF <- 0.5
# stIter <- 150
# stPodIter <- 5
# izbKrit <- c('accu', 'precision','p/rF', 'rmse')
	
	
	ensemble <- EnsembleSelection(X, Y, namesBL, iter = stIter, bagFrac = bagF, podIter = stPodIter, kriterij = izbKrit, dataBag = dataBag)
	setwd(FolderComparisonMethods)
	saveRDS(ensemble, paste0("FGS_ensemble", stIter, "_", bagF, "_", stPodIter, "_", dataBag, ".rds"))
	
	ensembleLast <- tail(readRDS(paste0("FGS_ensemble",stIter, "_", bagF, "_", stPodIter, "_", dataBag, ".rds")),1)
	
	## obravavaj kot nedinamična kompetentnost 
	FGS_kompetentnost <- matrix(ensembleLast, nrow=nrow(OP_test), ncol=length(ensembleLast), byrow=TRUE)
	colnames(FGS_kompetentnost) <- paste0(namesBL, "_kompetentnost")
	
	## napoved 
	setwd(FolderOP)
	OP_test <- readRDS("OP_Fold4.rds")
	Ytest <- data$Fold4_Y

	
	FGS_prediction <- ensemblePrediction(method = "weighted", 
							OutputProfile = OP_test, 
							namesBL,
							classesOfProblem, 
							competence = FGS_kompetentnost, 
							threshold = 0)
	
	aFGS <- confusionMatrix(Ytest, FGS_prediction)$overall[[1]]
	accuracyFGS <- c(accuracyFGS, aFGS)
	## print(accuracyFGS)

	sapply(probToClass(X,namesBL, classesOfProblem), function(x, y) confusionMatrix(x, y)$overall[[1]], Y)
	
	}

## best BL


{## best BL on valid set (already done above!!)

for(i in 1:nRepets){
	## accuracy modelov na validacijski mnozici ()
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	
	## namesBL <- paste0(modelName, "_", 1:nBL)
	## BLoneValidSet <- rbind(data$Fold2_FM, data$Fold3_FM)
	yBLoneValidSet <- factor(c(as.character(data$Fold2_Y), as.character(data$Fold3_Y)))
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")	
	setwd(FolderOP)
	OP_validBL <- rbind(readRDS("OP_Fold2.rds"), readRDS("OP_Fold3.rds"))
	predClass <- probToClass(OP_validBL, namesBL, classesOfProblem)	
	## accuracy of all models
	blAccuracy <- sapply(predClass, function(x, y) confusionMatrix(x, y)$overall[[1]], yBLoneValidSet)
	bestOnValid <- which.max(blAccuracy)
	
	blAccuracy[bestOnValid]

	
	
	
	yBLoneValidSet <- factor(c(as.character(data$Fold2_Y), as.character(data$Fold3_Y)))
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")	
	setwd(FolderOP)
	OP_validBL <- rbind(readRDS("OP_Fold2.rds"), readRDS("OP_Fold3.rds"))
	predClassVALID <- probToClass(OP_validBL, namesBL, classesOfProblem)	
	blAccuracy <- sapply(predClassVALID, function(x, y) confusionMatrix(x, y)$overall[[1]], yBLoneValidSet)
	bestBLOnValid <- which.max(blAccuracy)	
	
}






}

## oracle 
accuracyOracle <- NULL
for(i in 1:nRepets){
	FolderDataPartition <- paste0(FolderData, "/Partition_", i)
	FolderComparisonMethods <- paste0(FolderDataPartition, "/08_ComparisonModels")
	##dir.create(FolderComparisonMethods)
	setwd(FolderDataPartition)
	data <- readRDS(paste0(FileData, "_", i, ".rds"))
	
	FolderOP <- paste0(FolderDataPartition, "/03_OutputProfile")
	## base-learner library
	## X output profile ## TO BE ADOPTED FOR USE WITH MULTIPLE CLASSES
	## USE probabilityToClass in modelSelection
	## adopt bagging so that is selects all the probablities from one base learner and sends it to the next stage
	## Y respose vector
	## test this only on two class problems
	setwd(FolderOP)
	X <- readRDS("OP_Fold3.rds")
	Y <- data$Fold3_Y
	
	
	napovediBL <- probToClass(X, namesBL, classesOfProblem)	
	PravilnostNapovediBL <- apply(napovediBL, 2, "==", Y)
	oracle <- apply(PravilnostNapovediBL, 1, any)
	aOracle <- sum(oracle)/length(oracle)
	accuracyOracle  <- c(accuracyOracle, aOracle)
	##print(accuracyOracle)
}

print(mean(accuracyOracle))
print(var(accuracyOracle))


## comparison all
allAcc <- rbind(accuracyMean, accuracyBestValid, accuracyInd, accuracyOriginal, accuracyMetaLearner, accuracyBest, accuracyOracle)


allAcc <- cbind(allAcc, mean = apply(allAcc, 1, mean))


library(ggplot2)
library(RColorBrewer)  
library(reshape2)  
allAcc <- allAcc[-nrow(allAcc),]
accMelt <- melt(allAcc)
ggplot(accMelt, aes(x = Var1, y = Var2, fill = value)) + geom_tile()

