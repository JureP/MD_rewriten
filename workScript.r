##################################################################################################
##################################################################################################
##################################################################################################


## structure
	## partition data (DONE)
	## base learner training (DONE)
	## Output progile  (DONE)
		## (one matrix with: probability for each class and each bl)
	## Meta problem (DONE)
		## looking for neigbours on complelment of set
	## Meta classificator training 
	## Ensemble prediction
	## prediction evaluation
		## compare against oracle
		## compare agains select random classfier prediction


##################################################################################################
##################################################################################################
##################################################################################################

## funkcije:
source("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Koda/BackEnd/Funkcije.r")
library(mlbench)
library(dplyr)
library(caret)
data(PimaIndiansDiabetes)
library(FNN)

## razdelitev podatkov 

osnovniPodatki <- na.omit(PimaIndiansDiabetes)
onsovniPodatki_response <- osnovniPodatki[, 9]
onsovniPodatki_FM <- osnovniPodatki[, -9]

## adjustable parameters 
nBL <- 3 ## number of baseLearners
nRepets <- 4 ## number of cross validations


## parameters 
FolderData <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/01_Podatki"
FolderBL <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/02_BaseLearner"
FolderOP <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/03_OutputProfile"
dir.create(FolderData)
dir.create(FolderBL)
dir.create(FolderOP)
classesOfProblem <- levels(onsovniPodatki_response)


## data partition
Save_dataPartition(onsovniPodatki_FM, onsovniPodatki_response, k = 4, n = nRepets, Folder = FolderData)
## base learner training
Save_baseLearner(onsovniPodatki_FM, onsovniPodatki_response, nModels = nBL, Folder = FolderBL)
## base learner prediction
featureMatrix <- readRDS(paste0(FolderData, "/data_1.rds"))[[c("Fold2_FM")]]
namesBL <- paste0("linearModel_", 1:nBL)
load <- paste0(FolderBL, "/", namesBL, ".rds")
models <- lapply(load, readRDS)
names(models) <- namesBL
Save_OutputProfile(featureMatrix, models, Folder = FolderOP, File = "OP_Fold2")
## meta learner: feature matrix 
data <- readRDS(paste0(FolderData, "/data_1.rds"))
metaSet <- data[[1]]
yMetaSet <- data[[2]]
##yMetaSet = FALSE
neigbourSet <- data[[1]]
yNeigbourSet <- data[[2]]
OP_meta <- readRDS("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/03_OutputProfile/OP_Fold2.rds")
OP_neigbours <- OP_meta
FolderNeigbour <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/04_MetaProblem/neighbours"
FolderMetaProblem <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/04_MetaProblem"
baseLearner <- namesBL
namesOfSet <- c('trainBL','sosedSet', 'metaSet')
K <- 5
Kp <- 5
save_MetaProblem(metaSet, ## FM of meta set 
				yMetaSet, ## response vector of meta set 
								##[if false creates meta FM without meta response]
				neigbourSet, ## set from which neighbours are chosen
							## if neigbourSet = metaSet then neighbours of x from metaSet\x
				yNeigbourSet, ## response vector for neigbour set
				namesOfSet, ## name of sets (used to save stuff)
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
## meata learning
	## meta learning calls function save_MetaProblem (NO)
metaProblem <- readRDS("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/04_MetaProblem/metaProblem[BL]linearModel_1[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]5[Kp]5[Dist]euclid.rds")	
OP_classMetaSet <- probToClass(OP_meta, namesBL, levels(yMetaSet))
metaALG <- 'rf'
hC <- 0.95
FolderMetaClassifier <- "C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/05_MetaClassifier"
save_trainMetaClassifier(metaProblem = metaProblem, OP_classMetaSet = OP_classMetaSet,
						metaALG = metaALG, hC = hC, FolderMetaClassifier = FolderMetaClassifier)


save_metaPrediction <- function(metaClassifier, ##
								metaProblem							
								)
{
	
	metaModel <- readRDS("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\05_MetaClassifier\\metaKlasifikator[BL]linearModel_1[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]5[Kp]5[Dist]euclid[hC]0.95.rds")
	metaProblem <- readRDS("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\04_MetaProblem\\metaProblem[BL]linearModel_1[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]5[Kp]5[Dist]euclid.rds")
	A <- predict(metaModel, metaProblem$metaFM)
	
}
	
	
	
	
ucenjeMetaKlasifikator <- function(metaProblem, ## list of (metaFM, metaClass, parameters)
									OP_classMetaSet, ## output profile (class) za mnozico metaSet 
											## imena stolpcev: baseLearer1_predictedClass baseLearner1_predictedClass ... baseLearner1_predictedClass ... baseLearnerK_predictedClass									metaALG = c('rf', 'plr'), ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
									hC = 0.95, ## kako razlicne morajo biti napovedi base learnerjev, za x iz metaSet da ga vkljucimo v ucenje
									OkoljeMetaKlasifikator ## okolje, kamor se shranijo meta klasifikatorji
									)
## 
{
	## to be deleted
	metaProblem <- readRDS("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/04_MetaProblem/metaProblem[BL]linearModel_1[trainBL]trainBL[sosedSet]sosedSet[metaSet]metaSet[K]5[Kp]5[Dist]euclid.rds")
	OP_meta <- readRDS("C:/Users/jurep/OneDrive/Documents/Koda/MetaDes/Podatki/03_OutputProfile/OP_Fold2.rds")
	
	OP_classMetaSet <- probToClass(OP_meta, namesBL, levels(yMetaSet))
	
	metaALG <- 'rf'
	
	
	hC <- 0.95	
	
	## not to be deleted 
	library(caret)
	setwd(OkoljeMetaKlasifikator)
	
	parameters <- metaProblem$parameters
	
	imeMetaKlasifikator <-  paste0('metaKlasifikator[BL]', parameters$bl, '[trainBL]', parameters$trainBL, 
								'[sosedSet]', parameters$sosedSet, '[metaSet]', parameters$metaSet, 
								'[K]',parameters$K, '[Kp]',parameters$Kp,'[Dist]',parameters$Dist,'[hC]', hC, '.rds')
	
	
	
	
	## select only cases where bl are divesre enought 
	blDisagreement <- apply(OP_classMetaSet, 1, function(x) max(table(x)))/ncol(OP_classMetaSet)	
	selection <- which(blDisagreement > hC)

	if(length(selection) < 50){
		print(paste('Samo', length(selection), 'primerov iz metaSet-a ima dovolj razlicne napovedi (hC)!!!'))
	}
	if(is.null(selection)){
		stop('noben primer ne preseze praga razlicnosti napovedi hC!!')
	}

	metaFM <- metaProblem$metaFM[selection, ]
	metaClass <- metaProblem$metaClass[selection]

	if(imeMetaKlasifikator %in% dir()){
		print(paste('ZE NAUCEN:', imeMetaKlasifikator))	
	}
	else {
		metaModel <- train(metaFM, metaClass, method = metaALG)
		saveRDS(metaModel, imeMetaKlasifikator)
	}
}

