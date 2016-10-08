## functions 


probToClass <- function(outputProfile, ## matrika napovedi z imeni stolpcev [for(i in bls){for(j in classes){bl(j)_class(i)}}]
											## npr: gbm_neg gbm_pos rf_neg rf_pos ...
						baseLearner, ## imena modelove ki nas zanimajo
						classes ## razredi, ki jih lahko napovedujemo
						)
	#################
	## funkcija matriko z napovedmi verjetnosti posameznega classa za
	## vsak base learner prevede v matriko napovedi vsakega base learnerja
{
	
	matrikaNapovedi <- data.frame(matrix(NA, nrow(outputProfile), length(baseLearner)))
	colnames(matrikaNapovedi) <- baseLearner
	for(bl in baseLearner){
		izberi <- paste0(bl, '_',classes)
		matrikaNapovedi[, bl] <- factor(classes[apply(outputProfile[,izberi],1 , which.max)], levels = classes)
		# matrikaNapovedi[, bl] <- classes[apply(outputProfile[,izberi],1 , which.max)]
	}
	return(matrikaNapovedi)
	
}



calculateAccuracy <- function(obs, predict, probBorder)
{	
	acc <- NULL
	for(p in probBorder){
		takeProb <- predict[predict > p | predict < 1-p]
		takeObs <- factor(obs[predict > p | predict < 1-p], levels(0,1))
		predClass <- factor(ifelse(takeProb > 0.5, 1, 0), levels = c(0,1))
		acc <- c(acc, confusionMatrix(predClass, takeObs)$overall[1])
	}
	
	names(acc) <- probBorder
	return(acc)
}



Save_dataPartition <- function(podatki_FM, ## feature matrix
								podatki_response, ## response vector
								k = 4, ## number of folds
								n = 1, ## number of times data is partition (for puropses of cross-validation)
								Folder = tempdir(), ## folder where to save data 
								File = "data", ## name of file 
								seedV = 123  ## seed used to create folds
								)
	## creates list of data seperated in k folds and returns data as list
{
	library(caret)
	set.seed(seedV)
	
	for(i in 1:n){
		podatki <- list()
		cvRazdelitev <- createFolds(onsovniPodatki_response, k = 4, list = TRUE, returnTrain = FALSE)
		for(ime in names(cvRazdelitev)){
			podatki <- c(podatki, list(onsovniPodatki_FM[cvRazdelitev[[ime]], ]))
			podatki <- c(podatki, list(onsovniPodatki_response[cvRazdelitev[[ime]]]))
		}
		names(podatki) <- paste0(rep(names(cvRazdelitev), each = 2), rep(c('_FM', '_Y'), 4))	
		saveFolder <- paste0(Folder, "/Partition_", i)
		dir.create(saveFolder, recursive = TRUE)
		nameFile <- paste0(File, "_", i, ".rds")
		setwd(saveFolder)
		if(nameFile %in% dir()){
			print(paste("model", nameFile, "already saved"))
		} else {
			saveRDS(podatki, paste0(saveFolder, "/", nameFile))
		}
		
	}
}



Save_baseLearner <- function(data_FM, # feature matrix
							data_response, ## response vector
							## model, ## name of caret model
							nModels = 10, ## number of linear models
							Folder = tempdir(), ## name of folder where saved
							File = "linearModel", ## name of file where saved
							seedV = 123 ## seed used
							)
	## trains linear base learner model and saves it (uses bagging from package adabag to train nModels) 
{	
	library(adabag)
	set.seed(seedV)
	dir.create(Folder, recursive = TRUE)
	
	pd <- cbind(y = data_response, data_FM)
	mdl <- bagging(y ~ ., data = pd, mfinal = nModels)

	# a <- mapply(saveRDS, mdl$trees, paste0(Folder, "/", File, "_", 1:nModels, ".rds"))
	for(i in 1:nModels){
		nameFile <- paste0(File, "_", i, ".rds")
		setwd(Folder)
		if(nameFile %in% dir()){
			print(paste("model", nameFile, "already saved"))
		} else {		
			saveRDS(mdl$trees[i], paste0(Folder, "/", nameFile))
		}
	}	

}



Save_OutputProfile <- function(featureMatrix, ## list of feature matrix
							models, ## list of (trained) model used to predict 
							Folder = tempdir(), ## name of folder where saved
							File = "OP_linearModel", ## name of file where saved
							seedV = 123 ## seed used
					)
	##  saves predictions of models in form:
	##	baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
{

	set.seed(seedV)
	dir.create(Folder, recursive = TRUE)
	OP <-  NULL
	for(mdl in models){
		OPofMdl <- predict(mdl, featureMatrix)
		OP <- cbind(OP, OPofMdl)
	}
	colnames(OP) <- c(t(outer(names(models), colnames(OPofMdl), paste, sep = '_')))
	## paste0(names(models), "_", rep(colnames(OPofMdl), length(models)))
	nameFile <- paste0(File, ".rds")
	setwd(Folder)
	if(nameFile %in% dir()){
		print(paste("output profile", nameFile, "already saved"))
	} else {
		saveRDS(OP, paste0(Folder, "/" ,nameFile))
	}
	
}



save_MetaProblem <- function(metaSet, ## FM of meta set 
							yMetaSet = FALSE, ## response vector of meta set 
											##[if false creates meta FM without meta response]
							neigbourSet, ## set from which neighbours are chosen
										## if neigbourSet = metaSet then neighbours of x from metaSet\x
							yNeigbourSet, ## response vector for neigbour set
							namesOfSet = c('trainBL','sosedSet', 'metaSet'), ## name of sets (used to save stuff)
							baseLearner, ## names of base learners 
							OP_meta, ## output profile of meta set
								## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
							OP_neigbours, ## output profile of neigbour set
								## column names: baseLearer1_class(1) baseLearner1_ class(2) ... baseLearner1_class(n) ... baseLearnerK_class(n)
							K = 5, ## number of neighbours on FM
							Kp = 5, ## number of neigbours on OP
							FolderNeigbour = tempdir(), ## Folder where neigbours are saved
							FolderMetaProblem = tempdir() ## Folder where meta problem is saved
							)
{	
	dir.create(FolderMetaProblem, recursive = TRUE)
	dir.create(FolderNeigbour, recursive = TRUE)
	
	
	
	{ ## input check 
	if(!(all(paste0(baseLearner, '_', levels(yNeigbourSet)) %in% colnames(OP_meta)))){
		stop(paste('noben base learner med output profili nima imena', baseLearner,'ali pa so classi v ySosedSet in OP_meta razlicni'))
	}
	
	## preveri ali so imena stolpcev output profilov enaki
	if(!(setequal(colnames(OP_meta), colnames(OP_neigbours)))){
				stop('output profili imajo razlicne imena stolpcev')
	}
	## preveri ce je dimenzija OP_meta enak dimenziji metaSet
	## more than one datapoint can have same OP, you cannot just cut out the first line if the value
	## and call it the knn of complement: have to check if it contains "itself" and remove it, otherwise cut out the last one!!!!
	if(nrow(metaSet) != nrow(OP_meta)){
				stop('metaSet in OP_meta imata razlicno stevilo vrstic!!')
	}
	}

	{ ## neigbours	
	neigboursName <- paste0('matrikaSosedje[trainBL]',namesOfSet[1] ,'[neigbourSet]',namesOfSet[2], 
						'[metaSet]', namesOfSet[3], '[K]',K, '[Dist]euclid.rds')
	neigboursNameOP <- paste0('matrikaSosedje_OP[trainBL]', namesOfSet[1],'[neigbourSet]',namesOfSet[2], 
						'[metaSet]',namesOfSet[3], '[Kp]',Kp, '[Dist]euclid.rds')

	## if metaSet = neigbourSet then k+1 and remove first line of neigbours (datapoint itself)
	
	setwd(FolderNeigbour)

	## region of competence 
	if(!(neigboursName %in% dir())){
		kNN_RC <- get.knnx(neigbourSet, metaSet,  k=K)
		matrika_sosediRC <- kNN_RC$nn.index
		saveRDS(matrika_sosediRC, neigboursName)
	}else{
		matrika_sosediRC <- readRDS(neigboursName)
	}
	
	## sosedi na output profile
	if(!(neigboursNameOP %in% dir())){
		kNN_OP <- get.knnx(OP_neigbours, OP_meta,  k=Kp)
		matrika_sosediOP <- kNN_OP$nn.index
		saveRDS(matrika_sosediOP, neigboursNameOP)
	}else{
		matrika_sosediOP <- readRDS(neigboursNameOP)
	}
	}
	
	
	## meta feauture matrix
	setwd(FolderMetaProblem)
	## for each base learner
	for (bl in baseLearner){
	
		nameMetaFM <- paste0('metaFM[BL]', bl,'[trainBL]', namesOfSet[1],'[sosedSet]',
							namesOfSet[2], '[sosedSet]', namesOfSet[3],'[K]', K,
							'[Kp]',Kp, '[Dist]euclid.rds')
		nameMetaProblem <- paste0('metaProblem[BL]', bl,'[trainBL]', namesOfSet[1],
								'[sosedSet]',namesOfSet[2], '[metaSet]',namesOfSet[3], 
								'[K]', K, '[Kp]',Kp, '[Dist]euclid.rds')
								
		parameters <- list(bl = bl, trainBL = namesOfSet[1],
							sosedSet = namesOfSet[2], metaSet = namesOfSet[3],
							K = K, Kp = Kp, Dist = 'euclid')
		## check if already calculated
		if(nameMetaFM %in% dir() & identical(yMetaSet,FALSE)){
			print(print(paste(nameMetaFM, 'JE ZE SHRANJEN!!')))
		} else if(nameMetaProblem %in% dir()){
			print(paste(nameMetaProblem, 'JE ZE SHRANJEN!!'))
		} else {
			metaFM <- NULL	
			## for each datapoint in metaSet 
			for (i in 1:nrow(metaSet)){
				if(i %% 1000 == 0){
					print(paste0(round(100*i/nrow(metaSet),2), '% matrike sestavljene'))
				}
				## neigbours from region of competences (RC)
				neigboursOnRC <- matrika_sosediRC[i, ]
				## neigbours from output profile (OP)
				neigboursOnOP <- matrika_sosediOP[i,]
				
				## predictions on neigbours
				nameBLclassification <- paste0(bl, '_', levels(yNeigbourSet))
				
				## region of competence neigbours
					## verjetnosti posameznega classa (sosedov iz region of competences)
					neigbours_probabilityRC <- OP_neigbours[neigboursOnRC, nameBLclassification]			
					## predicted class for neigbours on RC
					pClassRC <- factor(levels(yNeigbourSet)[apply(neigbours_probabilityRC, 1, which.max)], levels(yNeigbourSet))			
				## Output profile neigbours
					neigbours_probabilityOP <- OP_neigbours[neigboursOnOP, nameBLclassification]			
					## predicted class for neigbours on RC
					pClassOP <- factor(levels(yNeigbourSet)[apply(neigbours_probabilityOP, 1, which.max)], levels(yNeigbourSet))
				
				
				## meta featurji

				
				## FEATURE
				## f1: pravilna/napacna napoved baseLearner na region of competence (sosediRC)
				f1 <- as.numeric(pClassRC == yNeigbourSet[neigboursOnRC])

				## f2: verjetnosti na region of competence (sosediRC)
				f2 <- NULL
				for(i in 1:nrow(neigbours_probabilityRC)){
					f2 <- c(f2, neigbours_probabilityRC[i, yNeigbourSet[neigboursOnRC][i]])
				}	
				
				## f3: accuracy na celotni region of competence (sosediRC)
				f3 <- sum(as.numeric(pClassRC == yNeigbourSet[neigboursOnRC]))/length(neigboursOnRC)
				
				## f4: pravilno/napacna napoved baseLearner na sosedih po output profile 
				f4 <- as.numeric(pClassOP == yNeigbourSet[neigboursOnRC])
				
				## f5: razdalja med ...??

				metaFM <- rbind(metaFM, c(f1,f2,f3,f4))	
				colnames(metaFM) <- c(paste0('f1_', 1:length(f1)), paste0('f2_', 1:length(f2)), paste0('f3_', 1:length(f3)), paste0('f4_', 1:length(f4)))				
			}
			if(identical(yMetaSet,FALSE)){						
				saveRDS(metaFM, nameMetaFM)
				print(paste('shranila se je metaFM matrika:', nameMetaFM))
			}else{
				## response vector (1/0: bl predicted correctly/ bl wrong)
				nameBL <- paste0(bl, '_', levels(yNeigbourSet))
				## class prediction on neigbour set
				predictedClass_metaSet <- factor(levels(yMetaSet)[apply(OP_meta[, nameBL], 1, which.max)], levels(yNeigbourSet))				
				
				
				yMetaSet == predictedClass_metaSet
				metaClass <- as.numeric(yMetaSet == predictedClass_metaSet, levels = c(''))
				metaClass[metaClass == 0] <- 'incompetent'
				metaClass[metaClass == 1] <- 'competent'
				metaClass <- factor(metaClass)
				metaPrblm <- list('metaFM' = metaFM, 'metaClass' = metaClass, parameters = parameters)
				saveRDS(metaPrblm, nameMetaProblem)
				print(paste('shranila se je metaProblem list:', nameMetaProblem))
			}
		}
	}
}



save_mergeMetaProblems <- function(outputProfile,
									outputProfileName = "metaSet",
									namesBL,
									namesOfSet = c("trainBL", "sosedSet", "metaSet"),
									K,
									Kp,
									FolderMetaProblem,
									FolderSave = tempdir(),
									seedV = 123
									)
{
	set.seed(seedV)
	dir.create(FolderSave, recursive = TRUE)
	metaFM_1 <- NULL
	metaClass_1 <- NULL
	OP_meta_1 <- NULL
	setwd(FolderMetaProblem)
	use <- dir()[grepl(paste0(namesBL, collapse ="\\[|"),dir())]
	parametriModela <- paste0("\\[trainBL]", namesOfSet[1], "\\[sosedSet]", namesOfSet[2], "\\[metaSet]", namesOfSet[3], "\\[K]", K, "\\[Kp]", Kp, "\\[Dist]euclid")
	use <- use[grepl(parametriModela,use)]
	for(mtPrblm in use){
		setwd(FolderMetaProblem)
		metaProblem_indv <- readRDS(mtPrblm)
		metaFM_1 <- rbind(metaFM_1, metaProblem_indv$metaFM)
		metaClass_1 <- c(metaClass_1, as.character(metaProblem_indv$metaClass))
		setwd(FolderOP)
		OP_meta_1 <- rbind(OP_meta_1, outputProfile)
	}
	metaClass_1 <- factor(metaClass_1)
	parameters <- metaProblem_indv$parameters
	parameters$bl <- "ALL"
	parameters$Models <- namesBL
	metaProblem <- list(metaFM = metaFM_1, metaClass = metaClass_1, parameters = parameters)
	
	nameProblem <- paste0('metaProblem[BL]', parameters$bl,'[trainBL]', namesOfSet[1],
								'[sosedSet]',namesOfSet[2], '[metaSet]',namesOfSet[3], 
								'[K]', K, '[Kp]',Kp, '[Dist]euclid.rds')
	setwd(FolderSave)
	saveRDS(metaProblem, nameProblem)
	saveRDS(OP_meta_1, paste0("OF_", outputProfileName, "X", length(namesBL), ".rds"))	
}


save_trainMetaClassifier <- function(metaProblem, ## list of (metaFM, metaClass, parameters)
									OP_classMetaSet, ## output profile (class) za mnozico metaSet 
											## imena stolpcev: baseLearer1_predictedClass baseLearner1_predictedClass ... baseLearner1_predictedClass ... baseLearnerK_predictedClass									metaALG = c('rf', 'plr'), ## klasifikacijski algoritmi uporabljeni pri ucenju meta klasifikatorja
									metaALG = 'rf', ## algoritem used to train meta classifier
									hC = 0.95, ## kako razlicne morajo biti napovedi base learnerjev, za x iz metaSet da ga vkljucimo v ucenje
									FolderMetaClassifier = tempdir() ## okolje, kamor se shranijo meta klasifikatorji
									)
## trains meta classifier for meta problem and saves it in folder 
{
	library(caret)
	dir.create(FolderMetaClassifier, recursive = TRUE)
	setwd(FolderMetaClassifier)
	
	parameters <- metaProblem$parameters
	
	imeMetaKlasifikator <-  paste0('metaKlasifikator[BL]', parameters$bl, '[trainBL]', parameters$trainBL, 
								'[sosedSet]', parameters$sosedSet, '[metaSet]', parameters$metaSet, 
								'[K]',parameters$K, '[Kp]',parameters$Kp,'[Dist]',parameters$Dist,'[hC]', hC, '.rds')
	
	
	
	
	## select only cases where bl are divesre enought 
	blDisagreement <- apply(OP_classMetaSet, 1, function(x) max(table(x)))/ncol(OP_classMetaSet)	
	selection <- which(blDisagreement < hC)

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
		print(paste("Naucen in shranjen:", imeMetaKlasifikator))	
	}
}




save_Kompetentnost <- function(metaProblem, ## list of meta problems 
								metaClassifier, ## list of meta classifiers
								mappingProblemClassifier, ## named vector mapping meta problem to meta classifier
								Folder = tempdir(), ##
								File = "Competence", ##
								seedV = 123 ## seed used
								)
{
	dir.create(Folder, recursive = TRUE)
	set.seed(seedV)
	metaProblem <- testMetaProblem
	metaClassifier <- testMetaClassifier
	mappingProblemClassifier <- metaModelMap
	
	if(!(all(names(metaClassifier) %in% names(mappingProblemClassifier)) &
	all(names(mappingProblemClassifier) %in% mappingProblemClassifier))){
		print("names of problem or classifier do not match with names in mapping")
	}
	
	kompetentnost <- NULL
	for(prblm in names(mappingProblemClassifier)){
		kompetentnost <- cbind(kompetentnost, predict(metaClassifier[[mappingProblemClassifier[prblm]]], metaProblem[[prblm]]$metaFM, type = "prob")[,1])
	}
	colnames(kompetentnost) <- paste(names(metaProblem), "kompetentnost", sep = "_")
	saveRDS(kompetentnost, paste0(Folder, "/",File, ".rds"))
}



ensemblePrediction <- function(method = "vote", ## vote
								predClass,	
								competence,
								threshold = 0.8
								)
{
	## take x with highest competence
	
	incompetent <- competence < threshold
	
	competentPredictions <- predClass
	competentPredictions[incompetent] <- NA
	
	## if non competent, choose all
	competentPredictions[apply(is.na(competentPredictions),1, all),] <- predClass[apply(is.na(competentPredictions),1, all),]
	
	## randomly selected if tied
	vote <- apply(competentPredictions, 1, table)
	ensemblePrediction <- factor(sapply(vote, function(x) names(which(rank(-x, ties.method="random") == 1))))
	return(ensemblePrediction)
}




















