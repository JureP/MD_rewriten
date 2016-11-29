## functions for forward greedy search


library(nnet)



## Funkcija z baggingom
## okolje kjer sta skripti modelSelection_function.r in ProbEnsSelKrit_function.r

# OkoljeFunkcije <- 'Z:/JurePodlogar/Ensemble selection/Selection/Funkcije/Izboljsane'
# setwd(OkoljeFunkcije)
# source("modelSelection_function.r")
# source("ProbEnsSelKrit_function.r")



EnsembleSelection <- function(	X, # output profile (z imeni modelvo stoplca)
								Y, # true class
								namesBL, ## imena base learnerjev ki jih uporabimo
								iter = 5L, # stevilo elementov ensemble (z vracanjem)
								bagFrac = 1L, # delez (nakljucnih) base learnerjev 
											# iz katerih izbira (na iteraciji)
								podIter = 1L, # kolikokrat izbere model na podmnozici
								kriterij = c('accu', 'p/rF', 'rmse'), ## katere kriterije izbire
											## modelov naj uporabi:
												## accu: accuracy
												## p/rF: precision/recall F score
												## rmse: root mean square error
								dataBag = 1 ## bagging of data 
						){
	## funkcija iz base-learnerjev sestavi ensemble model tako da iter-krat 
	## nakljucno izbere bagFrac delez base-learnerjev in osnovnemu ensemblu
	## doda najboljsi ensemble (podIter-ih modelov) {glede na izobljsanje
	## osnovnega ensembla}
	
	if(!all(c(t(outer(namesBL, levels(Y), paste, sep = '_'))) %in% colnames(X))){
		print("namesBL niso vsi v X!!")
	}
		
	N <- length(namesBL)
	pred <- 0 * X
	weights <- rep(0L, N)
	names(weights) <- namesBL
	sumWeights <- 0L
	potekUtezi <- matrix(NA, iter, N)
	while (sumWeights < iter){
		print(sumWeights + 1)
		## izbor nakljucne podmnozice modelov
		sumWeights <- sumWeights + 1L
		bag <- sort(sample(namesBL, length(namesBL)*bagFrac))
		selected <- c(t(outer(bag, levels(Y), paste, sep = '_')))
		Xb <- X[ , selected]
		## ze sestavljenim utezmi weights, z tezo podIter*sumWeights
		
		prob <- NULL
		for(class in levels(Y)){
			prob <- cbind(prob, X[, grepl(paste0("_", class), colnames(X))] %*% weights)
		}
		colnames(prob) <- levels(Y)
		
		addWeights <- probEnsSelKrit(Xb, Y, namesBL[namesBL %in% bag], iter = podIter, prob, podIter*sumWeights,
									kriterij = kriterij, dataBag = dataBag)
		# addWeights <- probEnsSel(Xb, Y, iter = podIter, prob, podIter * sumWeights)
		## dodajanje utezi
		weights[bag] <- weights[bag] + addWeights
		potekUtezi[sumWeights, ] <- weights/sumWeights
		#print(weights)
	}
	
	return(potekUtezi)
	#return(weights / sumWeights)
}



# t1 <- Sys.time()
# ensemble <- EnsembleSelection(X,Y, iter =10L, bagFrac = 1, podIter = 1L, kriterij = c('accu', 'r/pF', 'rmse'))
# t2 <- Sys.time()
# tail(ensemble,1)
# t2 - t1





## Kriterij: abs error 
## Funkcija z baggingom
## okolje kjer so funkcije kriterijev za izbiro base learneja v ensemble


# OkoljeKriterij <- 'Z:/JurePodlogar/Ensemble selection/Selection/Funkcije/Izboljsane/Kriteriji'
# setwd(OkoljeKriterij)
# source("modelSelection_function.r")



probEnsSelKrit <- function( X, ## library of model predictions (matrix)
						Y, ## class (vector)
						namesBL, ## imena base leaarnerjev
						iter = 20L, # number of iterations
						prob = matrix(0, nrow = nrow(X), ncol = length(levels(Y))), # probability predictions from existing model
						sumWeights = 0, # weight of initWeights 
						kriterij = c('accu', 'p/rF', 'rmse'), ## katere kriterije izbire
											## modelov naj uporabi:
												## accu: accuracy
												## p/rF: precision/recall F score
												## rmse: root mean square error
						n = 2L, ## stevilo najboljsih modelov, ki dobi tocke
						nSk = c(2,1), ## lestvica tock (dolzine n)
						utezKriterija = c(1, 1, 1, 1), ## utez vsakega kriterija 
													## dolzine 4!!! 
													## vrstni red: accu, precision, p/rF, rmse
						dataBag = 1 ## bagging of data 
						){
  ## funkcija izvede iter iteracij dodajanja base learner modelov iz matrike X ensemble modelu, ki 
  ## je karakteriziran s pomocjo zacetnih verjetnostnih napovedi (prob) in teze le-teh (sumWeights)
  delPodatki <- sample(1:length(Y), dataBag * length(Y))
  X <- X[delPodatki,]
  Y <- Y[delPodatki]
  N           <- length(namesBL)
  weights     <- rep(0L, N)
  pred        <- prob * sumWeights
  niter       <- 0L
  while(niter < iter) {
	  #print(niter)
	  niter         <- niter + 1L
      sumWeights    <- sumWeights + 1L
		newPred <- NULL
		for (i in namesBL){
			newPred <- cbind(newPred, X[ , grepl(paste0(i, '_'), colnames(X))] + pred)
		}
		pred <- newPred/sumWeights
	  ### kriterij izbire 
	  tocke <- modelSelection(pred, Y, namesBL, n = n, nSk = nSk, uporabaModelov = kriterij, utezKriterija = utezKriterija)
	  best          <- which.is.max(tocke)
	  ###
      weights[best] <- weights[best] + 1L
      pred          <- pred[, grepl(paste0(namesBL[best], "_"), colnames(pred))] * sumWeights
	  #print(weights)
	  }
  return(weights / niter)
}

# t1 <- Sys.time()
# probEnsSelKrit(X,Y, iter = 10L, kriterij = c('accu','p/rF','rmse'))
# t2 <- Sys.time()
# t2-t1


## mesana kriterijska funkcija
## izbere najboljsi model po kriterijih z tezo:
 										# rmse (1/4??)
										# accu (1/2??) 
										# precRec (1/4??)
## Vsak kriterij da modelu tocke. Tocke dobi najboljsih
## n modelov


 
modelSelection <- function(pred, ## verjetnostne napovedi (matrika)
							Y,	## true class
							namesBL, ## imena base leaarnerjev
							n = 2, ## stevilo najboljsih modelov ki dobijo tocke
							nSk = c(2,1), ## vektor tock (dolzine n!!!)
							uporabaModelov = c('accu', 'precision', 'p/rF', 'rmse'), ## katere modele naj uporabi
											## accu: accuracy
											## precision: precision
											## p/rF: precision/recall F score
											## rmse: root mean square error
							utezKriterija = c(1, 1, 1, 1) ## utez vsakega kriterija 
													## dolzine 4!!! 
													## vrstni red: accu, precision, p/rF, rmse
						){
	## funkcija sprejme verjetnostne napovedi in true class in 
	## vrne kateri model daje najboljse napovedi po kriteriju 
	## glasovanja accu 1/2, rmse 1/4, precRecF 1/4
	
	##  accuracy in precision/recall
	napoved <- probToClass(pred, namesBL, levels(Y))	
	tocke <- rep(0, length(namesBL))
	## accuracy 
	if ('accu' %in% uporabaModelov){
		acc <- rep(NA, length(namesBL))
		for (i in 1:length(namesBL)){
			cm <- table(napoved[ , i], Y)
			acc[i] <- sum(diag(cm)) / sum(cm)
			# acc[i] <- (cm[1,1]  + cm[2,2])/sum(cm)
		}
		## tocke
		ac <- order(acc, decreasing = TRUE)
		tocke[ac[1:n]] = tocke[ac[1:n]] + utezKriterija[1]*nSk
	}
	## precision/recall F score  in precsion
	if ('p/rF' %in% uporabaModelov || 'precision' %in% uporabaModelov){
		precision <- rep(NA, length(namesBL))
		for (i in 1:length(namesBL)){
			cm <- table(napoved[ , i], Y)
			precision[i] <- mean(diag(cm) / rowSums(cm))
		}
		## precision
		if ('precision' %in% uporabaModelov){
			prec <- order(precision, decreasing = TRUE)
			tocke[prec[1:n]] = tocke[prec[1:n]] + utezKriterija[2]*nSk
		}
		## precision/recall F score
		if ('p/rF' %in% uporabaModelov){
			recall <- rep(NA, length(namesBL))
			for (i in 1:length(namesBL)){
				cm <- table(napoved[ , i], Y)
				recall[i] <- mean(diag(cm) / colSums(cm))
			}
			F1 <- 2 * precision * recall / (precision + recall)
			## tocke
			F <- order(F1, decreasing = TRUE)
			tocke[F[1:n]] = tocke[F[1:n]] + utezKriterija[3]*nSk
		}
	}
	## root mean squared error
	if ('rmse' %in% uporabaModelov){
		error <- rep(NA, length(namesBL))
		for(mdlNum in 1:length(namesBL)){
			A <- X[, grepl(paste0(namesBL[mdlNum], "_"), colnames(X))]
			modelError <- rep(NA, length(Y))
			for(i in 1:length(Y)){
				modelError[i] <- 1 - A[i,grepl(paste0('_', Y[i]), colnames(A))]
			}
			error[mdlNum] <- sqrt(sum(abs(modelError)))
		}
		rse <- order(error)
		tocke[rse[1:n]] = tocke[rse[1:n]] + utezKriterija[4]*nSk
	}
	
	## 
	
	return(tocke)	
}


# t1 <- Sys.time()
# rez <- modelSelection(X,Y, n = 2, rev(1:2), uporabaModelov = c('accu', 'p/rF'), utezKriterija = c(2,1,1))
# t2 <- Sys.time()
# t2-t1




PlattScaling <- function(	X, ## vektor vrejetnostnih napovedi
							Y ## true class
						)
	## funkcija vrne model za skaliranje podatkov
	## uporaba: predict(calibModel, newdata = Data[colname 'x'], type = 'response')					
	{
	calibDataFrame <- data.frame(cbind(Y, X))
	colnames(calibDataFrame) <- c("y", "x")
	calibModel <- glm(y ~ x, calibDataFrame, family=binomial)
	return(calibModel)
}




precisionPercentileMatirx <- function(	X , ## matrika (verjetnostnih) napoved
										Y,  ## true class
										percentil = seq(0.5,0.05,by = -0.01) ## percentil podatkov za
																			## katere naj izracuna prec.
						){
	## funkcija izracuna precision[percentil] za matriko s stolpci (verjentostnih) napovedi
	precisionMatrix <- data.frame(matrix(NA, length(percentil), ncol(X)))
	names(precisionMatrix) <- names(X)
	for(j in 1:ncol(X)){
		print(j)

		precPerc <- rep(NA, length(percentil))
		i <- 0
		for(q in percentil){
			i <- i + 1 
			precPerc[i] <- precisionPercentile(X[ , j], Ytest, q = q)
		}
		precisionMatrix[ ,j] <- precPerc
		#lines(precPerc, type = 'l', col = j)
	}
	return(precisionMatrix)
}


## precision/percentile funkcija

precisionPercentile <- function(napoved, ## vektor (verjetnostnih) napoved
								Y, ## vektor pravilnih prednosti
								q = 0.5 ## delez najvishih+najnizjih napovedi
								){
	## izracuna precsion na q-tem percentilu napovedi
	qL <- q/2
	qH <- 1 - q/2
	## spodnja meja vzetih napovedi
	spMeja <- quantile(napoved, qL)
	## zgornja meja vzetih napovedi
	zgMeja <- quantile(napoved, qH)
	
	Xizbor <- napoved[napoved > zgMeja | napoved < spMeja]
	Xizbor[Xizbor > zgMeja] <- 1
	Xizbor[Xizbor < spMeja] <- 0
	Yizbor <- Y[napoved > zgMeja | napoved < spMeja]
	retrived <- sum(Xizbor)
	precision <- sum(Xizbor & Yizbor) / retrived
	}


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


	
	
#######################################################################################################
#######################################################################################################
### WRAPP #############################################################################################


X <- readRDS("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\Pima\\Partition_1\\03_OutputProfile\\OP_Fold3.rds")
Y <- readRDS("C:\\Users\\jurep\\OneDrive\\Documents\\Koda\\MetaDes\\Podatki\\Pima\\Partition_1\\Pima_1.rds")$Fold3_Y
namesBL <- paste0("linearModel_", 1:20)
bagF <- 0.5
stIter <- 150
stPodIter <- 5
izbKrit <- c('accu', 'precision','p/rF', 'rmse')
dataBag = 0.7

namesBL <- sample(namesBL)

X1 <- NULL
for(i in namesBL){
	X1 <- cbind(X1, X[,grepl(paste0(i, '_'), colnames(X))])
}

X <- X1


t1 <- Sys.time()
ensemble <- EnsembleSelection(X, Y, namesBL, iter = stIter, bagFrac = bagF, podIter = stPodIter, kriterij = izbKrit, dataBag = 1)
t2 <- Sys.time()
tp <- t2 - t1


# A <- probToClass(X, namesBL, levels(Y))
# for(i in 1:ncol(A)){
	# print(sum(A[,i] == Y))
# }



# A == Y



## izbira parametrov funkcije EnsembleSelection + Platt scalint Yes/No 
izvedbaES <- function(	X, ## library of base-learners
						Y, ## true class
						scaled = FALSE, ## ali uporabi Platt scaling
						stIter = 100, ## stevilo iteracij
						bagF =  0.5, ## bagging fraction
						stPodIter = 5, ## st iteracij na nakljucno izbrani podmnozici
						izbKrit = c('accu', 'precision','p/rF', 'rmse') ## kriterij izbire 
	## funkcija izvede ensemble selection s funkcijo EnsembleSelection in uporabi parametre 
	## iter =stIter, bagFrac = bagF, podIter = stPodIter, kriterij = izbKrit
	## scaled = TRUE uporabi Platt scaling na verjetnostnih napovedih,
	## scaled = FALSE uporabi neskalirane napovedi	
		){
	
	## Platt scaling
	if (scaled == TRUE){
		Xcalib <- NA * X
		for(i in 1:ncol(X)){
			print(i)
			## select model on train set
			model <- PlattScaling(as.vector(X[,i]) ,Y)

			## use model on test set
			dataTest <- data.frame(X[,i])
			names(dataTest) <- 'x'
			Xcalib[ , i] <- predict(model, newdata = dataTest, type = 'response')
		}
	}


	## Brez: Platt scaling
	if(scaled == FALSE){
		t1 <- Sys.time()
		ensemble <- EnsembleSelection(X, Y, iter =stIter, bagFrac = bagF, podIter = stPodIter, kriterij = izbKrit)
		t2 <- Sys.time()
		tp <- t2 - t1
	}
	## S: Platt scaling
	if (scaled == TRUE){
		t1 <- Sys.time()
		ensemble <- EnsembleSelection(Xcalib, Y, iter =stIter, bagFrac = bagF, podIter = stPodIter, kriterij = izbKrit)
		t2 <- Sys.time()
		tp <- t2 - t1
	}

	ensemble <- data.frame(ensemble)
	names(ensemble) <- names(X)

	#finalEnsemble <- as.vector(tail(ensemble,1))
	#names(finalEnsemble) <- names(X)
	#plot(finalEnsemble)
	
	return(ensemble)
}







