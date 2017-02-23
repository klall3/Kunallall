## setting working directory
#dir <- 'D:/Notes/SCM/Project'
#setwd(dir) #sets the working directory to dir path


## loading libraries
library(caret)
library(dummies)
library(plyr)


## loading data 
train_set <- read.csv("D:/Notes/SCM/Project/Data/train.csv", stringsAsFactors=F)
test_set <- read.csv("D:/Notes/SCM/Project/Data/test.csv", stringsAsFactors=F)


## cleaning data
main_data_set <- rbind(train_set[,-ncol(train_set)], test_set) #combines train_set and test_set data frame by rows

# creating feature variables
main_data_set$year <- substr(as.character(main_data_set$Open.Date),7,10) # extracting year from Open Date column
main_data_set$month <- substr(as.character(main_data_set$Open.Date),1,2) # extracting month from Open Date column
main_data_set$day <- substr(as.character(main_data_set$Open.Date),4,5) # extracting day from Open Date column

main_data_set$Date <- as.Date(strptime(main_data_set$Open.Date, "%m/%d/%Y")) # converts character representations of date

main_data_set$days <- as.numeric(as.Date("2014-02-02")-main_data_set$Date) # converts date to numeric value 
main_data_set$days <- as.factor(main_data_set$days) # converts date to factor 

main_data_set$City.Group <- as.factor(main_data_set$City.Group) #coerces city group to factor

main_data_set$Type[main_data_set$Type == "DT"] <- "IL" 
main_data_set$Type[main_data_set$Type == "MB"] <- "FC"
main_data_set$Type <- as.factor(main_data_set$Type) #coerces type to a factor

main_data_set <- subset(main_data_set, select = -c(Open.Date, Date, City))	# returns subsets of main_data_set combining date and city

# converting some categorical variables into dummies
main_data_set <- dummy.data.frame(main_data_set, names=c("P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23", "P24", "P25", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"), all=T) 

## finds unique value in each column
ldf <- lapply(1:ncol(main_data_set), function(k)
				{
					return(data.frame("column" = colnames(main_data_set)[k],
									  "unique" = length(unique(main_data_set[1:nrow(train_set),k]))))
				})

ldf <- ldply(ldf, data.frame)

# removes variables with unique values
main_data_set <- main_data_set[,!names(main_data_set) %in% ldf$column[ldf$unique == 1]]

# removes highly correlated variables
for (i in (6:ncol(main_data_set)))
{
	main_data_set[,i] <- as.numeric(main_data_set[,i])
}

cor <- cor(main_data_set[1:nrow(train_set), 6:ncol(main_data_set)])

high_cor <- findCorrelation(cor, cutoff = 0.99)

high_cor <- high_cor[high_cor != 186]

main_data_set <- main_data_set[,-c(high_cor+5)]

# splitting into train_set and test_set
X_train <- main_data_set[1:nrow(train_set),-1]
X_test <- main_data_set[(nrow(train_set)+1):nrow(main_data_set),]

# building model on log of revenue
result <- log(train_set$revenue)


## Random Forest
source("D:/Notes/SCM/Code/RandomForest_N.R")

# 5-fold cross validation and scoring
model_rf_1 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=235,metric="rmse")
model_rf_2 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=357,metric="rmse")
model_rf_3 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=13,metric="rmse")
model_rf_4 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=753,metric="rmse")
model_rf_5 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=532,metric="rmse")


## submission
test_rf_1 <- model_rf_1[[2]]
test_rf_2 <- model_rf_2[[2]]
test_rf_3 <- model_rf_3[[2]]
test_rf_4 <- model_rf_4[[2]]
test_rf_5 <- model_rf_5[[2]]

submit <- data.frame("Id" = test_rf_1$Id,
					 "Prediction" = 0.2*exp(test_rf_1$pred_rf) + 0.2*exp(test_rf_2$pred_rf) + 0.2*exp(test_rf_3$pred_rf) + 0.2*exp(test_rf_4$pred_rf) + 0.2*exp(test_rf_5$pred_rf))

write.csv(submit, "D:/Notes/SCM/submit.csv", row.names=F)
findCorrelation <- function(x, cutoff = 0.90, verbose = FALSE, names = FALSE, exact = ncol(x) < 100) {
  if(names & is.null(colnames(x)))
    stop("'x' must have column names when `names = TRUE`")
  out <- if(exact) 
    findCorrelation_exact(x = x, cutoff = cutoff, verbose = verbose) else 
      findCorrelation_fast(x = x, cutoff = cutoff, verbose = verbose)
  out
  if(names) out <- colnames(x)[out]
  out
}
findCorrelation_fast <- function(x, cutoff = .90, verbose = FALSE){
  if(any(!complete.cases(x)))
    stop("The correlation matrix has some missing values.")
  averageCorr <- colMeans(abs(x))
  averageCorr <- as.numeric(as.factor(averageCorr))
  x[lower.tri(x, diag = TRUE)] <- NA
  combsAboveCutoff <- which(abs(x) > cutoff)
  
  colsToCheck <- ceiling(combsAboveCutoff / nrow(x))
  rowsToCheck <- combsAboveCutoff %% nrow(x)
  
  colsToDiscard <- averageCorr[colsToCheck] > averageCorr[rowsToCheck]
  rowsToDiscard <- !colsToDiscard
  
  if(verbose){
    colsFlagged <- pmin(ifelse(colsToDiscard, colsToCheck, NA),
                        ifelse(rowsToDiscard, rowsToCheck, NA), na.rm = TRUE)
    values <- round(x[combsAboveCutoff], 3)
    cat('\n',paste('Combination row', rowsToCheck, 'and column', colsToCheck,
                   'is above the cut-off, value =', values,
                   '\n \t Flagging column', colsFlagged, '\n'
    ))
  }
  
  deletecol <- c(colsToCheck[colsToDiscard], rowsToCheck[rowsToDiscard])
  deletecol <- unique(deletecol)
  deletecol
}
findCorrelation_exact <- function(x, cutoff = 0.90, verbose = FALSE)
{
  varnum <- dim(x)[1]
  
  if (!isTRUE(all.equal(x, t(x)))) stop("correlation matrix is not symmetric")
  if (varnum == 1) stop("only one variable given")
  
  x <- abs(x)
  
  # re-ordered columns based on max absolute correlation
  originalOrder <- 1:varnum
  
  averageCorr <- function(x) mean(x, na.rm = TRUE)
  tmp <- x
  diag(tmp) <- NA
  
  maxAbsCorOrder <- order(apply(tmp, 2, averageCorr), decreasing = TRUE)
  x <- x[maxAbsCorOrder, maxAbsCorOrder]
  newOrder <- originalOrder[maxAbsCorOrder]
  rm(tmp)
  
  deletecol <- rep(FALSE, varnum)
  
  x2 <- x
  diag(x2) <- NA
  
  for (i in 1:(varnum - 1)) {
    if(!any(x2[!is.na(x2)] > cutoff)){
      if (verbose) cat("All correlations <=", cutoff, "\n")
      break()
    }
    if (deletecol[i]) next
    for (j in (i + 1):varnum) {
      if (!deletecol[i] & !deletecol[j]) {
        
        if (x[i, j] > cutoff) {
          mn1 <- mean(x2[i,], na.rm = TRUE)
          mn2 <- mean(x2[-j,], na.rm = TRUE)
          if(verbose) cat("Compare row", newOrder[i], 
                          " and column ", newOrder[j], 
                          "with corr ", round(x[i,j], 3), "\n")  
          if (verbose) cat("  Means: ", round(mn1, 3), "vs", round(mn2, 3))
          if (mn1 > mn2) {
            deletecol[i] <- TRUE
            x2[i, ] <- NA
            x2[, i] <- NA
            if (verbose) cat(" so flagging column", newOrder[i], "\n")
          }
          else {
            deletecol[j] <- TRUE
            x2[j, ] <- NA
            x2[, j] <- NA
            if (verbose) cat(" so flagging column", newOrder[j], "\n")
          }
        }
      }
    }
  }
  newOrder[which(deletecol)]
}
