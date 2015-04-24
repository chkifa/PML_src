library(caret)
library(corrplot)
source("pml_write_files.R")

# Load full dataset
dataPath = "../data";
fullDataName = "pml-training.csv";
testingDataName = "pml-testing.csv";

fullDataPath = paste(dataPath, fullDataName, sep = "/");
testingDataPath = paste(dataPath, testingDataName, sep = "/");

fullData = read.csv(fullDataPath, na.strings=c("NA", "#DIV/0!"));
finalTestingData = read.csv(testingDataPath);

# Set the seed for results reproducibility
set.seed(7)

# Split the training dataset into training and testing sub-datasets 
# over the feature "classe" (60%, 40% medium dataset).
# Let's reserve the training dataset for the very last/final test (20 tests to be submitted)
inDataSet <- createDataPartition(fullData$classe, p = .60, list = FALSE)
trainingSet <- fullData[inDataSet,]
testingSet <- fullData[-inDataSet,]

# From now on we use trainingSet to train the model and the testingSet to test it,
# the provided finalTestingData will be used only once when the model is trained and tested.

# Features (predictors) selection, 
# 1. let's start to remove predictors with NA values
filtTrainingSet <- trainingSet[,colSums(is.na(trainingSet)) == 0]
filtTestingSet <- testingSet[,colSums(is.na(testingSet)) == 0]

# 2. remove all the not meaningful, used to construct the dataset or index features
filtTrainingSet$X <- NULL
filtTrainingSet$raw_timestamp_part_1 <- NULL
filtTrainingSet$raw_timestamp_part_2 <- NULL
filtTrainingSet$cvtd_timestamp <- NULL
filtTrainingSet$new_window <- NULL
filtTrainingSet$num_window <- NULL
filtTrainingSet$user_name <- NULL

filtTestingSet$X <- NULL
filtTestingSet$raw_timestamp_part_1 <- NULL
filtTestingSet$raw_timestamp_part_2 <- NULL
filtTestingSet$cvtd_timestamp <- NULL
filtTestingSet$new_window <- NULL
filtTestingSet$num_window <- NULL
filtTestingSet$user_name <- NULL

# Multicollinearity
# 3. compute features correlation and remove high correlated predictors (do not add any additional information)
# 3.1 Compute the correlation matrix and remove features with correlation higher than 0.90
corrMat <- cor(filtTrainingSet[, !(colnames(filtTrainingSet) == "classe")])
#corrplot(corrMat, order = "hclust")
highCorr <- findCorrelation(corrMat, 0.90)
filtTrainingSet <- filtTrainingSet[,-highCorr]
filtTestingSet <- filtTestingSet[,-highCorr]
#corrMat <- cor(filtTrainingSet[, !(colnames(filtTrainingSet) == "classe")])
#corrplot(corrMat, order = "hclust")

# 4 Let's apply some pre-processing (all features except for the outcome)
preProc <- preProcess(filtTrainingSet[, !(colnames(filtTrainingSet) == "classe")], method = c("center", "scale"))
preProcTraining <- predict(preProc, filtTrainingSet[, !(colnames(filtTrainingSet) == "classe")])
preProcTesting <- predict(preProc, filtTestingSet[, !(colnames(filtTestingSet) == "classe")])

# 5 Append the outcome
preProcTraining$classe <- filtTrainingSet$classe
preProcTesting$classe <- filtTestingSet$classe

# 6 Fit a model (train)
# prepare training scheme
#control <- trainControl(method="repeatedcv", number=100, repeats=3)
# train the RPART model (accuracy sucks)
#set.seed(7)
#modelRpart <- train(preProcTraining$classe~., data=preProcTraining, method="rpart", trControl=control)
# summarize the distribution
#modelRpart

# train the RF model (accuracy quite good, 98%)
#control <- trainControl(method="cv", number=5, allowParallel=TRUE)
#set.seed(7)
#modelRf <- train(preProcTraining$classe~., data=preProcTraining, method="rf", trControl=control, prox=TRUE, do.trace=TRUE)

# train the RF model (accuracy quite good, 99%)
train_control <- trainControl(method="repeatedcv", number=10, repeats=10, allowParallel=TRUE)
set.seed(7)
modFit <- train(preProcTraining$classe~., data=preProcTraining, method="rf", trControl=train_control, ntree=500, prox=TRUE, do.trace=TRUE)

# train the RF model (accuracy quite good, 98.9% 5,2,100?)
#train_control_bis <- trainControl(method="repeatedcv", number=10, repeats=3, allowParallel=TRUE)
#set.seed(7)
#modFit_bis <- train(preProcTraining$classe~., data=preProcTraining, method="rf", trControl=train_control_bis, ntree=100, prox=TRUE, do.trace=TRUE)

#Check the accuracy of the model on the testing part of the dataset (99.41 %)
predictions <- predict(modFit$finalModel, newdata=preProcTesting[, !(colnames(preProcTesting) == "classe")])
confusionMatrix(data=predictions, preProcTesting$classe)

#Let's execute it on the final testing set
filtFinalTestingData <- finalTestingData[,colSums(is.na(finalTestingData)) == 0]
filtFinalTestingData$X <- NULL
filtFinalTestingData$raw_timestamp_part_1 <- NULL
filtFinalTestingData$raw_timestamp_part_2 <- NULL
filtFinalTestingData$cvtd_timestamp <- NULL
filtFinalTestingData$new_window <- NULL
filtFinalTestingData$num_window <- NULL
filtFinalTestingData$user_name <- NULL
filtFinalTestingData <- filtFinalTestingData[,-highCorr]
preFinalTesting <- predict(preProc, filtFinalTestingData[, !(colnames(filtFinalTestingData) == "problem_id")])
finalPredictions <- predict(modFit$finalModel, newdata=preFinalTesting[, !(colnames(preFinalTesting) == "problem_id")])

pml_write_files(finalPredictions)