library(caret)
library(corrplot)
source("pml_write_files.R")

# Set the seed for results reproducibility
set.seed(7)

# Load full dataset
dataPath = "../data";
fullDataName = "pml-training.csv";
testingDataName = "pml-testing.csv";

fullDataPath = paste(dataPath, fullDataName, sep = "/");
testingDataPath = paste(dataPath, testingDataName, sep = "/");

# Reading the testing data removing the "#DIV/0!" strings 
fullData = read.csv(fullDataPath, na.strings=c("NA", "#DIV/0!"));
finalTestingData = read.csv(testingDataPath);

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
filtTrainingSet <- subset(filtTrainingSet, select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window, user_name))
filtTestingSet <- subset(filtTestingSet, select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window, user_name))

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

# 6 train the RF model (accuracy quite good, 99%)
train_control <- trainControl(method="repeatedcv", number=10, repeats=10, allowParallel=TRUE)
modFit <- train(preProcTraining$classe~., data=preProcTraining, method="rf", trControl=train_control, ntree=500, prox=TRUE, do.trace=TRUE)

#Check the accuracy of the model on the testing part of the dataset (99.41 %)
predictions <- predict(modFit$finalModel, newdata=preProcTesting[, !(colnames(preProcTesting) == "classe")])
confusionMatrix(data=predictions, preProcTesting$classe)

#Let's execute it on the final testing set
filtFinalTestingData <- finalTestingData[,colSums(is.na(finalTestingData)) == 0]
filtFinalTestingData <- subset(filtFinalTestingData, select=-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window, user_name))

filtFinalTestingData <- filtFinalTestingData[,-highCorr]
preFinalTesting <- predict(preProc, filtFinalTestingData[, !(colnames(filtFinalTestingData) == "problem_id")])
finalPredictions <- predict(modFit$finalModel, newdata=preFinalTesting[, !(colnames(preFinalTesting) == "problem_id")])

pml_write_files(finalPredictions)