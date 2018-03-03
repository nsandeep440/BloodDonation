rm(list = ls(all = TRUE))
cat("\014")

path = "/Users/sandeepn/Desktop/DS_Competitions/Blood Donation"
setwd(path)
list.files()
targetLabel = "Made.Donation.in.March.2007"

library(dplyr)
library(DMwR)
library(mice)
library(ROCR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(e1071)

trainBD = read.csv("Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv")
testBD = read.csv("Warm_Up_Predict_Blood_Donations_-_Test_Data.csv")

str(trainBD)
summary(trainBD)
sum(is.na(trainBD))
colnames(trainBD)
trainBD$Made.Donation.in.March.2007 = as.factor(trainBD$Made.Donation.in.March.2007)

### Builing Model 
logisticModel = glm(trainBD$Made.Donation.in.March.2007 ~ ., data = trainBD, family = 'binomial')
summary(logisticModel)

##### Stepwise regression.
logisticStepAIC = stepAIC(logisticModel, direction = "both")
summary(logisticStepAIC)  ## AIC: 564.61

vif(logisticStepAIC)

######## Apply generalized model on test data.
predictOnTest = predict(logisticStepAIC, testBD)
predictResponse = predict(logisticStepAIC, type = "response")

predictionROC = prediction(predictResponse, trainBD$Made.Donation.in.March.2007)
performanceROC = performance(predictionROC, measure = "tpr", x.measure = 'fpr')

par(mfrow=c(1,1))
plot(performanceROC, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
performanceAUC = performance(predictionROC, measure="auc")
aucValue = performanceAUC@y.values #  perf_auc1@y.values[[1]]
print(aucValue) ## 74.9 %

# For different threshold values identifying the tpr and fpr
cutoffs = data.frame(cut = performanceROC@alpha.values[[1]], 
                     fpr = performanceROC@x.values[[1]], 
                     tpr = performanceROC@y.values[[1]])
# Sorting the data frame in the decreasing order based on tpr
cutoffs = cutoffs[order(cutoffs$tpr, decreasing=TRUE),]
cutoffs

# Plotting the true positive rate and false negative rate based based on the cutoff       
# increasing from 0.1-1
threshold.value = 0.3
plot(performanceROC, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

# Predictions on train -->

predictionClassTrain = ifelse(predictResponse > threshold.value, "yes", "no")
table(trainBD$Made.Donation.in.March.2007, predictResponse)
confusionMatrixTrain = table(trainBD$Made.Donation.in.March.2007, predictionClassTrain)
confusionMatrixTrain

probabilityOnTest = predict(logisticStepAIC, testBD, type = 'response')
probabilityOnTest
predictOnTest = ifelse(probabilityOnTest > threshold.value, "yes", "no")

sub_file <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier,       Item_Outlet_Sales = main_predict)

fileName = data.frame(testBD$X, "Made Donation in March 2007" = probabilityOnTest)
write.csv(fileName, 'submission.csv', row.names = FALSE)



##################################
# Random Forest
##################################

rf_train = trainBD
rf_test = testBD

# standardise data
#rf_train$X = NULL
rf_smote_train = SMOTE(rf_train$Made.Donation.in.March.2007 ~ ., rf_train, perc.over = 420, perc.under = 120)
table(rf_smote_train$Made.Donation.in.March.2007)
standardized = preProcess(rf_smote_train, method = c("center", "scale"))
rf_stand_train = predict(standardized, rf_smote_train)
rf_stand_test = predict(standardized, rf_test)

library(randomForest)
attach(rf_stand_train)
rf = randomForest(Made.Donation.in.March.2007 ~ ., rf_stand_train, ntree=50, mtry=5, importance = TRUE)
importance(rf)
varImpPlot(rf)

submission = data.frame(x = testBD$X)
rf_probabilityTest = predict(rf, rf_stand_test, type = 'prob')
submission["Made Donation in March 2007"] = rf_probabilityTest #predict(rf, rf_stand_test)
write.csv(submission, 'submissionRF.csv', row.names = FALSE)

### using variable imp to GLM.

newTrain = trainBD
str(newTrain)
#newTrain$Number.of.Donations = as.factor(newTrain$Number.of.Donations)
#testBD$Number.of.Donations = as.factor(testBD$Number.of.Donations)
newTrain$Number.of.Donations = ifelse(newTrain$Number.of.Donations <= 10, 'Very Low', 
                                      ifelse(newTrain$Number.of.Donations > 10 & newTrain$Number.of.Donations <= 20, 'Low', 
                                            ifelse(newTrain$Number.of.Donations > 20 & newTrain$Number.of.Donations <= 30, 'Medium', 
                                                  ifelse(newTrain$Number.of.Donations > 30 & newTrain$Number.of.Donations <= 40, 'High','V High'))))
                                      
dummies <- dummyVars( ~ ., data = newTrain[-6], levelsOnly = FALSE)
et <- as.data.frame(predict(dummies, newdata = newTrain))
et$Made.Donation.in.March.2007 = newTrain$Made.Donation.in.March.2007
train1 = et

newTest = testBD
newTest$Number.of.Donations = ifelse(newTest$Number.of.Donations <= 10, 'Very Low', 
                                     ifelse(newTest$Number.of.Donations > 10 & newTest$Number.of.Donations <= 20, 'Low', 
                                            ifelse(newTest$Number.of.Donations > 20 & newTest$Number.of.Donations <= 30, 'Medium', 
                                                   ifelse(newTest$Number.of.Donations > 30 & newTest$Number.of.Donations <= 40, 'High','V High'))))


dummiesTest <- dummyVars( ~ ., data = newTest, levelsOnly = FALSE)
et_test <- as.data.frame(predict(dummiesTest, newdata = newTest))
test1 = et_test

logisticModel_rf = glm(train1$Made.Donation.in.March.2007 ~ ., 
                    data = train1[-1], family = 'binomial')
summary(logisticModel_rf)
logisticStepAIC_rf = stepAIC(logisticModel_rf, direction = "both")
summary(logisticStepAIC_rf)  ## AIC: 564.61

probabilityOnTest_rf = predict(logisticStepAIC_rf, test1, type = 'response')
file1 = data.frame(testBD$X)
file1["Made Donation in March 2007"] = probabilityOnTest_rf
write.csv(file1, 'Logistic555.csv', row.names = FALSE)

########
# above code gives - 555 AIC
########

tr1_d = trainBD
tr1 = SMOTE(tr1_d$Made.Donation.in.March.2007 ~ ., tr1_d, perc.over = 420, perc.under = 120)
table(tr1$Made.Donation.in.March.2007)
te1 = testBD

#### 1. converting months to years.
# Months.since.First.Donation
summary(tr1)

#tr1$Months.since.First.Donation = ifelse(tr1$Months.since.First.Donation < 12, "0 Years", 
#                                         ifelse(tr1$Months.since.First.Donation < 24, "1 year", 
#                                                ifelse(tr1$Months.since.First.Donation < 36, "2 years", 
#                                                       ifelse(tr1$Months.since.First.Donation < 48, "3 years", 
#                                                              ifelse(tr1$Months.since.First.Donation < 60, "4 years", 
#                                                                     ifelse(tr1$Months.since.First.Donation < 72, "5 years", "6+ years"))))))

firstDonation = function(dataSet) {
  dataSet$Months.since.First.Donation = ifelse(dataSet$Months.since.First.Donation < 12, "0 Years", 
                                           ifelse(dataSet$Months.since.First.Donation < 24, "1 year", 
                                                  ifelse(dataSet$Months.since.First.Donation < 36, "2 years", 
                                                         ifelse(dataSet$Months.since.First.Donation < 48, "3 years", 
                                                                ifelse(dataSet$Months.since.First.Donation < 60, "4 years", 
                                                                       ifelse(dataSet$Months.since.First.Donation < 72, "5 years", "6+ years"))))))
  dataSet$Months.since.First.Donation = as.factor(dataSet$Months.since.First.Donation)
  return(dataSet)
}

lastDonation = function(dataSet) {
  dataSet$Months.since.Last.Donation = ifelse(dataSet$Months.since.Last.Donation < 12, "0 Years", 
                                               ifelse(dataSet$Months.since.Last.Donation < 24, "1 year", 
                                                      ifelse(dataSet$Months.since.Last.Donation < 36, "2 years", 
                                                             ifelse(dataSet$Months.since.Last.Donation < 48, "3 years", 
                                                                    ifelse(dataSet$Months.since.Last.Donation < 60, "4 years", 
                                                                           ifelse(dataSet$Months.since.Last.Donation < 72, "5 years", "6+ years"))))))
  dataSet$Months.since.Last.Donation = as.factor(dataSet$Months.since.Last.Donation)
  return(dataSet)
}

numberOfDonationsToCategorical = function(dataSet) {
  dataSet$Number.of.Donations = ifelse(dataSet$Number.of.Donations <= 10, 'Very Low', 
                                        ifelse(dataSet$Number.of.Donations > 10 & dataSet$Number.of.Donations <= 20, 'Low', 
                                               ifelse(dataSet$Number.of.Donations > 20 & dataSet$Number.of.Donations <= 30, 'Medium', 
                                                      ifelse(dataSet$Number.of.Donations > 30 & dataSet$Number.of.Donations <= 40, 'High','V High'))))
  dataSet$Number.of.Donations = as.factor(dataSet$Number.of.Donations)
  return(dataSet)
}

convertToDummies = function(dataSet) {
  dummiesTest <- dummyVars( ~ ., data = dataSet, levelsOnly = FALSE)
  dum_data <- as.data.frame(predict(dummiesTest, newdata = dataSet))
  return(dum_data)
}

buildLogisticModel = function(trainData, testData) {
  logisticModel_rf = glm(trainData$Made.Donation.in.March.2007 ~ ., 
                         data = trainData[-1], family = 'binomial')
  summary(logisticModel_rf)
  logisticStepAIC_rf = stepAIC(logisticModel_rf, direction = "both")
  summary(logisticStepAIC_rf)  
  return(logisticStepAIC_rf)
}

saveToCSV = function(probabilityForTest, fileName) {
  file1 = data.frame(testBD$X)
  file1["Made Donation in March 2007"] = probabilityForTest
  write.csv(file1, fileName, row.names = FALSE)
}
tr1 = firstDonation(tr1)
#tr1 = lastDonation(tr1)
tr1 = numberOfDonationsToCategorical(tr1)
#tr1$Months.since.Last.Donation = NULL
#tr1$X = NULL
summary(tr1)
te1 = firstDonation(te1)
#te1 = lastDonation(te1)
te1 = numberOfDonationsToCategorical(te1)
#te1$Months.since.Last.Donation = NULL
#te1$X = NULL
summary(te1)

#xxx = tr1[,-which(names(tr1) == targetLabel)]

tr1.dum = convertToDummies(tr1[,-which(names(tr1) == targetLabel)])
tr1.dum$Made.Donation.in.March.2007 = tr1$Made.Donation.in.March.2007
te1.dum = convertToDummies(te1)

model = buildLogisticModel(tr1.dum, te1.dum)
summary(model)
probabilityForTest = predict(logisticStepAIC_rf, te1.dum, type = 'response')





