"Practical Machine Learning - Course Project writeup"
-----------------------------------------------------------------
## Assignment background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3PeJdnp9u

The training data for this project is available here:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data is available here:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Data loading and cleaning
Data is read directly from the url and stored locally on R. This step can be avoided if you have the data saved locally.

```r
setwd("D:/Coursera/Practical Machine Learning")
if (!file.exists("./training.csv")) {
 download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', destfile="./training.csv")
 download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile="./test.csv") }
training <- read.csv("./training.csv", header=TRUE, na.strings=c("NA", "#DIV/0!"))
testing  <- read.csv("./test.csv", header=TRUE, na.strings=c("NA", "#DIV/0!"))
#summary(training)
```
It was observed that there were multiple NAs in a lot of the variables. We will be removing those variables from our data set and create a usable tidy dataset

```r
train_nar <- training[, apply(training, 2, function(x) !any(is.na(x)))] 
dim(train_nar)
```

```
## [1] 19622    60
```

```r
#removing variables not to be used in modeling - time, userid etc
trainData <- train_nar[, -c(1:7)]
dim(trainData)
```

```
## [1] 19622    53
```
Cleaning test dataset to remove variables excluded in training. Also we need to remove the 'classe' variable which has to be predicted.

```r
testData <- testing[, names(trainData[,-53])]
dim(testData)
```

```
## [1] 20 52
```
## Predictive model development methodology
We will be splitting the training datset 70-30 using caret, to be used in classification algorithms that will be used here - randomForest, c&rt, c5.0. Please make sure that the packages are installed already in your system else the models won't work.

```r
#Loading packages
library(caret)
library(rpart)
library(randomForest)
library(C50)
library(e1071)
```

```
## Error in library(e1071): there is no package called 'e1071'
```

```r
library(plyr)
split<-createDataPartition(y=trainData$classe, p=0.7,list=F)
train1<- trainData[split,] 
test1<- trainData[-split,] 
dim(train1)
```

```
## [1] 13737    53
```
The first model was developed for a C&RT algorithm using rpart

```r
fit1 <- randomForest(train1[,1:52], train1$classe, mtry= 20, importance=TRUE, proximity = TRUE, keep.forest=TRUE)
```

```
## Error: cannot allocate vector of size 1.4 Gb
```


