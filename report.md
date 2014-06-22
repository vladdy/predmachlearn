Practical Machine Learning - Project
====================================

## Summary

The purpose of this project is to create a predictive model to distinguish correct and incorrect Unilateral Dumbbell Biceps Curls^1 performed by a person.  Data used for prediction are motion data captured by sensors attached to various parts of the participant's body and dumbbell.  

The predictive model developed in this project is based on a Random Forests 
learning algorithm. The model is able to correctly predict 99% of the out-of-sample test cases.

The remainder of this report describes data preparation for modeling, model training and
assessing the model's performance.

## Data Preparation

Load required libraries and configure multiprocessing:


```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(cores = 4)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

Import data:


```r
data <- read.csv('data/pml-training.csv')
```

Clean up near zero variance parameters:


```r
data <- data[,-nearZeroVar(data)]
```

Remove non-numeric columns or columns without data:


```r
isNA <- apply(data, 2, function(x) { sum(is.na(x)) })
data <- subset(data[,which(isNA == 0)],select=-c(X,user_name,cvtd_timestamp,  num_window, raw_timestamp_part_1, raw_timestamp_part_2))
```

Find and remove highly correlated columns:


```r
numCols <- which(sapply(data, is.numeric))
corCols <- findCorrelation(cor(data[, numCols], use='pairwise'), cutoff=.95)
data <- data[,-numcols[corcols]]
```

```
## Error: object 'numcols' not found
```

Divide training data provided into two sets (training, testing):


```r
set.seed(12345)
inTrain <- createDataPartition(data$classe, list=FALSE, p=.7)
training = data[inTrain,]
testing = data[-inTrain,]
```

Impute the data and make the preprocess model using only numeric data:

## Building Random Forests Model

Now we can build our random forest model. By default Random Forests method uses bootstraping which is too resource consuming. We are going to use alternative settings passed via trControl:


```r
fit <- train(classe ~ ., method="rf", data = training, trControl = trainControl(method = "cv", number = 4))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

## Model verification

We will now test how our model performs on the testing set:


```r
confusionMatrix(predict(fit,testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   10    0    0    0
##          B    3 1125    8    0    0
##          C    0    4 1012   13    1
##          D    0    0    6  951    4
##          E    0    0    0    0 1077
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.989, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.988    0.986    0.987    0.995
## Specificity             0.998    0.998    0.996    0.998    1.000
## Pos Pred Value          0.994    0.990    0.983    0.990    1.000
## Neg Pred Value          0.999    0.997    0.997    0.997    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.172    0.162    0.183
## Detection Prevalence    0.286    0.193    0.175    0.163    0.183
## Balanced Accuracy       0.998    0.993    0.991    0.992    0.998
```

Looks like we got 99% accuracy. 

## Prediction for the test set

Now we can get the answers for the programming assignment:


```r
data.testing <-  read.csv('data/pml-testing.csv')
answers <- predict(fit, data.testing[,names(data)[-ncol(data)]])
```

... and save them into separate file each:


```r
pml_write_files = function(x){ # Instructor provided function to generate submission data for grading
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(as.character(answers))
```

-----------
## Reference

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises**. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.  [See Documentation](http://groupware.les.inf.puc-rio.br/har)
