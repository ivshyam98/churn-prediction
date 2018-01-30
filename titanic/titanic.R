# Classification models for Titanic challenge from Kaggle.
# Inspired by: https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook
# January 2018.



##########  Load packages 

# for the first time may need to install 'caret' and then a few more packages
# install.packages('caret', dependencies = c("Depends", "Imports", "Suggests"))

library(ggplot2)
library(caret)
set.seed(14)



##########  Load data 

train_df <- read.csv('./input/train.csv', stringsAsFactors=F)
test_df <- read.csv('./input/test.csv', stringsAsFactors=F)
combine <- list(train_df, test_df)



##########  Preview data 

head(train_df)
head(test_df)
summary(train_df)
summary(test_df)
str(train_df)
str(test_df)
colSums(is.na(train_df))
colSums(is.na(test_df))



##########  Fill missing values

# fill 'Age' by median value of subsets, acoording to 'Sex' and 'Pclass'
for (i in 1:length(combine)) {
    for (j in unique(combine[[i]]$Sex)) {
        for (k in unique(combine[[i]]$Pclass)){
            df <- combine[[i]]
            df <- df[(df$Sex == j) & (df$Pclass == k),'Age']
            age <- median(df, na.rm=T) 
            combine[[i]][(combine[[i]]$Sex == j) & 
                             (combine[[i]]$Pclass == k) &
                             (is.na(combine[[i]]$Age)), 'Age'] <- age }}}

# fill 'Embarked' by most frequent port
freq_port <- names(sort(-table(combine[[1]]$Embarked)))[1]
combine[[i]]$Embarked [combine[[i]]$Embarked == ''] <- freq_port

# fill 'Fare' by median value
combine[[2]]$Fare[is.na(combine[[2]]$Fare)] <- median(combine[[2]]$Fare, na.rm=T)



##########  Analyze by pivoting features

df <- as.data.frame.matrix(prop.table(table(combine[[1]]$Pclass, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <-as.data.frame.matrix(prop.table(table(combine[[1]]$Sex, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <-as.data.frame.matrix(prop.table(table(combine[[1]]$SibSp, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <-as.data.frame.matrix(prop.table(table(combine[[1]]$Parch, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))



##########  Data vizualization

ggplot(combine[[1]], aes(Age)) + geom_histogram(bins=20) + 
        facet_grid(.~Survived, labeller = labeller(.cols = label_both))

ggplot(combine[[1]], aes(Age)) + geom_histogram(bins=20) + 
        facet_grid(.~Pclass ~ Survived, labeller = labeller(.rows = label_both, .cols = label_both))



##########  Feature extraction

# extract 'Title' from 'Name' and remove rare values
for (i in 1:length(combine))
    combine[[i]]$Title <- gsub('(.*, )|(\\..*)', '', combine[[i]]$Name)

table(combine[[1]]$Title, combine[[1]]$Sex)
table(combine[[2]]$Title, combine[[2]]$Sex)

rare <- c('Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 
          'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona')

for (i in 1:length(combine)) {
    combine[[i]]$Title[combine[[i]]$Title %in% rare] <- 'Rare'
    combine[[i]]$Title[combine[[i]]$Title == 'Mlle'] <- 'Miss'
    combine[[i]]$Title[combine[[i]]$Title == 'Ms'] <- 'Miss'
    combine[[i]]$Title[combine[[i]]$Title == 'Mme'] <- 'Mrs' }

# extract 'FamilySize' and 'IsAlone'
for (i in 1:length(combine)) {
    combine[[i]]$FamilySize <- combine[[i]]$SibSp + combine[[i]]$Parch + 1 
    combine[[i]]$IsAlone[combine[[i]]$FamilySize == 1] <- 1
    combine[[i]]$IsAlone[is.na(combine[[i]]$IsAlone)] <- 0 }

# split 'Age' into bands by age intervals
breaks <- seq(0, 80, 16)
breaks[1] <- breaks[1] - 1

for (i in 1:length(combine)) {
    combine[[i]]$Age <- cut(combine[[i]]$Age, breaks=breaks) }

# split 'Fare' into bands by count of people
breaks <- quantile(combine[[2]]$Fare, probs=seq(0, 1, 0.25), na.rm=T)
breaks[1] <- breaks[1] - 1
for (i in 1:length(combine)) {
    combine[[i]]$Fare <- cut(combine[[i]]$Fare, breaks=breaks) }

# extract 'Age*Class'
for (i in 1:length(combine)) {
    combine[[i]] ['Age*Class'] <- (as.numeric(combine[[i]]$Age) - 1) * combine[[i]]$Pclass }



##########  Some more data overview

df <- as.data.frame.matrix(prop.table(table(combine[[1]]$Title, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <- as.data.frame.matrix(prop.table(table(combine[[1]]$Age, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <- as.data.frame.matrix(prop.table(table(combine[[1]]$Embarked, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))

df <- as.data.frame.matrix(prop.table(table(combine[[1]]$FamilySize, combine[[1]]$Survived), 1))
setNames(df[order(df[2], decreasing=T),2:1], c('Survived %', 'Died %'))



##########  Final data corrections

# change categorical features values to numbers
for (i in 1:length(combine)) {
    combine[[i]]$Sex <- (as.numeric(factor(combine[[i]]$Sex)) - 2) ** 2
    combine[[i]]$Age <- as.numeric(factor(combine[[i]]$Sex)) - 1 
    combine[[i]]$Fare <- as.numeric(factor(combine[[i]]$Fare)) - 1
    combine[[i]]$Embarked <- as.numeric(factor(combine[[i]]$Embarked)) - 1
    combine[[i]]$Title <- as.numeric(factor(combine[[i]]$Title)) }

# drop unnecessary features
for (i in 1:length(combine)) 
    combine[[i]][c('Ticket', 'Cabin', 'Name', 'PassengerId', 
                   'Parch', 'SibSp', 'FamilySize')] <- NULL

# get train and test sets
X_train <- combine[[1]] [,-1]
Y_train <- combine[[1]] ['Survived']
X_test <- combine[[2]]



##########  Data modeling

# (may need to install additional packages to make all these to work)
algorithms <- c('glm',        # Logistic Regression
                'svmRadial',  # Support Vector Machines with Radial Basis Kernel
                'knn',        # k-Nearest Neighbors 
                'nb',         # Gaussian Naive Bayes
                'mlp',        # Perceptron
                'svmLinear',  # Support Vector Machines with Linear Kernel
                'mlpSGD',     # Stochastic Gradient Descent (mlp)
                'rpart',      # Decision Tree
                'rf')         # Random Forest
                
# fit models and get results
for (alg in algorithms){
    
    print(paste('working on', alg, 'model...'))
    
    # fit model
    model <- train(X_train, factor(Y_train$Survived), method=alg)
    
    # predict output
    preds <- predict(object=model, X_test, type='raw')
    
    # save predictions to files
    predictions <- data.frame(PassengerID=test_df$PassengerId, Survived=preds)
    write.csv(predictions, file=paste('./output/predictions_', alg, '.csv', sep=''), row.names=F) }


