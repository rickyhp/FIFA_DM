# Importing the dataset
dataset = read.csv('FullData.csv')

# Data pre-processing
dataset = dataset[,c(2,10,11,12,15,18:53)]
#dataset$Nationality <- as.numeric(as.factor(dataset$Nationality))



dataset$Height <- as.numeric(gsub(" cm", "", dataset$Height))
dataset$Weight <- as.numeric(gsub(" kg", "", dataset$Weight))

# Categorise rating
##dataset$Rating = ifelse(dataset$Rating >= 0 & dataset$Rating <= 25,
##                     1,
##                     dataset$Rating)
##dataset$Rating = ifelse(dataset$Rating >= 26 & dataset$Rating <= 50,
##                        2,
##                        dataset$Rating)
##dataset$Rating = ifelse(dataset$Rating >= 51 & dataset$Rating <= 75,
##                        3,
##                        dataset$Rating)
##dataset$Rating = ifelse(dataset$Rating >= 76 & dataset$Rating <= 100,
##                        4,
##                        dataset$Rating)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Rating, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Rating ~ .,
               data = training_set)
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Mean absolute percentage error (MAPE). Lower the better.
mape = mean(abs(y_pred - test_set$Rating)/test_set$Rating)

# k-Fold cross validation
library(DAAG)
cvResults <- suppressWarnings(CVlm(data=dataset, form.lm=Rating ~ ., m=5, dots=FALSE, seed=10, legend.pos="topleft",  printit=FALSE, main="Small symbols are predicted values while bigger ones are actuals."));  # performs the CV
attr(cvResults, 'ms')

# Plot
plot(y_pred)


