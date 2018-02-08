# Importing the dataset
dataset = read.csv('CompleteDataset.csv')

# Data pre-processing
dataset = dataset[,c(3,7,8,11,12,13:47)]

#dataset$Nationality <- as.numeric(as.factor(dataset$Nationality))
dataset$Value <- gsub("\342\202\254", "", dataset$Value)
dataset$Wage <- gsub("\342\202\254", "", dataset$Wage)

#transform M and K to digits for Value and Wage column
dataset$Value = ifelse(grepl("M",dataset$Value),
                        as.numeric(gsub("M","",dataset$Value))*1000000,
                        as.numeric(gsub("K","",dataset$Value))*1000)
dataset$Wage = ifelse(grepl("M",dataset$Wage),
                       as.numeric(gsub("M","",dataset$Wage))*1000000,
                       as.numeric(gsub("K","",dataset$Wage))*1000)

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
split = sample.split(dataset$Value, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Value~Overall,
               data = training_set)
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Mean absolute percentage error (MAPE). Lower the better.
#mape = mean(abs(y_pred - test_set$Value)/test_set$Value)

# k-Fold cross validation
library(DAAG)
cvResults <- suppressWarnings(CVlm(data=dataset, form.lm=Value~Overall, m=5, dots=FALSE, seed=10, legend.pos="topleft",  printit=FALSE, main="Small symbols are predicted values while bigger ones are actuals."));  # performs the CV
attr(cvResults, 'ms')

# Plot
plot(y_pred)


