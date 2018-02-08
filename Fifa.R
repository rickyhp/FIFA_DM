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


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Value, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Linear Regression to the training set
regressor = lm(formula = Value~Overall,
               data = training_set)
# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Fitting Polynomial Linear Regression to the training set
training_set$Overall2 = training_set$Overall^2
training_set$Overall3 = training_set$Overall^3
training_set$Overall4 = training_set$Overall^4

poly_regressor = lm(formula = Value~Overall+Overall2+Overall3+Overall4,
               data = training_set)

# Visualising the Linear Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Overall, y = dataset$Value),
             colour = 'red') +
  geom_line(aes(x = dataset$Overall, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Value vs Overall (Linear Regression)') +
  xlab('Overall') +
  ylab('Value')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Overall), max(dataset$Overall), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Overall, y = dataset$Value),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_regressor,
                                        newdata = data.frame(Overall = x_grid,
                                                             Overall2 = x_grid^2,
                                                             Overall3 = x_grid^3,
                                                             Overall4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Value vs Overall (Polynomial Regression)') +
  xlab('Overall') +
  ylab('Value')

# Predicting a new result with Linear Regression
predict(regressor, data.frame(Overall = 94))

# Predicting a new result with Polynomial Regression
predict(poly_regressor, data.frame(Overall = 94,
                                   Overall2 = 94^2,
                                   Overall3 = 94^3,
                                   Overall4 = 94^4))

# k-Fold cross validation
library(DAAG)
cvResults <- suppressWarnings(CVlm(data=dataset, form.lm=Value~Overall+Potential, m=5, dots=FALSE, seed=10, legend.pos="topleft",  printit=FALSE, main="Small symbols are predicted values while bigger ones are actuals."));  # performs the CV
attr(cvResults, 'ms')

# Plot
plot(y_pred)


