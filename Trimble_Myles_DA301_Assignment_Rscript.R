###############################################################################
###############################################################################


#######################      TURTLE GAMES R SCRIPT      #######################


###############################################################################
###############################################################################



## Import libraries & data set.

# Import tidyverse.
library(tidyverse)

# import csv file.
# ts <- read.csv('turtle_sales.csv', header = T)
ts <- read.csv(file.choose(), header = T)

# View data.
head(ts)

# View data in new window.
View(ts)


#######################


## Remove redundant columns.

# Remove the 'Ranking', 'Year', 'Genre', and 'Publisher' columns.
ts2 <- select(ts, -Ranking, -Year, -Genre, -Publisher)


## Explore the new dataframe.

# View the dataframe.
head(ts2)

# tibble().
as_tibble(ts2)

# glimpse().
glimpse(ts2)

# summary().
summary(ts2)


#######################


## Visualise the data.


# Scatterplot for 'Global_Sales' with (x = seq_along(y)).
qplot(y = Global_Sales,
      data = ts2)

# Histogram for 'Global_Sales'.
qplot(Global_Sales,
      data = ts2)

### There is a significant outlier in 'Global_Sales'.
### --> investigate further.

# Boxplot for 'Platform' & 'Global_Sales'.
qplot(Platform, Global_Sales,
      fill = Platform,
      data = ts2,
      geom = 'boxplot')

### Wii platform contains significant outlier.


# Boxplot for 'NA_Sales'.
qplot(NA_Sales,
      data = ts2,
      geom = 'boxplot')

### There are several outliers in 'NA_Sales'.


# Boxplot for 'EU_Sales'.
qplot(EU_Sales,
      data = ts2,
      geom = 'boxplot')

### There is one significant outlier in 'EU_Sales'.



# Remove Extreme Outliers.
ts3 <- filter(ts, Global_Sales < 40)




###############################################################################




###################    Compute descriptive statistics     #####################



# Calculate the mean of each sales column.
colMeans(ts3[ , 7:9])

# Calculate the min of each sales column.
ts_min <- sapply(ts3[ , 7:9], min)
ts_min

# Calculate the max of each sales colum.
ts_max <- sapply(ts3[ , 7:9], max)
ts_max

# Calculate Q1 of each sales column.
ts_25q <- sapply(ts3[ , 7:9], quantile, 0.25)
ts_25q

# Calculate Q4 of each sales column.
ts_75q <- sapply(ts3[ , 7:9], quantile, 0.75)
ts_75q

# Calculate IQR of each sales column.
ts_IQR <- sapply(ts3[, 7:9], IQR)
ts_IQR

# Calculate the varience of each sales column.
ts_var <- sapply(ts3[ , 7:9], var)
ts_var

# Calculate the standard deviation of each sales column.
ts_sd <- sapply(ts3[ , 7:9], sd)
ts_sd


# Create a summary of the dataframe.
summary(ts3)


########################################


# Aggregate.
# aggregate(Global_Sales~Product+NA_Sales+EU_Sales, ts2, sum)


## Group the dataframe by product.

# Group by product sales.
ts_id <- ts3 %>% group_by(Product) %>%
  summarise(sum_NA = sum(NA_Sales),
            sum_EU = sum(EU_Sales),
            sum_Global = sum(Global_Sales),
            .groups = 'drop')

# View group by.
ts_id

# Create a summary of the data.
summary(ts_id)


###############################################################################


#######################       Visualise the data        #######################


## Percentage of Products Belonging to Each Platform.


# Group by platform sales.
ts_plat <- ts3 %>% group_by(Platform, Product) %>%
  summarise(sum_NA = sum(NA_Sales),
            sum_EU = sum(EU_Sales),
            sum_Global = sum(Global_Sales),
            .groups = 'drop')

# View Groupby.
ts_plat


# Save plot to directory.
tiff("Platform_Percentage.tiff", units = "in", width = 9.3, height = 5.2, 
     res = 300)

# Specify the ggplot function. 
ggplot(ts_plat,
       # Specify 'y' to create a percentage. 
       aes(x = Platform, y = ..count../sum(..count..),
           fill = ..count../sum(..count..))) +  
  # Specify attributes.
  geom_bar(stat = 'count', show.legend = F) +
  # Specify titles.
  labs(x = "Platform",
       y = "Percentage",
       title = "Percentage of Products Belonging to Each Platform") +  
  # Pass labels to the scale.
  scale_y_continuous(label = scales::percent) +
  scale_fill_gradient(low = "lightpink1",high = "tomato4") +
  theme_classic()

# Save plot to directory.
dev.off()

### X360, PS3, and PC are the platforms with the most products attached.
### 2600 and GEN have very few products.

#######################


# Save plot to directory.
tiff("Platform_Distribution.tiff", units = "in", width = 9.3, height = 5.2, 
     res = 300)

# Boxplot for Global Sales per Platform
ggplot(ts_plat, aes(x = Platform, y = sum_Global, fill = Platform)) +
  geom_boxplot( outlier.color = 'red', show.legend = F) +
  labs(title = "Distribution of Global Sales per Platform",
       x = "Platform",
       y = "Global Sales (in £ millions)") +
  theme_classic()

# Save plot to directory.
dev.off()


#######################


# Boxplot for NA Sales per Platform
ggplot(ts_plat, aes(x = Platform, y = sum_NA, fill = Platform)) +
  geom_boxplot( outlier.color = 'red', show.legend = F) +
  labs(title = "Distribution of North American Sales per Platform",
       x = "Platform",
       y = "North American Sales (in £ millions)") +
  theme_classic()


# Boxplot for EU Sales per Platform
ggplot(ts_plat, aes(x = Platform, y = sum_EU, fill = Platform)) +
  geom_boxplot( outlier.color = 'red', show.legend = F) +
  labs(title = "Distribution of European Sales per Platform",
       x = "Platform",
       y = "European Sales (in £ millions)") +
  theme_classic()


#######################


## Distribution of regional sales.

# Specify the ggplot function:
ggplot(ts3, aes(x = Global_Sales)) +
  geom_histogram(fill = 'forestgreen') + 
  labs(x = "Global Sales (in £millions)",
       y = " ",
       title = "Distribution of Global Sales") +
  theme_classic()


# Specify the ggplot function:
ggplot(ts3, aes(x = NA_Sales)) +
  geom_histogram(fill = 'firebrick') + 
  labs(x = "North American Sales (in £millions)",
       y = " ",
       title = "Distribution of North American Sales") +
  theme_classic()


# Specify the ggplot function:
ggplot(ts3, aes(x = EU_Sales)) +
  geom_histogram(fill = 'blue4') + 
  labs(x = "European Sales (in £millions)",
       y = " ",
       title = "Distribution of European Sales") +
  theme_classic()


#######################


# Import Reshape.
library("reshape")

## Create a melted DataFrame.

# Create a subsetted DataFrame.
ts_m = data.frame(ts3$NA_Sales, ts3$EU_Sales, ts3$Global_Sales)

# Melt the DataFrame.
ts_m <- melt(ts_m)

# Rename values.
ts_m <- ts_m %>%
  mutate(variable = recode(variable,
                           ts3.NA_Sales = 'North America',
                           ts3.EU_Sales = 'Europe',
                           ts3.Global_Sales =  'Global' ))

# View DataFrame.
head(ts_m)


# Save plot to directory.
tiff("Regional_Sales_Distribution.tiff", units = "in", width = 9.3,
     height = 5.2, res = 300)

ggplot(ts_m, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot( outlier.color = 'red', show.legend = F) +
  labs(title = "Distribution of Regional Sales",
       x = "Region",
       y = "Sales (in £ millions)") +
  theme_classic()

# Save plot to directory.
dev.off()


#######################


# Density of regional sales.


# Specify the ggplot function:
ggplot(ts3, aes(x = Global_Sales)) +
  geom_density(fill = 'forestgreen', bw = 1) + 
  labs(x = "Global Sales (in £millions)",
       y = " ",
       title = "Density of Global Sales",
       subtitle = "Bandwidth = 1") +
  theme_classic()


# Specify the ggplot function:
ggplot(ts3, aes(x = NA_Sales)) +
  geom_density(fill = 'firebrick', bw = 1) + 
  labs(x = "North American Sales (in £millions)",
       y = " ",
       title = "Density of North American Sales",
       subtitle = "Bandwidth = 1") +
  theme_classic()


# Specify the ggplot function:
ggplot(ts3, aes(x = EU_Sales)) +
  geom_density(fill = 'blue4', bw = 1) + 
  labs(x = "European Sales (in £millions)",
       y = " ",
       title = "Density of European Sales",
       subtitle = "Bandwidth = 1") +
  theme_classic()


#######################


# Group by Genre sales.
ts_g <- ts3 %>% group_by(Genre, Year) %>%
  summarise(sum_NA = sum(NA_Sales),
            sum_EU = sum(EU_Sales),
            sum_Global = sum(Global_Sales),
            .groups = 'drop')

# View Groupby.
ts_g

# Save plot to directory.
tiff("Genre_Time.tiff", units = "in", width = 9.3, height = 5.2, 
     res = 300)

# Line plot showing the popularity of each genre through time.
ggplot(ts_g, 
       mapping = aes(x = Year, y = sum_Global, colour = Genre)) +
  # geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(se = FALSE) +
  scale_x_continuous(breaks = seq(1980, 2020, 5), "Year") +
  scale_y_continuous(breaks = seq(0, 90, 10), "Global Sales (in £ millions)") +
  labs(title = "Global Sales for Each Genre Through Time") +
  theme_classic()

# Save plot to directory.
dev.off()


#######################


# Group by platform sales.
ts_plat_y <- ts3 %>% group_by(Platform, Year) %>%
  summarise(sum_NA = sum(NA_Sales),
            sum_EU = sum(EU_Sales),
            sum_Global = sum(Global_Sales),
            .groups = 'drop')

# View Groupby.
ts_plat_y


# Save plot to directory.
tiff("Platform_Time.tiff", units = "in", width = 9.3, height = 5.5, 
     res = 300)

ggplot(ts_plat_y, 
       mapping = aes(x = Year, y = sum_Global, colour = Platform)) +
  # geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(se = FALSE) +
  scale_x_continuous(breaks = seq(1980, 2020, 5), "Year") +
  scale_y_continuous(breaks = seq(0, 90, 10), "Global Sales") +
  labs(title = "Global Sales for Each Genre Through Time") +
  theme_classic()

# Save plot to directory.
dev.off()


###############################################################################



#####################    Determine Normality of Data    #######################


# Import the necessary packages.
library (moments)
library(BSDA)


# Visualise.
hist(ts3 $ NA_Sales)
hist(ts3 $ EU_Sales)
boxplot(ts3 $ NA_Sales)
boxplot(ts3 $ EU_Sales)

# Q-Q plot to determine normality.
qqnorm(ts3 $ NA_Sales)
qqline(ts3 $ NA_Sales, col = 'red')
qqnorm(ts3 $ EU_Sales)
qqline(ts3 $ EU_Sales, col = 'red')

# Shapiro-Wilk test to determine normality.
shapiro.test(ts3 $ NA_Sales)
shapiro.test(ts3 $ EU_Sales)

# Compute the skewness and kurtosis.
skewness(ts3 $ NA_Sales)
kurtosis(ts3 $ NA_Sales)
skewness(ts3 $ EU_Sales)
kurtosis(ts3 $ EU_Sales)

# The Q-Q plot indicates that the data is very much not normally distributed.
# The Shapiro-Wilk test returned a p < 2.2e-16, which is smaller than 0.05. 
# Therefore, we can conclude that the North American and European sales data
# are not normally distributed.
# I will transform the data so as to hopefully glean a more accurate insight.


#####################


# Transform NA_Sales by adding a column with the square root.
ts3 $ NA_sqrt = sqrt(ts3 $ NA_Sales)
ts3 $ EU_sqrt = sqrt(ts3 $ EU_Sales)

# View the new column.
head(ts3)


## Check distribution with transformed data.


# Visualise.
hist(ts3 $ NA_sqrt)
hist(ts3 $ EU_sqrt)
boxplot(ts3 $ NA_sqrt)
boxplot(ts3 $ EU_sqrt)


# Q-Q plot to determine normality.
qqnorm(ts3 $ NA_sqrt)
qqline(ts3 $ NA_sqrt, col = 'red')
qqnorm(ts3 $ EU_sqrt)
qqline(ts3 $ EU_sqrt, col = 'red')

# Shapiro-Wilk test to determine normality.
shapiro.test(ts3 $ NA_sqrt)
shapiro.test(ts3 $ EU_sqrt)

# Compute the skewness and kurtosis.
skewness(ts3 $ NA_sqrt)
kurtosis(ts3 $ NA_sqrt)
skewness(ts3 $ EU_sqrt)
kurtosis(ts3 $ EU_sqrt)

# The histogram, boxplot, and Q-Q plot indicate that the data is still not
# normally distributed.
# The Shapiro-Wilk test returned NA: p = 1.555e-08 and EEU: p = 2.389e-06
# which is smaller than 0.05.

# The skewness of 0.751 does not fall between the range of -0.5 and 0.5,
# indicating asymmetrical distribution.
# The kurtosis of 4.262 is greater than 3, indicating more extreme tails.
# W e can conclude that the North American and European sales data is 
# not normally distributed.

# Therefore, I will employ the use of the Mann-Whitney U Test / Wilcox Rank-Sum
# Test as the data is not normally distributed and sample sizes > 30.


#######################


# Mann-Whitney U test.
mwt <-wilcox.test(ts3 $ NA_Sales, ts3 $ EU_Sales)

# View results.
mwt

# As p = 0.0005501 we can reject the null-hypothesis and can conclude that
# the difference between the population medians is statistically significant.



###############################################################################



#######################        Linear Regression        #######################


# Visualise data to understand dataset.
ggplot(ts3, 
       mapping = aes(x = NA_Sales, y = EU_Sales)) +
  geom_point()


# Check for correlation.
cor(ts3 $ NA_Sales, ts3 $ EU_Sales)

# 0.59 indicates a relatively weak correlation between North American sales
# and European sales.


# Transform the data set to limit errors.
# Improve linearity of data set and increase R^2.
Sqrt_EU <- sqrt(ts3 $ EU_Sales)
Sqrt_NA <- sqrt(ts3 $ NA_Sales)


# Visualise the result of transformed data.
plot(Sqrt_NA, Sqrt_EU)

#######################

## Test the relationship between NA_Sales and EU_Sales.

# Create a linear regression model.
model1 <- lm(ts3 $ NA_Sales ~ ts3 $ EU_Sales)

# View the summary stats.
summary(model1)

# Create a visualisation to determine normality of data set.
qqnorm(residuals(model1))
qqline(residuals(model1), col='red')


#######################


# Create a second linear regression model.
model2 <- lm(Sqrt_NA ~ Sqrt_EU)


# View the summary stats.
summary(model2)

# Create a visualisation to determine normality of data set.
qqnorm(residuals(model2))
qqline(residuals(model2), col='blue')


#######################


## Compare the two models.

# Arrange plot with the par(mfrow) function.
par(mfrow = c(2, 1))


# Compare both graphs (model1 and model2).
plot(ts3 $ NA_Sales, ts3 $ EU_Sales)
abline(coefficients(model1), col = 'red')

plot(Sqrt_NA, Sqrt_EU)
abline(coefficients(model2), col = 'blue') 


# Model2 is slightly better than model1.
# However, neither model provides significant evidence that European sales are
# a strong predictor of North American sales.



#######################################



### Multiple Linear Regression


# Import the psych package.
library(psych)



# Subset the data to only contain numerical data.
ts_mlr <- ts3[ , c('NA_Sales', 'EU_Sales', 'Global_Sales')]


# Determine correlation between variables.
cor(ts_mlr)


# Use the corPlot() function.
corPlot(ts_mlr, cex = 2)

# There is a strong positive correlation (0.91) between NA_Sales and
# Global_Sales, and a strong positive correlation (0.82) between EU_Sales and 
# Global_Sales.
# There is a mild positive correlation (0.59) between EU_Sales and NA_Sales.


#######################


## Create a model


# Create a new object and 
# specify the lm function and the variables.
modela = lm(Global_Sales ~ NA_Sales + EU_Sales,
            data = ts_mlr)

# Print the summary statistics.
summary(modela)

# An Adjusted R-squared of 0.9581 indicates that Global sales are very strongly
# predicted by European and North American sales.

# - This model shows that an increase in North American sales of £1 million 
#   would result in an increase of £1.1 million in Global Sales.
#   - This coefficient indicates that European sales are positively correlated
#     with 'Other' sales from the rest of the world (which are not included in
#     this data set).

# - A more significant positive correlation is shown in the coefficient for
#   European sales, predicting that an increase of £1 million in Eeuropean sales
#   would result in an increase of £1.4 million in Global sales.


#######################


## Test the model


# Create a DataFrame with new observed values.
ts_mlr_test <- data.frame(
  NA_Sales = c(34.02, 3.93, 2.73, 2.26, 22.08),
  EU_Sales = c(23.80, 1.56, 0.65, 0.97, 0.52))

# View the data.
ts_mlr_test


# Create a new object and specify the predict function.
predictTest = predict(modela, newdata = ts_mlr_test,
                      interval = 'confidence')

# Print the object.
predictTest


#######################


# Log the Global Sales.
Sqrt_Global <- sqrt(ts3 $ Global_Sales)

# Create new dataset.
ts_mlr2 <- data.frame(Sqrt_Global, Sqrt_NA, Sqrt_EU)


# Determine correlation between variables.
cor(ts_mlr2)


# Use the corPlot() function.
corPlot(ts_mlr2, cex = 2)


#######################


## Modelb

# Create a new object and 
# specify the lm function and the variables.
modelb = lm(Sqrt_Global ~ Sqrt_NA + Sqrt_EU,
            data = ts_mlr2)

# Print the summary statistics.
summary(modelb)

# An Adjusted R-squared of 0.9557 indicates that Global sales are very strongly
# predicted by European and North American sales.

# modela appears to be a stronger fit, but given the relative unreliability of
# the data, modelb may be more accurate.


###############################################################################
###############################################################################
###############################################################################