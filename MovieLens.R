################################
#          MovieLens Capstone project for online course HarvardX: PH125.9x
#          Author: Lukasz Kolodziejek
#          Date: 23.09.2019
################################

################################
# Create edx set, validation set
# This part of code is provided by course authors
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# MovieLens Capstone project
################################

################################
# ANALYSIS
################################

################################
# QUIZ
################################

# Initial dataset analysis

# Number of rows and columns in dataset
dim(edx)

# Number of zeroes given as ratings in datase
nrow(filter(edx, rating==0))

# Number of threes given as ratings in datase
nrow(filter(edx, rating==3))

# Number of different movies in dataset
edx %>% summarize(n_movies = n_distinct(movieId))

# Number of different users in dataset
edx %>% summarize(n_movies = n_distinct(userId))

# Number of movie ratings per genre
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Title with most ratings
edx %>% 
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(1)

# Top 5 most often given ratings
edx %>% 
  group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(5)

################################
# MODEL OPTIMIZATION
################################

# Let's define RMSE function as it will be used to assess model's performance

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Just the average - predicting average rating for all titles

mu <- mean(edx$rating)  

model_1_rmse <- RMSE(validation$rating, mu)

rmse_results <- tibble(method = "Model 1 - Just the average", RMSE = model_1_rmse)

# Movie effect - predicting average rating and adjusting for movie effect (average rating for specific movie)

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 25, data = ., color = I("black"))

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_2_rmse <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 2 - Movie effect",
                                     RMSE = model_2_rmse ))

# Movie + user effect - predicting average rating and adjusting for movie & user effect (average rating for specific movie and average for user)

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 25, data = ., color = I("black"))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_3_rmse <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3 - Movie + User effects",  
                                     RMSE = model_3_rmse ))

# Regularization + movie + user effect - additionally to the previous model, now we will be trying to reduce variability of the effect sizes by penalizing large estimates that come from small sample sizes

# We need to select optimal lambda parameter to optimize regularization. 
# As validation dataset cannot be used to optimize model parameters, we will divide train set into two subsets 'trainsubset' and 'testsubset'
# Trainsubset will be used to train model, while testsubset to calculate MRSE for given lambda
# Once optimal lambda will be selected, final MRSE will be calculated on validation set
# This will help us not to overtrain the model

# Initial split

testsubset_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
trainsubset <- edx[-testsubset_index,]
temp <- edx[testsubset_index,]

# Make sure userId and movieId in testsubset set are also in trainsubset

testsubset <- temp %>% 
  semi_join(trainsubset, by = "movieId") %>%
  semi_join(trainsubset, by = "userId")

# Add rows removed from testsubset back into trainsubet

removed <- anti_join(temp, testsubset)
trainsubset <- rbind(trainsubset, removed)

lambdas <- seq(0, 20, 1)


rmses <- sapply(lambdas, function(l){
  
  mu <- mean(trainsubset$rating)
  
  b_i <- trainsubset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- trainsubset %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- testsubset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(testsubset$rating ,predicted_ratings))
})

# Generate plot of lambdas vs rmse
qplot(lambdas, rmses)  

# Pick lambda which minimizes RMSE of test subset
lambda <- lambdas[which.min(rmses)]
lambda

# MRSE calculated on test subset for optimal lambda equals to

min(rmses)

# Now we need to calculate final RMSE by training model with fixed lambda on full training set and calculate results for validation set

mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_4_rmse <- RMSE(validation$rating,predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4 - Regularized Movie + User effects",  
                                     RMSE = model_4_rmse))

# Regularization + movie + user + genre effect - additionally to the previous model, we will now account for next available feature: genres.

# Because genres come not in tidy format, we need first to separate rows

edx_separated <- edx %>% separate_rows(genres, sep = "\\|")
validation_separated <- validation %>% separate_rows(genres, sep = "\\|")
trainsubset_separated <- trainsubset %>% separate_rows(genres, sep = "\\|")
testsubset_separated <- testsubset %>% separate_rows(genres, sep = "\\|")

# As in previous regularization model to select lambda we will be working on trainsubset and testsubset, here on their separated versions

# We need to select optimal lambda parameter to optimize regularization. 

lambdas <- seq(0, 20, 1)

rmsesG <- sapply(lambdas, function(l){
  
  mu <- mean(trainsubset_separated$rating)
  
  b_i <- trainsubset_separated %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- trainsubset_separated %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- trainsubset_separated %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- testsubset_separated %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  
  return(RMSE(testsubset_separated$rating,predicted_ratings))
})

# Generate plot of lambdas vs rmse
qplot(lambdas, rmsesG)  

# Pick lambda which minimizes regularization
lambdaG <- lambdas[which.min(rmsesG)]
lambdaG

# MRSE calculated on train and test subset for optimal lambda equals to

min(rmsesG)

# Now we need to calculate final RMSE by training model with fixed lambda on full training set and calculate results for validation set

mu <- mean(edx_separated$rating)

b_i <- edx_separated %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambdaG))

b_u <- edx_separated %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambdaG))

b_g <- edx_separated%>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambdaG))

predicted_ratings <- validation_separated %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

model_5_rmse <- RMSE(validation_separated$rating,predicted_ratings)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 5 - Regularized Movie + User + Genres effects",  
                                     RMSE = model_5_rmse))

rmse_results %>% knitr::kable()

################################
# RESULTS
################################

# Regularized Movie + User + Genres Effect Model offers best results 

# Let's double confirm our result. For this we will recalculate final model with all it's parameters

# Mu

final_mu <- mean(edx$rating)  

# Optimal lambda optimizing MRSE has been found to be 13

final_lambda <- 13

# As genres comes not in tidy format, we need first to separate rows for training and validation datasets

edx_final <- edx %>% separate_rows(genres, sep = "\\|")
validation_final <- validation  %>% separate_rows(genres, sep = "\\|")

# Now, when all parameters are known let's define model

b_i <- edx_final %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - final_mu)/(n()+final_lambda))

b_u <- edx_final %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - final_mu)/(n()+final_lambda))

b_g <- edx_final %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - final_mu)/(n()+final_lambda))

# Let's use model to predict ratings

predicted_ratings <- validation_final %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = final_mu + b_i + b_u + b_g) %>%
  .$pred

# And finally we calculate RMSE of our predictions

RMSE <- RMSE(validation_final$rating,predicted_ratings)

RMSE

################################
# CONCLUSION
################################

#It was confirmed that among 5 checked models, the last one 'Regularized Movie + User + Genres Effect Model' offers best performance 

rmse_results %>% knitr::kable()

# Optimal achieved RMSE checked on validation dataset is as follow

RMSE