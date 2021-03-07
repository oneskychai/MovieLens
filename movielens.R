# Load necessary libraries
library("tidyverse")
library("lubridate")
library("caret")

# Create and save training and test sets
# https://github.com/oneskychai/MovieLens/blob/trunk/create_data_sets.R

# Load edx training set object
load("rdas/edx.rda")

# Partition edx further into test and train sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = F)
train_set <- edx[-test_index]
temp <- edx[test_index]

# Make sure users and movies in test set are in train set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back to train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# Remove unnecessary objects
rm(removed, temp, test_index, edx)

# Calculate average rating of train set
mu <- mean(train_set$rating)

# Calculate RMSE of guessing mu for all ratings
naive_rmse <- RMSE(test_set$rating, mu)

# Start a tibble of RMSE's for different methods
rmse_results <- tibble(method = "Guess the average", RMSE = naive_rmse)
rm(naive_rmse)

# Plot distribution of average ratings by movie
train_set %>% 
  group_by(movieId) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "slateblue4") +
  xlab("Average rating by movie") +
  ggtitle("Distribution of average ratings by movie")

# Save plot
ggsave("figs/avg_movie_rating_dist.png")

# Calculate movie biases
movie_bias <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating) - mu)

# Generate predictions and calculate RMSE of guessing mu + b_i
predicted_ratings <- mu + test_set %>%
  left_join(movie_bias, by = "movieId") %>%
  .$b_i
rmse_movie_bias <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results, c("Movie bias", rmse_movie_bias))
rm(rmse_movie_bias)

# Calculate user biases
user_bias <- train_set %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i) - mu)

# Plot distribution of user biases
user_bias %>%
  ggplot(aes(b_u)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "seagreen4") +
  xlab("User rating bias") +
  ggtitle("Distribution of user rating biases")

# Save plot
ggsave("figs/user_bias_dist.png")

# Generate predictions and calculate RMSE of guessing mu + b_i + b_u
predicted_ratings <- test_set %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
rmse_movie_user_bias <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results, c("Movie/user bias", rmse_movie_user_bias))
rm(rmse_movie_user_bias, movie_bias, user_bias)

# Regularize movie and user biases with K-fold cross validation

# Create 5 folds for cross validation on lambda
set.seed(1, sample.kind = "Rounding")
indexes <- createFolds(train_set$rating, k = 5)

# Explore possible optimal values for lambda
# I ran the code below a couple times with different values for lambdas
# This allowed me to hone in on the optimal range to explore

# Set lambda to optimal range determined by exploration
lambdas <- seq(4.65, 5.05, 0.1)

# Create empty data frame to store results
cv_results <- data.frame(iteration = character(), lambda = double(),
                      rmse = double())

# Iterate over the 5 folds
# Note this step will take a couple minutes
for (i in 1:5) {
  
  # Divide train set further for cross validation
  cv_train <- train_set[-indexes[[i]]]
  temp <- train_set[indexes[[i]]]
  
  # Make sure movies and users in test set are in train set
  cv_test <- temp %>%
    semi_join(cv_train, by = "movieId") %>%
    semi_join(cv_train, by = "userId")
  removed <- anti_join(temp, cv_test)
  cv_train <- rbind(cv_train, removed)
  
  # Calculate RMSE's for different lambdas
  for (j in 1:5) {
    
    # Calculate average rating of cv_train set
    mu_cv <- mean(cv_train$rating)
    
    # Calculate regularized movie biases
    b_i <- cv_train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu_cv) / (n() + lambdas[j]))
    
    # Calculate regularized user biases
    b_u <- cv_train %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - mu_cv - b_i) / (n() + lambdas[j]))
    
    # Generate predictions using regularized movie and user biases
    predicted_ratings <- cv_test %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu_cv + b_i + b_u) %>%
      .$pred
    
    # Calculate RMSE
    rmse <- RMSE(predicted_ratings, cv_test$rating)
    
    # Add results to cv_results data frame
    cv_results[nrow(cv_results) + 1,] <- list(paste("CV", i), lambdas[j], rmse)
  }
}

# Plot RMSE's versus lambdas from cross validation
cv_results %>%
  group_by(iteration) %>%
  ggplot(aes(lambda, rmse, fill = iteration)) +
  geom_point(alpha = 0.5, shape = 21, size = 3, show.legend = FALSE) +
  geom_line(aes(color = iteration), show.legend = FALSE) +
  ggtitle("Cross validation results") +
  scale_x_continuous(breaks = lambdas) +
  scale_y_continuous(labels = NULL) +
  theme(axis.ticks = element_blank()) +
  facet_wrap(~iteration, scales = "free")

# Save plot
ggsave("figs/cv_results_3.png")

# Extract best lambdas from cv_results
best_lambdas <- cv_results %>%
  group_by(iteration) %>%
  summarize(ind = which.min(rmse), lambda = lambda[ind]) %>%
  .$lambda

# Set lambda to average of best lambdas
lambda <- mean(best_lambdas) # lambda <- 4.83

# Remove unnecessary objects
rm(b_i, b_u, cv_results, cv_test, cv_train, indexes, removed, temp,
   best_lambdas, i, j, lambdas, mu_cv, rmse)

# Calculate RMSE using regularized movie and user biases

# Calculate regularized movie biases
movie_bias_reg <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu) / (n() + lambda))

# Save regularized movie biases
save(movie_bias_reg, file = "rdas/movie_bias_reg.rda")

# Calculate regularized user biases
user_bias_reg <- train_set %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))

# Save regularized user biases
save(user_bias_reg, file = "rdas/user_bias_reg.rda")

# Generate predictions using regularized movie and user biases
predicted_ratings <- test_set %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  mutate(pred = mu + b_i_reg + b_u_reg) %>%
  .$pred

# Calculate RMSE with regularized biases
rmse_reg_biases <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results, c("Regularized biases", rmse_reg_biases))
rm(rmse_reg_biases)

# Explore genres

# Inspect individual genres
train_set %>%
  group_by(genres) %>%
  .$genres %>%
  unique() %>%
  str_split("\\|") %>%
  unlist() %>%
  unique()

# Determine max number of genres for any movie
train_set %>%
  group_by(genres) %>%
  .$genres %>%
  unique() %>%
  str_count("\\|") %>%
  max() + 1

# Calculate genre biases
genre_bias <- train_set %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i_reg - b_u_reg) - mu)

# Save genre biases
save(genre_bias, file = "rdas/genre_bias.rda")

# Examine highest and lowest genre biases
genre_bias %>%
  arrange(desc(b_g)) %>%
  head(10)

genre_bias %>%
  arrange(b_g) %>%
  head(10)

# Plot distribution of genre biases
genre_bias %>%
  ggplot(aes(b_g)) +
  geom_histogram(binwidth = 0.1, color = "black", fill = "firebrick4") +
  xlab("Genre rating bias") +
  ggtitle("Distribution of genre rating biases")

# Save plot
ggsave("figs/genre_bias_dist.png")

# Generate predictions using regularized movie/user biases and genre bias
predicted_ratings <- test_set %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  left_join(genre_bias, by = "genres") %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_g) %>%
  .$pred

# Calculate RMSE
rmse_regs_plus_genres <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results,
                      c("Reg movie/user + genre", rmse_regs_plus_genres))
rm(rmse_regs_plus_genres)

# Explore temporal effects

# Extract time data from train set
train_time <- train_set %>%
  
  # Convert rating time stamp to datetime format
  mutate(date = as_datetime(timestamp)) %>%
  
  # Extract year, month, and hour from rating date
  mutate(year = year(date), month = month(date), hour = hour(date)) %>%
  
  # Extract movie release year from title
  mutate(movie_year = str_extract(title, "\\d{4}\\)$")) %>%
  
  # Remove end parenthesis and convert movie release year to integer
  mutate(movie_year = as.integer(str_remove(movie_year, "\\)"))) %>%
  
  # Calculate number of years between movie release and rating
  mutate(years_old = year - movie_year) %>%
  
  # Remove unnecessary columns
  select(-timestamp:-genres)
  
# Calculate residuals after removing movie, user, and genre biases
residuals <- train_set %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  left_join(genre_bias, by = "genres") %>%
  mutate(resid = rating - mu - b_i_reg - b_u_reg - b_g) %>%
  .$resid

# Add a column to train set with residuals
train_time <- train_time %>%
  mutate(resid = residuals)
rm(residuals)

# Plot relationship between rating hour and average residual
train_time %>%
  group_by(hour) %>%
  summarize(avg_resid = mean(resid)) %>%
  ggplot(aes(hour, avg_resid)) +
  geom_point(alpha = 0.5, size = 3, shape = 21, fill = "turquoise4") +
  ggtitle("Rating variation by hour") +
  scale_x_continuous(name = "Hour of day", breaks = seq(0, 20, 4), labels =
                       c("midnight", "4 am", "8 am", "noon", "4 pm", "8 pm")) +
  ylab("Average residual")

# Save plot
ggsave("figs/hour_effect.png")

# Plot relationship between rating month and average residual
train_time %>%
  group_by(month) %>%
  summarize(avg_resid = mean(resid)) %>%
  ggplot(aes(month, avg_resid)) +
  geom_point(alpha = 0.5, size = 3, shape = 21, fill = "sienna3") +
  ggtitle("Rating variation by month") +
  scale_x_continuous(name = "Month", breaks = 1:12, labels = month.abb) +
  scale_y_continuous(name = "Average residual") +
  theme(panel.grid.minor = element_blank())

# Save plot
ggsave("figs/month_effect.png")

# Plot relationship between movie age and average residual
train_time %>%
  group_by(years_old) %>%
  summarize(avg_resid = mean(resid), n = n()) %>%
  ggplot(aes(years_old, avg_resid, size = n)) +
  geom_point(alpha = 0.5, shape = 21, fill = "darkorchid3") +
  labs(size = "number of ratings") +
  theme(legend.position = c(0.43, 0.27)) +
  ggtitle("Time effect on rating") +
  xlab("Years since movie released") +
  ylab("Average residual")

# Save plot
ggsave("temporal_effect.png")

# Store average residuals in time bias tibble
time_bias <- train_time %>%
  group_by(years_old) %>%
  summarize(b_t = mean(resid))

# Save time biases
save(time_bias, file = "rdas/time_bias.rda")

# Extract time data from test set
test_time <- test_set %>%
  
  # Convert rating time stamp to datetime format
  mutate(date = as_datetime(timestamp)) %>%
  
  # Extract year from rating date
  mutate(year = year(date)) %>%
  
  # Extract movie release year from title
  mutate(movie_year = str_extract(title, "\\d{4}\\)$")) %>%
  
  # Remove end parenthesis and convert movie release year to integer
  mutate(movie_year = as.integer(str_remove(movie_year, "\\)"))) %>%
  
  # Calculate number of years between movie release and rating
  mutate(years_old = year - movie_year) %>%
  
  # Select necessary columns
  select(userId, movieId, rating, genres, years_old)

# Generate predictions using movie, user, genre, and time biases
predicted_ratings <- test_time %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  left_join(genre_bias, by = "genres") %>%
  left_join(time_bias, by = "years_old") %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_g + b_t) %>%
  .$pred

# Calculate RMSE
rmse_regs_genre_time <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results, c("Reg movie/user + genre + movie age",
                                      rmse_regs_genre_time))
rm(rmse_regs_genre_time)

# Examine range of predicted ratings
range(predicted_ratings)

# Set predictions below min and above max to 0.5 and 5 respectively
predicted_ratings <- ifelse(predicted_ratings < 0.5, 0.5, predicted_ratings)
predicted_ratings <- ifelse(predicted_ratings > 5, 5, predicted_ratings)

# Calculate new RMSE
rmse_regs_genre_time_adj <- RMSE(predicted_ratings, test_set$rating)

# Add RMSE to methods tibble
rmse_results <- rbind(rmse_results,
                      c("Final algorithm", rmse_regs_genre_time_adj))
rm(rmse_regs_genre_time_adj, test_set, test_time, train_set, train_time)

############################
# Final algorithm and test #
############################

# Load final test object
load("rdas/validation.rda")

# Extract time data from final test set
validation <- validation %>%
  
  # Convert time stamp from rating to datetime format
  mutate(date = as_datetime(timestamp)) %>%
  
  # Extract year from date
  mutate(year = year(date)) %>%
  
  # Extract movie release year from title
  mutate(movie_year = str_extract(title, "\\d{4}\\)$")) %>%
  
  # Remove end parenthesis and convert movie release year to integer
  mutate(movie_year = as.integer(str_remove(movie_year, "\\)"))) %>%
  
  # Calculate number of years between movie release and rating
  mutate(years_old = year - movie_year) %>%
  
  # Select necessary columns 
  select(userId, movieId, rating, title, genres, years_old)

# Make final predictions
predicted_ratings <- validation %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  left_join(genre_bias, by = "genres") %>%
  left_join(time_bias, by = "years_old") %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_g + b_t) %>%
  mutate(pred = ifelse(pred < 0.5, 0.5, pred)) %>%
  mutate(pred = ifelse(pred > 5, 5, pred)) %>%
  .$pred

# Save final predictions
save(predicted_ratings, file = "rdas/final_predictions.rda")

# Calculate final RMSE
rmse_final <- RMSE(predicted_ratings, validation$rating)
rmse_final

# Add final RMSE to results tibble
rmse_results <- rbind(rmse_results, c("Final test", rmse_final))

# Save rmse results tibble
save(rmse_results, file = "rdas/rmse_results.rda")

# Add prediction and error columns to validation set
validation <- validation %>%
  mutate(pred = predicted_ratings, error = abs(pred - rating))

# Save modified validation set
save(validation, file = "rdas/validation_2.rda")

# Plot distribution of errors
validation %>%
  ggplot(aes(error)) +
  geom_histogram(aes(y = stat(width * density)), breaks = seq(0, 4.5, 0.25),
                 color = "black", fill = "orangered3") +
  xlab("| prediction - rating |") +
  scale_y_continuous(breaks = seq(0, 0.25, 0.05),
                     labels = paste0(seq(0, 25, 5), "%")) +
  ylab("percentage") +
  ggtitle("Prediction error distribution")

# Save plot
ggsave("error_distribution.png")

# Summarize errors
summary(validation$error)

# Examine greatest errors
validation %>%
  arrange(desc(error)) %>%
  select(pred, rating, error, title) %>%
  head(10)

# Examine movies with greatest average errors
validation %>%
  group_by(title) %>%
  summarize(avg_rating = mean(rating), avg_error = mean(error), n = n()) %>%
  arrange(desc(avg_error)) %>%
  head(10)

# Examine users with greatest average errors
validation %>%
  group_by(userId) %>%
  summarize(avg_rating = mean(rating), avg_error = mean(error), n = n()) %>%
  arrange(desc(avg_error)) %>%
  head(10)

# Examine genres with greatest average errors
validation %>%
  group_by(genres) %>%
  summarize(avg_rating = mean(rating), avg_error = mean(error), n = n()) %>%
  arrange(desc(avg_error)) %>%
  head(10)

# Examine movie ages with greatest average errors
validation %>%
  group_by(years_old) %>%
  summarize(avg_rating = mean(rating), avg_error = mean(error), n = n()) %>%
  arrange(desc(avg_error)) %>%
  head(10)
