library(caret)
library(tidyverse)
library(rmoo)
library(reticulate)
library(class)
library(e1071)

myenvs=conda_list()

envname=myenvs$name[2]
use_condaenv(envname, required = TRUE)

np <- import("numpy")
pd <- import("pandas")
sklearn <- import("sklearn")


dataset <- read_csv('win10_normalize.csv',
                    col_names = TRUE)

features <- dataset %>% select(-type,-label) #Network, win10, linux_process
classes <- dataset$type
classes <- as.factor(classes)

# Calculate the mutual information
np$random$seed(as.integer(42))
features.mutual_info <- sklearn$feature_selection$mutual_info_classif(features, classes)
np$random$seed(NULL)

# --------------------Solo para Network Dataset-----------------------
set.seed(42)
index_list <- sample(1:nrow(dataset), 80000, replace = TRUE)
rm(.Random.seed, envir=globalenv())

dataset <- dataset[index_list, ]

features <- dataset %>% select(-type,-label)
# features <- dataset %>% select(-type)
classes <- dataset$type
classes <- as.factor(classes)
# --------------------------------------------------------------------


set.seed(123)
train_index <- createDataPartition(dataset$type, p = 0.7, list = FALSE)
rm(.Random.seed, envir=globalenv())

dataset_train <- dataset[train_index,]
dataset_test <- dataset[-train_index,]


# X_train <- dataset_train %>% select(-type)
X_train <- dataset_train %>% select(-type,-label)
#X_train <- X_train[,features.mutual_info>0]
y_train <- dataset_train$type
y_train <- as.factor(y_train)

# X_test <- dataset_test %>% select(-type)
X_test <- dataset_test %>% select(-type,-label)
#X_test <- X_test[,features.mutual_info>0]
y_test <- dataset_test$type
y_test <- as.factor(y_test)


features.mutual_info <- features.mutual_info[features.mutual_info>0]

knn=sklearn$neighbors$KNeighborsClassifier()
gnb=sklearn$naive_bayes$GaussianNB()
model <- naiveBayes
# model <- naiveBayes(type ~ ., data = dataset_train)

macro_f1 <- function(act, prd) {
  # Create a data frame of actual and predicted labels
  df <- data.frame(act = act, prd = prd)
  # Initialize a vector to store the class-wise F1 scores
  f1 <- numeric()
  # Loop over the unique classes
  for (i in unique(act)) {
    # Calculate true positives, false positives and false negatives
    tp <- nrow(df[df$prd == i & df$act == i,])
    fp <- nrow(df[df$prd == i & df$act != i,])
    fn <- nrow(df[df$prd != i & df$act == i,])
    # Calculate precision, recall and F1 score for each class
    prec <- tp / (tp + fp)
    rec <- tp / (tp + fn)
    f1[i] <- 2 * prec * rec / (prec + rec)
    # Replace NA values with zero
    f1[is.na(f1)] <- 0
  }
  # Return the macro F1 score as the mean of class-wise F1 scores
  return(mean(f1))
}

featureSelectionManyProblem <- function(x, X_train, X_test, y_train, y_test, mutual_info, estimator, ...) {
  x <- as.logical(x)
  feature_costs <- rep(1, ncol(X_train))

  validation <- function(x, X_train, X_test, y_train, y_test, estimator) {
    clf <- sklearn$clone(estimator)
    if (all(!x)) {
      metrics <- metrics1 <- 0
      return(list(metrics = metrics, metrics1 = metrics1))
    } else {
      clf$fit(X_train[,x], y_train)
      y_pred <- clf$predict(X_test[,x])
      acc <- mean(y_pred == y_test)

      # mafs <- sklearn$metrics$f1_score(y_test, y_pred, labels=unique(y_train), average='macro')
      mafs <- macro_f1(act=y_test, prd=y_pred)
      return(list(metrics = acc, metrics1 = mafs))
    }
  }

  scores_list <- validation(x, X_train, X_test, y_train, y_test, estimator)
  acc <- scores_list$metrics
  mafs <- scores_list$metrics1

  costs_selected <- feature_costs[which(x)]
  cost_sum <- sum(costs_selected) / sum(feature_costs)
  mutual_info_costs <- sum(mutual_info[which(x)]) / sum(mutual_info)

  if (cost_sum == 0) {
    out <- cbind(0, 0, 0, 0)
  } else {
    f1 <- -1 * acc
    f2 <- cost_sum
    f3 <- -1 * mutual_info_costs
    f4 <- -1 * mafs
    out <- cbind(f1, f2, f3, f4)
  }
  return(as.vector(out))
}


selection <- function(object, k = 2, ...) {
  popSize <- object@popSize
  front <- object@front
  fit <- object@fitness
  sel <- rep(NA, popSize)
  for (i in 1:popSize) {
    s <- sample(1:popSize, size = k)
    s <- s[which.min(front[s, ])]
    if (length(s) > 1 & !anyNA(fit[s, ])) {
      sel[i] <- s[which.max(front[s, ])]
    } else {
      sel[i] <- s[which.min(front[s, ])]
    }
  }
  out <- list(population = object@population[sel, ],
              fitness = object@fitness[sel, ])
  return(out)
}

population <- function(object) {
  population <- matrix(NA_real_,
                       nrow = object@popSize,
                       ncol = object@nBits)
  for (i in 1:object@popSize) {
    population[i, ] <- round(runif(object@nBits))
    if (all(population[i, ] == 0)) population[i, ][sample.int(length(population[i, ]),1)] <- 1
  }
  storage.mode(population) <- "integer"
  return(population)
}

# To superficially see the evolution of the execution, we create a custum monitor function
monitortest <- function(object, number_objectives, ...) {
  iter <- object@iter
  cat("Iter:", object@iter, ", ")
}

# Define the reference points
reference_dirs <- generate_reference_points(4,7)
reference_point <- apply(reference_dirs, 2, max)

dataset_name <- "linux_process"
algorithm <- "NSGA-III"
model <- "knn"

# Use tidyverse
# Open the files for writing
f <- file(paste0(dataset_name,"_fitness_",algorithm,"_maop_",model,".csv"), "w")
g <- file(paste0(dataset_name,"_solution_",algorithm,"_maop_",model,".csv"), "w")
h <- file(paste0(dataset_name,"_metrics_",algorithm,"_maop_",model,".csv"), "w")

# Write the column names
writeLines(c("ACC,NFS,MI,MacroF1"), f)
writeLines(c("GD,IGD,HV"), h)

# Write the data
for (i in 1:12) {

  maofs  <- rmoo(type = "binary",
                fitness = featureSelectionManyProblem,
                strategy = algorithm,
                nBits = ncol(X_train),
                popSize = 120,
                selection=selection,
                population = population,
                reference_dirs = reference_dirs,
                nObj = 4,
                maxiter = 90,
                monitor = monitortest,
                parallel = FALSE,
                summary = FALSE,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                mutual_info = features.mutual_info,
                estimator = get(model))

  igd <- ecr::computeInvertedGenerationalDistance(t(unique(maofs@fitness[maofs@f[[1]],])),
                                                  t(reference_dirs))
  gd  <- ecr::computeGenerationalDistance(t(unique(maofs@fitness[maofs@f[[1]],])),
                                          t(reference_dirs))
  hv  <- emoa::dominated_hypervolume(points = t(unique(maofs@fitness[maofs@f[[1]],])),
                                     ref = reference_point)
  writeLines(as.character(c(gd, igd, hv)), h, sep = ",")

  unique_fitness <- unique(maofs@fitness[maofs@f[[1]],])

  for (j in seq_len(nrow(unique_fitness))) {
    writeLines(as.character(unique_fitness[j,]), f, sep = ",")
    writeLines("\n", f)
  }

  unique_population <- unique(maofs@population[maofs@f[[1]],])

  for (j in seq_len(nrow(unique_population))) {
    writeLines(as.character(unique_population[j,]), g, sep = ",")
    writeLines("\n", g)
  }

  writeLines("\n", f)
  writeLines("\n", f)
  writeLines("\n", g)
  writeLines("\n", g)
  writeLines("\n", h)

  cat("\n","----",i,"----","\n")

}

# Close the files
close(f)
close(g)
close(h)



