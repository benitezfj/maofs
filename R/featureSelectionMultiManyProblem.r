library(caret)
library(tidyverse)
library(rmoo)
library(reticulate)
library(class)
library(e1071)
library(ModelMetrics)
myenvs=conda_list()

envname=myenvs$name[2]
use_condaenv(envname, required = TRUE)

np <- import("numpy")
pd <- import("pandas")
sklearn <- import("sklearn")
source_python("C:/Users/Maria/Documents/R/maofs/Python/methods/symmetrical_uncertainty.py") #su_measure

balance_train <- function(data) {
  label_counts <- table(data$label)
  majority_label <- names(which.max(label_counts))

  balanced_train_df <- rbind(
    data[data$label != majority_label, ],
    data[data$label == majority_label, ][sample(sum(data$label == majority_label), min(label_counts)), ]
  )
  return(balanced_train_df)
}


dataset <- read_csv('C:/Users/Maria/Documents/R/maofs/Python/datasets/win10_normalize.csv',
                    col_names = TRUE)

features <- dataset %>% select(-type,-label) #Network, win10, linux_process
classes <- dataset$label
classes <- as.factor(classes)

# Calculate the mutual information or symmetrical uncentently
np$random$seed(as.integer(42))
# features.mutual_info <- sklearn$feature_selection$mutual_info_classif(features, classes)
features.mutual_info <- su_measure(features, classes)
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
train_index <- createDataPartition(dataset$label, p = 0.7, list = FALSE)
rm(.Random.seed, envir=globalenv())

dataset_test <- dataset[-train_index,]
dataset_train <- dataset[train_index,]

set.seed(123)
train_index <- createDataPartition(dataset_train$label, p = 0.7, list = FALSE)
rm(.Random.seed, envir=globalenv())

dataset_val <- dataset_train[-train_index,]
dataset_train <- dataset_train[train_index,]

dataset_train <- balance_train(dataset_train)


# X_train <- dataset_train %>% select(-label)
X_train <- dataset_train %>% select(-type,-label)
#X_train <- X_train[,features.mutual_info>0]
y_train <- dataset_train$label
y_train <- as.factor(y_train)

# X_test <- dataset_test %>% select(-label)
X_test <- dataset_test %>% select(-type,-label)
#X_test <- X_test[,features.mutual_info>0]
y_test <- dataset_test$label
y_test <- as.factor(y_test)

# X_val <- dataset_val %>% select(-label)
X_val <- dataset_val %>% select(-type,-label)
#X_val <- X_val[,features.mutual_info>0]
y_val <- dataset_val$label
y_val <- as.factor(y_val)


features.mutual_info <- features.mutual_info[features.mutual_info>0]

rfost=sklearn$ensemble$RandomForestClassifier()
knn=sklearn$neighbors$KNeighborsClassifier()
gnb=sklearn$naive_bayes$GaussianNB()
tree=sklearn$tree$DecisionTreeClassifier()
# model <- naiveBayes
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
      recall <- caret::specificity(table(Actual=as.factor(y_test),
                                         Predicted=as.factor(y_pred)))
      # recall <- tnr(actual = y_test,predicted = y_pred)
      # acc <- mean(y_pred == y_test)

      mafs <- macro_f1(act=y_test, prd=y_pred)
      return(list(metrics = recall, metrics1 = mafs))
      # return(list(metrics = acc, metrics1 = mafs))
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


featureManyProblem <- function(x, X_train, X_test, y_train, y_test, mutual_info, estimator, ...) {
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
      recall <- caret::specificity(table(Actual=as.factor(y_test),
                                         Predicted=as.factor(y_pred)))
      mafs <- macro_f1(act=y_test, prd=y_pred)
      return(list(metrics = recall, metrics1 = mafs, metrics2 = acc))
    }
  }

  scores_list <- validation(x, X_train, X_test, y_train, y_test, estimator)
  recall <- scores_list$metrics
  mafs <- scores_list$metrics1
  acc <- scores_list$metrics2

  costs_selected <- feature_costs[which(x)]
  cost_sum <- sum(costs_selected) / sum(feature_costs)
  mutual_info_costs <- sum(mutual_info[which(x)]) / sum(mutual_info)

  if (cost_sum == 0) {
    out <- cbind(0, 0, 0, 0)
  } else {
    f1 <- -1 * recall
    f2 <- cost_sum
    f3 <- -1 * mutual_info_costs
    f4 <- -1 * mafs
    f5 <- -1 * acc
    out <- cbind(f1, f2, f3, f4, f5)
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

dataset_name <- "win10"
algorithm <- "NSGA-II"
model <- "tree"#"rfost", "knn", "rfost"

# Use tidyverse
# Open the files for writing
f <- file(paste0(dataset_name,"_fitness_",algorithm,"_",model,".csv"), "w")
g <- file(paste0(dataset_name,"_solution_",algorithm,"_",model,".csv"), "w")
h <- file(paste0(dataset_name,"_metrics_",algorithm,"_",model,".csv"), "w")
k <- file(paste0(dataset_name,"_evaluation_",algorithm,"_",model,".csv"), "w")

# Write the column names
writeLines(c("Recall,NFS,MI,MacroF1"), f)
writeLines(c("GD,IGD,HV"), h)
writeLines(c("Recall,NFS,MI,MacroF1,ACC"), k)

# Write the data
for (i in 1:5) {
  maofs  <- rmoo(type = "binary",
                fitness = featureSelectionManyProblem,
                strategy = algorithm,
                nBits = ncol(X_train),
                popSize = 120,
                selection=selection,
                population = population,
                reference_dirs = reference_dirs,
                nObj = 4,
                maxiter = 100,
                monitor = monitortest,
                parallel = FALSE,
                summary = FALSE,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                mutual_info = features.mutual_info,
                estimator = get(model),
                seed=i)

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

  for (j in seq_len(nrow(unique_population))) {
    evaluation <- featureManyProblem(x=unique_population[j,],
                                     X_train=X_train,
                                     X_test=X_val,
                                     y_train=y_train,
                                     y_test=y_val,
                                     mutual_info = features.mutual_info,
                                     estimator=get(model))
    writeLines(as.character(evaluation), k, sep = ",")
    writeLines("\n", k)
  }

  writeLines("\n", f)
  writeLines("\n", f)
  writeLines("\n", g)
  writeLines("\n", g)
  writeLines("\n", k)
  writeLines("\n", k)
  writeLines("\n", h)

  cat("\n","----",i,"----","\n")

}

# Close the files
close(f)
close(g)
close(h)
close(k)
