library(tidyverse)
library(rmoo)
library(reticulate)
library(caret)

myenvs <- conda_list()

envname <- myenvs$name[2]
use_condaenv(envname, required = TRUE)

np <- import("numpy")
# pd <- import("pandas")
sklearn <- import("sklearn")

optimize_model <- function(model_name, dataset_name, algorithm_name, num_iterations, problem) {
  datasets_path <- NULL
  for (dir in c(getwd(), list.dirs(getwd(), recursive = TRUE))) {
    if (dir.exists(file.path(dir, "datasets"))) {
      datasets_path <- file.path(dir, "datasets")
      break
    }
  }

  datasets <- load_and_prepare_data(datasets_path, dataset_name)
  features <- datasets$features
  classes <- datasets$classes

  X_train <- datasets$X_train
  y_train <- datasets$y_train
  X_test <- datasets$X_test
  y_test <- datasets$y_test

  datasets_subset <- calculate_mutual_info(features, classes)

  mutual_info_selected <- datasets_subset$mutual_info_selected
  features_subset <- datasets_subset$features_subset

  X_train <- X_train[colnames(features_subset)]
  X_test <- X_test[colnames(features_subset)]

  if (model_name == "knn") {
    knn <- sklearn$neighbors$KNeighborsClassifier()
  } else if (model_name == "gnb") {
    gnb <- sklearn$naive_bayes$GaussianNB()
  }

  ref_dirs <- generate_reference_points(4, 7)

  optimize_and_save(
    ref_dirs,
    dataset_name,
    algorithm_name,
    model_name,
    num_iterations,
    problem,
    mutual_info_selected,
    X_train,
    y_train,
    X_test,
    y_test
  )
}


optimize_and_save <- function(ref_dirs, dataset_name, algorithm_name, model_name, num_iterations, problem, mutual_info, X_train, y_train, X_test, y_test) {
  fitness_file <- paste0(dataset_name, "_fitness_", algorithm_name, "_maop_", model_name, ".csv")
  solution_file <- paste0(dataset_name, "_solution_", algorithm_name, "_maop_", model_name, ".csv")

  f <- file(fitness_file, open = "w")
  g <- file(solution_file, open = "w")

  writerFitness <- csv.writer(f)
  writerPopulation <- csv.writer(g)

  writerFitness.writerow(c("ACC", "NFS", "MI", "MacroF1"))


  for (i in seq_len(num_iterations)) {
    res <- rmoo(
      type = "binary",
      fitness = problem,
      strategy = algorithm_name,
      nBits = ncol(X_train),
      popSize = 120,
      selection = selection,
      population = population,
      reference_dirs = ref_dirs,
      nObj = 4,
      pcrossover = 0.8,
      pmutation = 0.1,
      maxiter = 90,
      monitor = monitortest,
      parallel = FALSE,
      summary = FALSE,
      X_train = X_train,
      X_test = X_test,
      y_train = y_train,
      y_test = y_test,
      mutual_info = mutual_info,
      estimator = get(model_name)
    )

    unique_fitness <- unique(res@fitness[res@f[[1]], ])
    unique_population <- unique(res@population[res@f[[1]], ])

    writerFitness.writerows(unique_fitness)
    writerPopulation.writerows(unique_population)

    writerMetric.writerow("")
    writerFitness.writerow("")
    writerPopulation.writerow("")
  }

  close(f)
  close(g)
}


load_and_prepare_data <- function(datasets_path, dataset_name, seed = 123) {
  dataset_path <- file.path(datasets_path, paste0(dataset_name, "_normalize.csv"))
  dataset <- read_csv(dataset_path)

  set.seed(seed)
  train_index <- createDataPartition(dataset$type, p = 0.7, list = FALSE)
  dataset_train <- dataset[train_index, ]
  dataset_test <- dataset[-train_index, ]

  if (dataset_name == "linux_memory" || dataset_name == "linux_disk" || dataset_name == "win7") {
    features <- dataset %>% select(-type)
    X_train <- dataset_train %>% select(-type)
  } else if (dataset_name == "network" || dataset_name == "linux_process" || dataset_name == "win10") {
    features <- dataset %>% select(-type, -label)
    X_test <- dataset_test %>% select(-type, -label)
  }

  return(list(
    features = features,
    classes = as.factor(dataset$type),
    X_train = X_train,
    y_train = as.factor(dataset_train$type),
    X_test = X_test,
    y_test = as.factor(dataset_test$type)
  ))
}

calculate_mutual_info <- function(features, classes, seed = 42) {
  np$random$seed(as.integer(seed))
  mutual_info <- sklearn$feature_selection$mutual_info_classif(features, classes)
  np$random$seed(NULL)

  mutual_info_selected <- mutual_info[mutual_info > 0]
  features_subset <- features[, mutual_info > 0]

  return(list(
    mutual_info_selected = mutual_info_selected,
    features_subset = features_subset
  ))
}

macro_f1 <- function(act, prd) {
  # Create a data frame of actual and predicted labels
  df <- data.frame(act = act, prd = prd)
  # Initialize a vector to store the class-wise F1 scores
  f1 <- numeric()
  # Loop over the unique classes
  for (i in unique(act)) {
    # Calculate true positives, false positives and false negatives
    tp <- nrow(df[df$prd == i & df$act == i, ])
    fp <- nrow(df[df$prd == i & df$act != i, ])
    fn <- nrow(df[df$prd != i & df$act == i, ])
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
      clf$fit(X_train[, x], y_train)
      y_pred <- clf$predict(X_test[, x])
      acc <- mean(y_pred == y_test)

      mafs <- macro_f1(act = y_test, prd = y_pred)
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

monitortest <- function(object, number_objectives, ...) {
  iter <- object@iter
  cat("Iter:", object@iter, ", ")
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
  out <- list(
    population = object@population[sel, ],
    fitness = object@fitness[sel, ]
  )
  return(out)
}

population <- function(object) {
  population <- matrix(NA_real_,
    nrow = object@popSize,
    ncol = object@nBits
  )
  for (i in 1:object@popSize) {
    population[i, ] <- round(runif(object@nBits))
    if (all(population[i, ] == 0)) population[i, ][sample.int(length(population[i, ]), 1)] <- 1
  }
  storage.mode(population) <- "integer"
  return(population)
}

model_names <- c("gnb", "knn")
dataset_aliases <- c("linux_memory", "linux_disk", "linux_process", "network", "win7", "win10")
algorithm_aliases <- c("NSGA-II", "NSGA-III")
num_iterations <- 12

for (model_name in model_names) {
  for (dataset_name in dataset_aliases) {
    for (algorithm_name in algorithm_aliases) {
      optimize_model(
        model_name,
        dataset_name,
        algorithm_name,
        num_iterations,
        featureSelectionManyProblem
      )
    }
  }
}
