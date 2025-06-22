setwd('C:/Users/Maria/Documents/R/maofs')

library(tidyverse)
library(rmoo)
library(reticulate)
library(caret)

myenvs <- conda_list()

envname <- myenvs$name[2]
use_condaenv(envname, required = TRUE)

np <- import("numpy")
pd <- import("pandas")
sklearn <- import("sklearn")
source_python("C:/Users/Maria/Documents/R/maofs/Python/methods/symmetrical_uncertainty.py") #su_measure
source_python("C:/Users/Maria/Documents/R/maofs/Python/methods/smote_balancing.py")

# Leer y preparar particiones train/test/validation
load_and_prepare_data <- function(datasets_path, dataset_name, seed = 123) {
  dataset_path <- file.path(datasets_path, paste0(dataset_name, "_normalize.csv"))
  dataset <- read_csv(dataset_path, show_col_types = FALSE)

  set.seed(seed)
  train_index <- createDataPartition(dataset$label, p = 0.8, list = FALSE)
  dataset_test <- dataset[-train_index, ]
  dataset_train <- dataset[train_index, ]

  val_index <- createDataPartition(dataset_train$label, p = 0.8, list = FALSE)
  dataset_val <- dataset_train[-val_index, ]
  dataset_train <- dataset_train[val_index, ]

  features <- dataset %>% select(-type, -label)
  X_train <- dataset_train %>% select(-type, -label)
  X_test <- dataset_test %>% select(-type, -label)
  X_val <- dataset_val %>% select(-type, -label)

  return(list(
    features = features,
    classes = as.factor(dataset$label),
    X_train = X_train,
    y_train = as.factor(dataset_train$label),
    X_test = X_test,
    y_test = as.factor(dataset_test$label),
    X_val = X_val,
    y_val = as.factor(dataset_val$label)
  ))
}

# Calcular información mutua usando scikit-learn
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

# Calcular symmetrical uncertainty usando función Python su_measure
calculate_symmetrical_uncertainty <- function(features, classes, seed = 42) {
  np$random$seed(as.integer(seed))
  su_info <- su_measure(features, classes)
  np$random$seed(NULL)

  return(list(su_info_selected = su_info))
}

# Calcular Macro F1 manualmente
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

featureManyProblem <- function(x, X_train, X_test, y_train, y_test, mutual_info, estimator, ...) {
  x <- as.logical(x)
  feature_costs <- rep(1, ncol(X_train))

  validation <- function(x, X_train, X_test, y_train, y_test, estimator) {
    clf <- sklearn$clone(estimator)
    if (all(!x)) {
      metrics <- metrics1 <- 0
      return(list(metrics = metrics, metrics1 = metrics1))
    } else {
      clf$fit(as.matrix(X_train[, x, drop = FALSE]), y_train)
      y_pred <- clf$predict(as.matrix(X_test[, x, drop = FALSE]))
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


# Función de fitness para selección de características (4 objetivos)
featureSelectionManyProblem <- function(x, X_train, X_test, y_train, y_test, mutual_info, estimator, ...) {
  x <- as.logical(x)
  feature_costs <- rep(1, ncol(X_train))

  if (ncol(X_train) != length(mutual_info)) {
    stop(sprintf("ncol(X_train)=%d pero length(mutual_info)=%d", ncol(X_train), length(mutual_info)))
  }

  validation <- function(x, X_train, X_test, y_train, y_test, estimator) {
    clf <- sklearn$clone(estimator)
    if (all(!x)) {
      metrics <- metrics1 <- 0
      return(list(metrics = metrics, metrics1 = metrics1))
    } else {
      clf$fit(as.matrix(X_train[, x, drop = FALSE]), y_train)
      y_pred <- clf$predict(as.matrix(X_test[, x, drop = FALSE]))
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
  cat("Generation:", object@iter, ", ")
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

# Función principal de optimización y guardado
optimize_and_save <- function(ref_dirs, dataset_name, algorithm_name, model_name, num_iterations,
                              problem, mutual_info, X_train, y_train, X_test, y_test, X_val, y_val) {

  fitness_file <- paste0(dataset_name, "_fitness_", algorithm_name, "_", model_name, ".csv")
  solution_file <- paste0(dataset_name, "_solution_", algorithm_name, "_", model_name, ".csv")
  metric_file <- paste0(dataset_name, "_metric_", algorithm_name, "_", model_name, ".csv")
  evaluation_file <- paste0(dataset_name, "_evaluation_", algorithm_name, "_", model_name, ".csv")

  if (model_name == "knn") {
    knn <- sklearn$neighbors$KNeighborsClassifier()
  } else if (model_name == "gnb") {
    gnb <- sklearn$naive_bayes$GaussianNB()
  } else if (model_name == "rfost") {
    rfost <- sklearn$ensemble$RandomForestClassifier()
  } else if (model_name == "tree") {
    tree <- sklearn$tree$DecisionTreeClassifier()
  }

  # Inicializar listas para almacenar los datos
  fitness_data <- list()
  solution_data <- list()
  metric_data <- list()
  evaluation_data <- list()

  cat("\n Runtime Configuration:\n")
  cat(" - Dataset:", dataset_name, "\n")
  cat(" - Evolutionary Algorithm:", algorithm_name, "\n")
  cat(" - Model:", model_name, "\n")
  cat(" - Population: 120\n - Crossover: 0.8\n - Mutation: 0.1\n - Generations: 100\n")
  cat(" - Train Set: ", nrow(X_train), "instances,", ncol(X_train), "features\n")
  cat("   - Class distribution:\n")
  print(table(y_train))

  cat(" - Test Set: ", nrow(X_test), "instances,", ncol(X_test), "features\n")
  cat("   - Class distribution:\n")
  print(table(y_test))

  cat(" - Validation Set: ", nrow(X_val), "instances,", ncol(X_val), "features\n")
  cat("   - Class distribution:\n")
  print(table(y_val))

  for (i in seq_len(num_iterations)) {

    cat("\n[Iteration", i, "] \n")
    cat(" - Seed:", i, "\n")

    res <- tryCatch({
      rmoo(
        type = "binary",
        fitness = problem,
        algorithm = algorithm_name,
        nBits = ncol(X_train),
        popSize = 120,
        selection = selection,
        population = population,
        reference_dirs = ref_dirs,
        nObj = 4,
        pcrossover = 0.8,
        pmutation = 0.1,
        maxiter = 100,
        monitor = monitortest,
        parallel = FALSE,
        summary = FALSE,
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        mutual_info = mutual_info,
        estimator = get(model_name),
        seed=i
      )
    }, error = function(e) {
      stop(sprintf("Error en rmoo iter %d: %s", iteration, e$message))
    })

    cat("\n")

    igd <- ecr::computeInvertedGenerationalDistance(t(unique(res@fitness[res@f[[1]],])), t(ref_dirs))
    gd  <- ecr::computeGenerationalDistance(t(unique(res@fitness[res@f[[1]],])), t(ref_dirs))
    hv  <- emoa::dominated_hypervolume(points = t(unique(res@fitness[res@f[[1]],])), ref = apply(ref_dirs, 2, max))

    # writerMetric.writerow(paste(gd, igd, hv, sep = ","))
    metric_data[[length(metric_data) + 1]] <- data.frame(GD = gd, IGD = igd, HV = hv)

    unique_fitness <- unique(res@fitness[res@f[[1]], , drop = FALSE])
    unique_population <- unique(res@population[res@f[[1]], , drop = FALSE])

    fitness_data[[length(fitness_data) + 1]] <- as.data.frame(unique_fitness)
    solution_data[[length(solution_data) + 1]] <- as.data.frame(unique_population)

    eval_list <- list()
    for (j in seq_len(nrow(unique_population))) {
      eval_list[[j]] <- featureManyProblem(x=unique_population[j,],
                                       X_train=X_train,
                                       X_test=X_val,
                                       y_train=y_train,
                                       y_test=y_val,
                                       mutual_info = mutual_info,
                                       estimator=get(model_name))

    }

    evaluation_data[[length(evaluation_data) + 1]] <- do.call(rbind, eval_list)

  }
  # Función auxiliar para escribir con separador de ejecución
  write_with_separator <- function(data_list, file_path, col.names = TRUE) {
    con <- file(file_path, open = "w")
    for (data in data_list) {
      write.table(data, file = con, sep = ",", row.names = FALSE, col.names = col.names, append = TRUE)
      writeLines("", con)  # línea vacía como separador
    }
    close(con)
  }

  write_with_separator(fitness_data, fitness_file, col.names = TRUE)
  write_with_separator(solution_data, solution_file, col.names = FALSE)
  write_with_separator(metric_data, metric_file, col.names = TRUE)
  write_with_separator(evaluation_data, evaluation_file, col.names = TRUE)

}

optimize_model <- function(model_name, dataset_name, algorithm_name, num_iterations, problem) {
  # Para funcionar este codigo, cambiar el wd to setwd('C:/Users/Maria/Documents/R/maofs')
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
  X_val <- datasets$X_val
  y_val <- datasets$y_val

  features_pd <- r_to_py(X_train)
  label_pd <- r_to_py(y_train)

  balanced <- balance_smote(features_pd, label_pd, as.integer(42))

  X_train <- py_to_r(balanced[[1]])
  y_train <- py_to_r(balanced[[2]])

  # datasets_subset <- calculate_mutual_info(features, classes)
  # mutual_info_selected <- datasets_subset$mutual_info_selected
  # hist(su_info_selected, breaks = seq(min(su_info_selected), max(su_info_selected), length.out = 50+1))
  mutual_info_selected <- calculate_symmetrical_uncertainty(X_train, y_train) ##Para no cambiar todo el nombre mantengo mutual info

  su_info <- mutual_info_selected$su_info_selected
  su_info_selected <- su_info[su_info>0]

  features_subset <- features[, su_info > 0]
  X_train_subset <- X_train[, su_info > 0]
  X_test_subset <- X_test[, su_info > 0]
  X_val_subset <- X_val[, su_info > 0]

  ref_dirs <- generate_reference_points(4, 7)

  optimize_and_save(
    ref_dirs,
    dataset_name,
    algorithm_name,
    model_name,
    num_iterations,
    problem,
    su_info_selected, # mutual_info_selected,
    X_train_subset,
    y_train,
    X_test_subset,
    y_test,
    X_val_subset,
    y_val
  )
}

model_names <- c("gnb", "knn", "tree", "rfost")
dataset_aliases <- c("linux_memory", "linux_disk", "linux_process", "network", "win7", "win10")
algorithm_aliases <- c("NSGA-II", "NSGA-III")
num_iterations <- 20

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
