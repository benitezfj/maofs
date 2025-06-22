setwd('C:/Users/Maria/Documents/R/Notebooks/Result Smote')
setwd('C:/Users/Maria/Documents/Python Scripts/resultados 90 iteraciones/toniot')

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


# Function to normalize data frame columns to range [0, 1]
normalize_data <- function(data) {
  normalized_data <- apply(data, 2, function(column) {
    (column - min(column)) / (max(column) - min(column))
  })
  return(as.data.frame(normalized_data))
}

macro_f1 <- function(act, prd) {

  df <- data.frame(act = act, prd = prd)

  f1 <- numeric()

  for (i in unique(act)) {

    tp <- nrow(df[df$prd == i & df$act == i,])
    fp <- nrow(df[df$prd == i & df$act != i,])
    fn <- nrow(df[df$prd != i & df$act == i,])

    prec <- tp / (tp + fp)
    rec <- tp / (tp + fn)
    f1[i] <- 2 * prec * rec / (prec + rec)

    f1[is.na(f1)] <- 0
  }

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

# Calcular los fitness de todos los features
calculate_all_features <- function(dataset_names, algorithms, models, seed_1=42, seed_2=123){
  knn=sklearn$neighbors$KNeighborsClassifier()
  gnb=sklearn$naive_bayes$GaussianNB()
  results <- list()
  for (data in dataset_names) {
      pattern <- "%s_normalize.csv"
    dataset_name <- sprintf(pattern, data)
    dataset <- read_csv(dataset_name, col_names = TRUE)

    if (data=="linux_memory" || data=="linux_disk" || data=="win7") {
      features <- dataset %>% select(-type) #linux_memory, linux_disk, win7
    }else if (data=="network" || data=="linux_process" || data=="win10") {
      features <- dataset %>% select(-type,-label) #Network, win10, linux_process
    }

    classes <- dataset$type
    classes <- as.factor(classes)

    np$random$seed(as.integer(seed_1))
    features.mutual_info <- sklearn$feature_selection$mutual_info_classif(features, classes)
    np$random$seed(NULL)

    set.seed(seed_2)
    train_index <- createDataPartition(dataset$type, p = 0.7, list = FALSE)
    rm(.Random.seed, envir=globalenv())

    dataset_train <- dataset[train_index,]
    dataset_test <- dataset[-train_index,]


    if (data=="linux_memory" || data=="linux_disk" || data=="win7") {
      X_train <- dataset_train %>% select(-type)
      X_test <- dataset_test %>% select(-type)
    }else if (data=="network" || data=="linux_process" || data=="win10") {
      X_train <- dataset_train %>% select(-type,-label)
      X_test <- dataset_test %>% select(-type,-label)
    }

    X_train <- X_train[,features.mutual_info>0]
    y_train <- dataset_train$type
    y_train <- as.factor(y_train)

    X_test <- X_test[,features.mutual_info>0]
    y_test <- dataset_test$type
    y_test <- as.factor(y_test)


    features.mutual_info <- features.mutual_info[features.mutual_info>0]

    x <- as.logical(round(rep(1,ncol(X_train))))
    # x <- as.logical(round(runif(ncol(X_train))))

    for (model in models) {
      all_feature <- featureSelectionManyProblem(x = rep(1, length(features.mutual_info)),
                                                 X_train = X_train,
                                                 X_test = X_test,
                                                 y_train = y_train,
                                                 y_test = y_test,
                                                 mutual_info = features.mutual_info,
                                                 estimator = get(model))

      all_feature <- data.frame(Dataset = data,
                                Model = model,
                                ACC = all_feature[1],
                                NFS = all_feature[2],
                                MI = all_feature[3],
                                MACRO_F1 = all_feature[4])

      print(all_feature)

      results[[length(results) + 1]] <- all_feature

    }
    all_results <- do.call(rbind, results)
    print(all_results)

  }
  write.csv(all_results, "all_features_fitness.csv", row.names = FALSE)
}

calculate_distance <- function(data_point, ref_point) {
  distance <- 1
  for (i in seq_along(ref_point)) {
    distance <- distance * (ref_point[i] - data_point[i])
  }
  return(distance)
}

calculate_metrics_all_features <- function(all_features_dataset,dataset_names,models,reference_dirs,reference_point){
  all_metrics <- list()
  for (dataset in dataset_names) {
    for (model in models) {
      all_features <- all_features_dataset[all_features_dataset$Dataset==dataset & all_features_dataset$Model==model,]
      all_features <- c(all_features$ACC, all_features$NFS, all_features$MI, all_features$MACRO_F1)

      hv_values <- calculate_distance(data_point = all_features, ref_point = reference_point)
      gd_values <- ecr::computeGenerationalDistance(as.matrix(all_features), t(reference_dirs))
      igd_plus_values <- eaf::igd_plus(data = t(all_features), reference = reference_dirs)

      metrics <- data.frame(Model = model,
                            Dataset = dataset,
                            HV = hv_values,
                            GD = gd_values,
                            IGD = igd_plus_values)

      all_metrics[[length(all_metrics) + 1]] <- metrics
    }
  }
  all_metrics <- do.call(rbind, all_metrics)
  write.csv(all_metrics, "all_features_metrics.csv", row.names = FALSE)

}

# --------------------------------NUEVO 25/5/2025----------------------------------------------
library(caret)
library(tidyverse)
library(rmoo)
library(reticulate)
library(class)
library(e1071)
library(dplyr)
library(readr)
library(ecr)
library(eaf)

# Configurar entorno de Python
myenvs <- conda_list()
use_condaenv(myenvs$name[2], required = TRUE)

# Importar librerías de Python
np <- import("numpy")
pd <- import("pandas")
sklearn <- import("sklearn")

# Funciones auxiliares
normalize_data <- function(data) {
  as.data.frame(lapply(data, function(col) (col - min(col)) / (max(col) - min(col))))
}

macro_f1 <- function(act, prd) {
  f1_scores <- sapply(unique(act), function(i) {
    tp <- sum(act == i & prd == i)
    fp <- sum(act != i & prd == i)
    fn <- sum(act == i & prd != i)
    prec <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    rec <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    f1 <- ifelse((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    return(f1)
  })
  mean(f1_scores)
}

featureSelectionManyProblem <- function(x, X_train, X_test, y_train, y_test, mutual_info, estimator) {
  x <- as.logical(x)
  feature_costs <- rep(1, ncol(X_train))

  if (all(!x)) return(rep(0, 4))

  clf <- sklearn$clone(estimator)
  clf$fit(X_train[, x], y_train)
  y_pred <- clf$predict(X_test[, x])

  acc <- mean(y_pred == y_test)
  mafs <- macro_f1(y_test, y_pred)
  cost_sum <- sum(feature_costs[x]) / sum(feature_costs)
  mutual_info_costs <- sum(mutual_info[x]) / sum(mutual_info)

  c(-acc, cost_sum, -mutual_info_costs, -mafs)
}

# Lógica principal

calculate_all_features <- function(dataset_names, algorithms, models, seed_1 = 42, seed_2 = 123) {
  model_map <- list(knn = sklearn$neighbors$KNeighborsClassifier(),
                    gnb = sklearn$naive_bayes$GaussianNB())

  results <- list()

  for (dataset_name in dataset_names) {
    dataset_path <- sprintf("%s_normalize.csv", dataset_name)
    dataset <- read_csv(dataset_path,show_col_types = FALSE)

    features <- dataset %>% select(-type, -label)

    # Seleccionar columnas de características
    # if (dataset_name %in% c("linux_memory", "linux_disk", "win7")) {
    #   features <- dataset %>% select(-type)
    # } else {
    #   features <- dataset %>% select(-type, -label)
    # }

    classes <- as.factor(dataset$type)

    # Calcular mutual information
    np$random$seed(as.integer(seed_1))
    mutual_info <- sklearn$feature_selection$mutual_info_classif(features, classes)
    np$random$seed(NULL)

    # Particionar datos
    set.seed(seed_2)
    idx_train <- createDataPartition(classes, p = 0.7, list = FALSE)
    rm(.Random.seed, envir = globalenv())

    dataset_train <- dataset[idx_train, ]
    dataset_test  <- dataset[-idx_train, ]

    # Seleccionar características informativas
    informative_cols <- which(mutual_info > 0)
    mutual_info <- mutual_info[informative_cols]

    X_train <- dataset_train[informative_cols]
    X_test  <- dataset_test[informative_cols]
    y_train <- as.factor(dataset_train$type)
    y_test  <- as.factor(dataset_test$type)

    x_init <- rep(1, length(informative_cols))

    for (model in models) {
      estimator <- model_map[[model]]

      metrics <- featureSelectionManyProblem(x = x_init,
                                             X_train = X_train,
                                             X_test = X_test,
                                             y_train = y_train,
                                             y_test = y_test,
                                             mutual_info = mutual_info,
                                             estimator = estimator)

      df_result <- data.frame(Dataset = dataset_name, Model = model,
                              ACC = metrics[1], NFS = metrics[2],
                              MI = metrics[3], MACRO_F1 = metrics[4])

      print(df_result)
      results[[length(results) + 1]] <- df_result
    }
  }

  all_results <- bind_rows(results)
  write_csv(all_results, "all_features_fitness.csv")
  print(all_results)
}

calculate_distance <- function(data_point, ref_point) {
  prod(ref_point - data_point)
}

# Función principal para calcular las métricas
calculate_metrics_all_features <- function(all_features_dataset, dataset_names, models, reference_dirs, reference_point) {
  all_metrics <- lapply(dataset_names, function(dataset) {
    lapply(models, function(model) {
      # Filtrado de datos para el dataset y modelo actuales
      features_row <- all_features_dataset %>%
        filter(Dataset == dataset, Model == model)

      # Extraer las 4 métricas
      all_features <- c(features_row$ACC, features_row$NFS, features_row$MI, features_row$MACRO_F1)

      # Cálculo de métricas
      hv <- calculate_distance(all_features, reference_point)
      gd <- ecr::computeGenerationalDistance(as.matrix(all_features), t(reference_dirs))
      igd <- eaf::igd_plus(data = t(all_features), reference = reference_dirs)

      # Crear data.frame con las métricas
      data.frame(
        Model = model,
        Dataset = dataset,
        HV = hv,
        GD = gd,
        IGD = igd
      )
    })
  })

  # Aplanar la lista y guardar como CSV
  all_metrics_df <- do.call(rbind, unlist(all_metrics, recursive = FALSE))
  write.csv(all_metrics_df, "all_features_metrics.csv", row.names = FALSE)
}
# ------------------------------------------------------------------------------

# Parámetros
dataset_names <- c("win10", "win7", "linux_process", "linux_disk", "linux_memory", "network")
algorithms <- c("rvea", "moead", "nsgaiii", "nsgaii")
models <- c("knn", "gnb")

worst_point <- c(0,1,0,0)
ideal_point <- c(-1,0,-1,-1)
nadir_point <- c(0.5,1.5,0.5,0.5)

# Generar puntos de referencia y escalar columnas
reference_dirs <- reference_dirs_norm <- generate_reference_points(4, 7)
reference_dirs <- sweep(reference_dirs, 2, c(2, 1, 2, 2), FUN = "/") #sweep(reference_dirs, 2, (1, 0.6, 1, 1), FUN = "/")
reference_point <- nadir_point

# Calcular features
calculate_all_features(dataset_names, algorithms, models)
all_features <- read_csv("all_features_fitness.csv")

# Calcular métricas
calculate_metrics_all_features(all_features, dataset_names, models, reference_dirs, reference_point)


