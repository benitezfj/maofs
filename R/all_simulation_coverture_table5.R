# Calcular pairwise matrix comparative con los datos de la metrica de covertura
coverage <- function(X_prime, X_double_prime) {
  if(!is.matrix(X_prime)){
    X_prime <- t(X_prime)
  }
  if(!is.matrix(X_double_prime)){
    X_double_prime <- t(X_double_prime)
  }

  coverage_points <- 0
  for (i in seq_len(nrow(X_double_prime))) {
    for (j in seq_len(nrow(X_prime))) {
      if (all(X_double_prime[i,] >= X_prime[j,]) && any(X_double_prime[i,] > X_prime[j,])) {
        coverage_points <- coverage_points + 1
        break
      }
    }
  }
  return(coverage_points / nrow(X_double_prime))
}

# test <- function(dataset_names, models, algorithms, ideal_point, nadir_point){
test <- function(all_features_dataset, dataset_names, models, algorithms){
  fitness_pattern <- "%s_fitness_%s_maop_%s.csv"
  # all_features_pattern <- "all_features_%s_%s"

  for (dataset in dataset_names) {
    for (model in models) {
      n <- length(algorithms)+1
      coverture_matrix <- matrix(0, nrow = n, ncol = n)
      # all_feature_name <- sprintf(all_features_pattern, dataset, model)

      for (i in seq(length(algorithms))) {
        all_features <- all_features_dataset[all_features_dataset$Dataset==dataset & all_features_dataset$Model==model,]
        all_features <- c(all_features$ACC, all_features$NFS, all_features$MI, all_features$MACRO_F1)
        var_name_i <- paste0(dataset,"_",algorithms[i],"_",model)
        dataset_name_i <- sprintf(fitness_pattern, dataset, algorithms[i], model)

        content_i <- readLines(dataset_name_i)

        fitness_values_i <- character()

        for (line in content_i) {
          if (nchar(trimws(line)) > 0) {
            fitness_values_i <- c(fitness_values_i, line)
          }
        }

        fitness_values_i <- unique(fitness_values_i)
        matrix_fitness_i <- do.call(rbind, lapply(fitness_values_i, function(x) as.numeric(strsplit(x, ",")[[1]])))


        assign(var_name_i, matrix_fitness_i)
        # assign(var_name_i, (matrix_fitness_i - min(ideal_point)) / (max(nadir_point) - min(ideal_point)))

        for (j in seq(length(algorithms))) {

          var_name_j <- paste0(dataset,"_",algorithms[j],"_",model)
          dataset_name_j <- sprintf(fitness_pattern, dataset, algorithms[j], model)

          content_j <- readLines(dataset_name_j)

          fitness_values_j <- character()

          for (line in content_j) {
            if (nchar(trimws(line)) > 0) {
              fitness_values_j <- c(fitness_values_j, line)
            }
          }

          fitness_values_j <- unique(fitness_values_j)
          matrix_fitness_j <- do.call(rbind, lapply(fitness_values_j, function(x) as.numeric(strsplit(x, ",")[[1]])))

          assign(var_name_j, matrix_fitness_j)

          if (i > j) {
            coverture_matrix[i, j] <- coverage(get(var_name_i), get(var_name_j))
            coverture_matrix[j, i] <- coverage(get(var_name_j), get(var_name_i))
          }

          coverture_matrix[length(algorithms)+1, j] <- coverage(all_features, get(var_name_j))

        }

        coverture_matrix[i, length(algorithms)+1] <- coverage(get(var_name_i), all_features)

      }
      coverture_df <- as.data.frame(coverture_matrix)

      # rownames(coverture_df) <- c(algorithms, all_feature_name)
      rownames(coverture_df) <- c(algorithms, paste0("all_features_",dataset, "_", model))
      # colnames(coverture_df) <- c(algorithms, all_feature_name)
      colnames(coverture_df) <- c(algorithms, paste0("all_features_",dataset, "_", model))

      cat(dataset,"_",model,"\n")
      print(coverture_df)
      cat("\n")
      matrix_name <- paste0("coverture_",dataset,"_",model,".csv")

      # write.table(coverture_df,
      write.csv(coverture_df,
                file = matrix_name,
                row.names = c(algorithms, paste0("all_features_",dataset, "_", model)),
                col.names = c(algorithms, paste0("all_features_",dataset, "_", model)))
                # sep=",")
    }
  }
}

dataset_names <- c("win10", "win7", "linux_process", "linux_disk", "linux_memory","network")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")

ideal_point <- c(-1,0,-1,-1)
nadir_point <- c(0.5,1.5,0.5,0.5)
worst_point <- c(0,1,0,0)
reference_point <- nadir_point

all_features <- read_csv("all_features_fitness.csv", col_names = TRUE)

test(all_features,dataset_names,models,algorithms)
