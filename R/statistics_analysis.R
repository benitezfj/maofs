setwd('C:/Users/Maria/Documents/R/Notebooks/Result Smote')

mean_worst_best <- function(dataset_names, models, algorithms){
  pattern <- "%s_fitness_%s_%s.csv"

  results_mean <- list()
  results_worst <- list()
  results_best <- list()

  for (model in models) {
    for (dataset in dataset_names) {
      for (algorithm in algorithms) {
        dataset_name <- sprintf(pattern, dataset, algorithm, model)
        content <- readLines(dataset_name)

        fitness_values <- character()

        for (line in content) {
          if (nchar(trimws(line)) > 0) {
            fitness_values <- c(fitness_values, line)
          }
        }

        fitness_values <- unique(fitness_values)
        matrix_fitness <- do.call(rbind, lapply(fitness_values, function(x) as.numeric(strsplit(x, ",")[[1]])))


        mean_values <- colMeans(matrix_fitness) * 100
        worst_values <- apply(matrix_fitness, 2, max) * 100
        best_values <- apply(matrix_fitness, 2, min) * 100


        result_mean <- data.frame(Model = model,
                             Dataset = dataset,
                             Algorithm = algorithm,
                             ACC = mean_values[1] * -1,
                             NFS = mean_values[2],
                             MI = mean_values[3] * -1,
                             MAF = mean_values[4] * -1)

        result_worst <- data.frame(Model = model,
                                   Dataset = dataset,
                                   Algorithm = algorithm,
                                   ACC = worst_values[1] * -1,
                                   NFS = worst_values[2],
                                   MI = worst_values[3] * -1,
                                   MAF = worst_values[4] * -1)

        result_best <- data.frame(Model = model,
                                  Dataset = dataset,
                                  Algorithm = algorithm,
                                  ACC = best_values[1] * -1,
                                  NFS = best_values[2],
                                  MI = best_values[3] * -1,
                                  MAF = best_values[4] * -1)

        results_mean[[length(results_mean) + 1]] <- result_mean
        results_worst[[length(results_worst) + 1]] <- result_worst
        results_best[[length(results_best) + 1]] <- result_best

        cat(dataset,"_",algorithm,"_",model,"\n")
        cat("\n")

      }
    }

    all_results_mean <- do.call(rbind, results_mean)
    all_results_worst <- do.call(rbind, results_worst)
    all_results_best <- do.call(rbind, results_best)

    matrix_name_mean <- paste0("resume_mean_",model,"_",dataset,".csv")
    matrix_name_worst <- paste0("resume_worst_",model,"_",dataset,".csv")
    matrix_name_best <- paste0("resume_best_",model,"_",dataset,".csv")

    write.csv(all_results_mean, matrix_name_mean, row.names = FALSE)
    write.csv(all_results_worst, matrix_name_worst, row.names = FALSE)
    write.csv(all_results_best, matrix_name_best, row.names = FALSE)

  }
}

# --------------------------------NUEVO 25/5/2025----------------------------------------------
mean_worst_best <- function(dataset_names, models, algorithms) {
  pattern <- "%s_fitness_%s_%s.csv"

  for (model in models) {
    for (dataset in dataset_names) {

      results_mean <- list()
      results_worst <- list()
      results_best <- list()

      for (algorithm in algorithms) {
        file_name <- sprintf(pattern, dataset, algorithm, model)
        lines <- readLines(file_name, warn = FALSE)

        # Filtramos líneas no vacías y limpiamos espacios
        lines <- trimws(lines)
        lines <- lines[nzchar(lines)]
        lines <- lines[lines != "\"\""]

        # Eliminamos cabecera si detectamos texto no numérico
        if (any(grepl("[A-Za-z]", lines[1]))) {
          lines <- lines[-1]
        }

        lines <- unique(lines)

        # Convertimos a matriz numérica
        matrix_fitness <- do.call(rbind, lapply(strsplit(lines, ","), as.numeric))

        # Métricas
        mean_vals <- colMeans(matrix_fitness) * 100
        worst_vals <- apply(matrix_fitness, 2, max) * 100
        best_vals <- apply(matrix_fitness, 2, min) * 100

        # Formateamos resultados
        make_result <- function(vals) {
          data.frame(
            Model = model,
            Dataset = dataset,
            Algorithm = algorithm,
            ACC = -vals[1],
            NFS = vals[2],
            MI = -vals[3],
            MAF = -vals[4]
          )
        }

        results_mean[[algorithm]] <- make_result(mean_vals)
        results_worst[[algorithm]] <- make_result(worst_vals)
        results_best[[algorithm]] <- make_result(best_vals)

        cat(sprintf("%s_%s_%s\n\n", dataset, algorithm, model))
      }

      # Guardamos por dataset-model
      save_results <- function(results_list, prefix) {
        file_name <- sprintf("resume_%s_%s_%s.csv", prefix, model, dataset)
        write.csv(do.call(rbind, results_list), file_name, row.names = FALSE)
      }

      save_results(results_mean, "mean")
      save_results(results_worst, "worst")
      save_results(results_best, "best")
    }
  }
}
# ------------------------------------------------------------------------------

dataset_names <- c("win10", "win7", "linux_process", "linux_disk", "linux_memory","network")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")

mean_worst_best(dataset_names, models, algorithms)



boxplot_fitness <- function(dataset_names, models, algorithms){
  pattern <- "%s_fitness_%s_maop_%s.csv"

  win10 <- 113
  win7 <- 103

  results_mean <- list()
  results_worst <- list()
  results_best <- list()

  for (model in models) {
    for (dataset in dataset_names) {
      for (algorithm in algorithms) {
        dataset_name <- sprintf(pattern, dataset, algorithm, model)
        content <- readLines(dataset_name)

        fitness_values <- character()

        for (line in content) {
          if (nchar(trimws(line)) > 0) {
            fitness_values <- c(fitness_values, line)
          }
        }

        fitness_values <- unique(fitness_values)
        matrix_fitness <- do.call(rbind, lapply(fitness_values, function(x) as.numeric(strsplit(x, ",")[[1]])))

        if (dataset=="win7") {
          matrix_fitness[,2] <- matrix_fitness[,2] * 103
        }else if (dataset=="win10") {
          matrix_fitness[,2] <- matrix_fitness[,2] * 113
        }
        matrix_fitness[,1] <- matrix_fitness[,1] * -100
        matrix_fitness[,3] <- matrix_fitness[,3] * -100
        matrix_fitness[,4] <- matrix_fitness[,4] * -100


      }
    }

    all_results_mean <- do.call(rbind, results_mean)
    all_results_worst <- do.call(rbind, results_worst)
    all_results_best <- do.call(rbind, results_best)

    matrix_name_mean <- paste0("resume_mean_",model,"_",dataset,".csv")
    matrix_name_worst <- paste0("resume_worst_",model,"_",dataset,".csv")
    matrix_name_best <- paste0("resume_best_",model,"_",dataset,".csv")

    write.csv(all_results_mean, matrix_name_mean, row.names = FALSE)
    write.csv(all_results_worst, matrix_name_worst, row.names = FALSE)
    write.csv(all_results_best, matrix_name_best, row.names = FALSE)

  }
}

# --------------------------------NUEVO 25/5/2025----------------------------------------------
boxplot_fitness <- function(dataset_names, models, algorithms) {
  pattern <- "%s_fitness_%s_%s.csv"
  win_multipliers <- list(win10 = 113, win7 = 103)

  for (model in models) {
    for (dataset in dataset_names) {

      results_mean <- list()
      results_best <- list()
      results_worst <- list()

      for (algorithm in algorithms) {
        filename <- sprintf(pattern, dataset, algorithm, model)
        lines <- readLines(filename, warn = FALSE)

        # Filtramos líneas no vacías y limpiamos espacios
        lines <- trimws(lines)
        lines <- lines[nzchar(lines)]
        lines <- lines[lines != "\"\""]

        # Eliminamos cabecera si detectamos texto no numérico
        if (any(grepl("[A-Za-z]", lines[1]))) {
          lines <- lines[-1]
        }

        lines <- unique(lines)

        # Convertimos a matriz numérica
        matrix_fitness <- do.call(rbind, lapply(strsplit(lines, ","), as.numeric))

        # Separar ejecuciones por saltos de línea vacíos originales
        # execution_blocks <- split(lines, cumsum(lines == ""))

        # Aplicar transformaciones
        mat[, 1] <- matrix_fitness[, 1] * -100          # Recall
        mat[, 2] <- matrix_fitness[, 2] * win_multipliers[[dataset]]  # NFS
        mat[, 3] <- matrix_fitness[, 3] * -100          # SU
        mat[, 4] <- matrix_fitness[, 4] * -100          # MacroF1

        # Calcular estadísticas por ejecución
        mean_vals <- colMeans(mat, na.rm = TRUE)
        best_vals <- apply(mat, 2, max, na.rm = TRUE)
        worst_vals <- apply(mat, 2, min, na.rm = TRUE)

        results_mean[[length(results_mean) + 1]] <- c(dataset, algorithm, model, mean_vals)
        results_best[[length(results_best) + 1]] <- c(dataset, algorithm, model, best_vals)
        results_worst[[length(results_worst) + 1]] <- c(dataset, algorithm, model, worst_vals)

      }

      # Convertir a data.frame
      col_names <- c("Dataset", "Algorithm", "Model", "Recall", "NFS", "SU", "MacroF1")
      df_mean <- as.data.frame(do.call(rbind, results_mean), stringsAsFactors = FALSE)
      df_best <- as.data.frame(do.call(rbind, results_best), stringsAsFactors = FALSE)
      df_worst <- as.data.frame(do.call(rbind, results_worst), stringsAsFactors = FALSE)

      # Convertir columnas numéricas
      df_mean[4:7] <- lapply(df_mean[4:7], as.numeric)
      df_best[4:7] <- lapply(df_best[4:7], as.numeric)
      df_worst[4:7] <- lapply(df_worst[4:7], as.numeric)

      # Guardar
      write.csv(df_mean, sprintf("resume_mean_%s_%s.csv", model, dataset), row.names = FALSE)
      write.csv(df_best, sprintf("resume_best_%s_%s.csv", model, dataset), row.names = FALSE)
      write.csv(df_worst, sprintf("resume_worst_%s_%s.csv", model, dataset), row.names = FALSE)
    }
  }
}

# ------------------------------------------------------------------------------

dataset_names <- c("win10", "win7")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")

boxplot_fitness(dataset_names, models, algorithms)
