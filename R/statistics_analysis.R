mean_worst_best <- function(dataset_names, models, algorithms){
  pattern <- "%s_fitness_%s_maop_%s.csv"

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

dataset_names <- c("win10", "win7")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")

boxplot_fitness(dataset_names, models, algorithms)