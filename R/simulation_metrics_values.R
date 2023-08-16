#  Normaliza el fitness value y recalcula las metricas HV, IGD, GD y EPS de todos las simulaciones
# Los datos generados aquí se visualizan en las tablas II, III, IV

# Using for loop

# Define the reference points
ideal_point <- c(-1,0,-1,-1)
worst_point <- c(0,1,0,0)
nadir_point <- c(0.5,1.5,0.5,0.5)
reference_point <- nadir_point
reference_dirs <- reference_dirs_norm <- generate_reference_points(4,7)
reference_dirs <- sweep(reference_dirs,2,c(1,0.6,1,1))

# The pattern of the dataset names
pattern <- "%s_fitness_%s_maop_%s.csv"

# The combinations of dataset, algorithm, and model
dataset_names <- c("win10", "win7", "linux_process", "linux_disk", "linux_memory","network")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")


for (dataset in dataset_names) {
  for (algorithm in algorithms) {
    for (model in models) {
      dataset_name <- sprintf(pattern, dataset, algorithm, model)

      content <- readLines(dataset_name)

      # Initialize a list to store the sections and variable to keep track of the current section
      sections <- list()
      current_section <- character()

      # Loop through each line of the content
      for (line in content) {
        # If the line is not empty, add it to the current section
        if (nchar(trimws(line)) > 0) {
          current_section <- c(current_section, line)
        } else {
          # If the line is empty, save the current section and reset it
          if (length(current_section) > 0) {
            sections <- c(sections, list(current_section))
            current_section <- character()
          }
        }
      }

      # Save each section in different variables
      for (i in seq_along(sections)) {
        currect_section <- c()
        for (j in seq(length(sections[[i]]))) {
          currect_section <- rbind(currect_section,as.numeric(strsplit(sections[[i]], ",")[[j]]))
        }
        assign(paste0(dataset, algorithm, model,"_", i), currect_section)
      }

      # Normalize data
      for (i in seq_along(sections)) {
        assign(paste0(dataset, algorithm, model,"_", i), (get(paste0(dataset, algorithm, model,"_", i)) - min(ideal_point)) / (max(nadir_point) - min(ideal_point)))
      }

      hv_values <- c()
      igd_values <- c()
      gd_values <- c()
      eps_values <- c()

      for (i in seq_along(sections)) {
        hv_values <- rbind(hv_values, emoa::dominated_hypervolume(points = t(get(paste0(dataset, algorithm, model,"_", i))), ref = reference_point))
        igd_values <- rbind(igd_values, ecr::computeInvertedGenerationalDistance(t(get(paste0(dataset, algorithm, model,"_", i))), t(reference_dirs)))
        gd_values <- rbind(gd_values, ecr::computeGenerationalDistance(t(get(paste0(dataset, algorithm, model,"_", i))), t(reference_dirs)))
        if (is.matrix(get(paste0(dataset, algorithm, model,"_", i)))){
          eps_values <- rbind(eps_values, ecr::computeGenerationalDistance(t(get(paste0(dataset, algorithm, model,"_", i))), t(reference_dirs_norm)))
        }else{
          eps_values <- rbind(eps_values, ecr::computeGenerationalDistance(as.matrix(get(paste0(dataset, algorithm, model,"_", i))), t(reference_dirs_norm)))
        }
      }

      metrics <- cbind(hv_values, gd_values, igd_values, eps_values)

      f <- file(paste0(dataset,"_fitness_",algorithm,"_maop_",model,"_normalize.csv"), "w")
      h <- file(paste0(dataset,"_metrics_",algorithm,"_maop_",model,"_normalize.csv"), "w")

      writeLines(c("ACC,NFS,MI,MacroF1"), f)
      writeLines(c("HV,GD,IGD,EPS"), h)

      for (i in seq_along(sections)) {
        for (j in seq_len(nrow(get(paste0(dataset, algorithm, model,"_", i))))) {
          writeLines(as.character(get(paste0(dataset, algorithm, model,"_", i))[j,]), f, sep = ",")
          writeLines("\n", f)
        }
        writeLines("\n", f)
      }

      for (i in seq_len(nrow(metrics))) {
        writeLines(as.character(metrics[i,]), h, sep = ",")
        writeLines("\n", h)
      }

      writeLines("\n", f)
      writeLines("\n", h)

      close(f)
      close(h)

    }
  }
}





# Este codigo evalua directamente los valores de Mean, SD y Best para cada simulación
calculate_only_metrics <- function(dataset_names, algorithms, models, reference_dirs, reference_point){
  pattern <- "%s_fitness_%s_maop_%s.csv"

  for (dataset in dataset_names) {
    for (algorithm in algorithms) {
      for (model in models) {
        dataset_name <- sprintf(pattern, dataset, algorithm, model)

        content <- readLines(dataset_name)

        # Initialize a list to store the sections and variable to keep track of the current section
        sections <- list()
        current_section <- character()

        # Loop through each line of the content
        for (line in content) {
          # If the line is not empty, add it to the current section
          if (nchar(trimws(line)) > 0) {
            current_section <- c(current_section, line)
          } else {
            # If the line is empty, save the current section and reset it
            if (length(current_section) > 0) {
              sections <- c(sections, list(current_section))
              current_section <- character()
            }
          }
        }

        # Save each section in different variables
        for (i in seq_along(sections)) {
          currect_section <- c()
          for (j in seq(length(sections[[i]]))) {
            currect_section <- rbind(currect_section,as.numeric(strsplit(sections[[i]], ",")[[j]]))
          }
          assign(paste0(dataset, algorithm, model,"_", i), currect_section)
        }

        # Normalize data
        # for (i in seq_along(sections)) {
        #   assign(paste0(dataset, algorithm, model,"_", i), (get(paste0(dataset, algorithm, model,"_", i)) - min(ideal_point)) / (max(nadir_point) - min(ideal_point)))
        # }

        hv_values <- c()
        igd_plus_values <- c()
        gd_values <- c()
        for (i in seq_along(sections)) {
          gd_values <- rbind(gd_values, ecr::computeGenerationalDistance(t(get(paste0(dataset, algorithm, model,"_", i))), t(reference_dirs)))
          if (is.matrix(get(paste0(dataset, algorithm, model,"_", i)))){
            hv_values <- rbind(hv_values, emoa::dominated_hypervolume(points = t(get(paste0(dataset, algorithm, model,"_", i))), ref = reference_point))
            igd_plus_values <- rbind(igd_plus_values, eaf::igd_plus(data = get(paste0(dataset, algorithm, model,"_", i)), reference = reference_dirs))
          }else{
            hv_values <- rbind(hv_values, calculate_distance(data_point=get(paste0(dataset, algorithm, model,"_", i)), ref_point=reference_point))
            igd_plus_values <- rbind(igd_plus_values, eaf::igd_plus(data = t(get(paste0(dataset, algorithm, model,"_", i))), reference = reference_dirs))
          }

        }

        metrics <- cbind(hv_values, gd_values, igd_plus_values)
        metrics_best <- data.frame(HV=max(metrics[,1]),
                                         GD=min(metrics[,2]),
                                         IGD=min(metrics[,3]))

        rownames(metrics_best) <- c("Best")

        metrics_mean <- t(as.matrix(apply(metrics,2,mean)))

        metrics_mean <- as.data.frame(metrics_mean)
        colnames(metrics_mean) <- c("HV","GD","IGD")
        rownames(metrics_mean) <- c("Mean")

        metrics_sd <- t(as.matrix(apply(metrics,2,sd)))
        colnames(metrics_sd) <- c("HV","GD","IGD")
        rownames(metrics_sd) <- c("SD")

        results <- rbind(metrics_best,metrics_mean,metrics_sd)

        metrics_name <- paste0(dataset,"_metrics_",algorithm,"_maop_",model,".csv")

        write.csv(results, metrics_name, row.names = FALSE)


      }
    }
  }

}


reference_point <- c(0.5,1.5,0.5,0.5)
reference_dirs <- reference_dirs_norm <- generate_reference_points(4,7)
reference_dirs <- sweep(reference_dirs,2,c(2,1,2,2))

dataset_names <- c("win10", "win7", "linux_process", "linux_disk", "linux_memory","network")
algorithms <- c("rvea", "moead","nsgaiii", "nsgaii")
models <- c("knn","gnb")

calculate_only_metrics(dataset_names, algorithms, models, reference_dirs, reference_point)
