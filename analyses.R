require(MplusAutomation)
require(rlist)
require(stats)
require(data.table)
require(openxlsx)
require(jsonlite)
require(hdf5)


## Dictionaries
data_files <- jsonlite::read_json(path = "Dictionaries/dataset_files.json")
models <- jsonlite::read_json(path = "Dictionaries/models.json")
models_all <- jsonlite::read_json(path = "Dictionaries/models_all.json")
fac_series <- jsonlite::read_json(path = "Dictionaries/model_factors.json")
fac_series <- sapply(fac_series, unlist)
indicators <- jsonlite::read_json(path = "Dictionaries/indicators.json")
indicators_all <- jsonlite::read_json(path = "Dictionaries/indicators_all.json")
DV_subsets <- jsonlite::read_json(path = "Dictionaries/DV_subsets.json")
DV_subsets <- sapply(DV_subsets, unlist)
rename <- jsonlite::read_json(path = "Dictionaries/rename.json")
abbreviations <- jsonlite::read_json(path = "Dictionaries/abbreviations.json")
model_names <- names(fac_series) # factor models names

## Output File Names
cfas_list <- structure(vector(mode='list', length=6), 
                       names=c('descriptives_factors_r_within', 'descriptives_factors_r_between', # factor correlations
                               'results_correlations', 'results_regressions', 'results_r2')) # results 
cfas <- rlist::list.append(cfas_list, mplus_out_files = structure(paste0(model_names, '_CFA'), names = model_names), model_names = model_names, fac_series = fac_series)
cfas_PTSD <- rlist::list.append(cfas_list, mplus_out_files = structure(paste0(model_names, '_CFA_PTSD'), names = model_names), model_names = model_names, fac_series = fac_series)
bfs <- rlist::list.append(cfas_list, mplus_out_files = structure(paste0(model_names, '_Bifactor'), names = model_names), model_names = model_names, fac_series = fac_series)

## MPlus Output
cfas$mplus_output <- structure(sapply(paste0('MPlus/', cfas$mplus_out_files, '.out'), MplusAutomation::readModels, simplify = FALSE), 
                               names = names(cfas$mplus_out_files))
cfas_PTSD$mplus_output <- structure(sapply(paste0('MPlus/', cfas_PTSD$mplus_out_files, '.out'), MplusAutomation::readModels, simplify = FALSE), 
                                    names = names(cfas_PTSD$mplus_out_files))
bfs$mplus_output <- structure(sapply(paste0('MPlus/', bfs$mplus_out_files, '.out'), MplusAutomation::readModels, simplify = FALSE), 
                              names = names(bfs$mplus_out_files))

## Bayes Column Names (if needed)
for (o in 1:length(list(cfas, cfas_PTSD, bfs))) {
  mod <- list(cfas, cfas_PTSD, bfs)[[o]]
  for (m in 1:length(mod$mplus_output)) {
    if (toupper(mod$mplus_output[[m]]$input$analysis$estimator) == 'BAYES') {
      whiters <- sapply(mod$mplus_output[[m]][["input"]][["savedata"]], function(x) grepl('fscores', x))
      iters <- as.numeric(gsub('*SAVE.*fscores.*[(](.*) .*;', '\\1', mod$mplus_output[[m]][["input"]][["savedata"]][which(whiters)]))
      cols <- sapply(mod$mplus_output[[m]]$savedata_info$fileVarNames, 
                     function(x) if (grepl('[+]', x)) paste0(rep(gsub('[+]', '', paste0(x, '_')), iters), 1:iters) else gsub(' Mean', '', x))
      if (o == 1) {
        colnames(cfas$savedata[[m]]) <- unname(unlist(cols)) 
      } else if (o == 2) {
        colnames(cfas_PTSD$mplus_output[[m]]$savedata) <- unname(unlist(cols)) 
      } else {
        colnames(bfs$mplus_output[[m]]$savedata) <- unname(unlist(cols))
      }
    }
  }
}

## Data
data <- read.csv(data_files$nesarc3_cleaned)
for (a in 1:length(list(cfas, cfas_PTSD, bfs))) {
  mod <- list(cfas, cfas_PTSD, bfs)[[a]]
  df <- structure(vector(mode='list', length=length(mod$model_names)), names=mod$model_names)
  for (m in mod$model_names) {
    df[[m]] <- mod$mplus_output[[m]]$savedata
    vars_origin  <- unname(c(mod$fac_series[[m]], unlist(DV_subsets)))
    vars <- sapply(vars_origin, function(y) gsub(', ', '', toString(toupper(strsplit(y, '')[[1]][1:min(8, length(strsplit(y, '')[[1]]))]))[[1]]))
    for (f in names(vars)) {
      colnames(df[[m]]) <-  gsub(paste0('^', vars[f], '$'), f, colnames(df[[m]]))
    }
  }
  if (a == 1) cfas$data <- df else if (a == 2) cfas_PTSD$data <- df else bfs$data <- df
}


analyses_tables <- function(DVs, mplus_file_stems, models, factors, mplus_output, DV_data = NULL, family = 'gaussian', coef_link = 'identity', index = 'MPLUS_IN') { 
  # models: vector of strings with model names (must be names of fac_series list)
  # factors: named list of vectors with strings that are names of each model's factors (must be column names in output)
  # mplus_output: list of output objects returned from MplusAutomation::readModels(), named after models
  # DVs: vector with strings identifying names of outcome variables (must be column names in data)
  # DV_data: dataframe with outcome variables
  # family: string identifying which distribution family to use in stats::glm regression analyses
  correlations <- structure(vector(mode='list', length=length(models)), names=models) # empty list for correlations
  regressions <- structure(vector(mode='list', length=length(models)), names=models) # empty list for regressions
  regressions_p <- structure(vector(mode='list', length=length(models)), names=models) # empty list for regressions (with p-value stars)
  results_regressions_objects <- structure(vector(mode='list', length=length(models)), names=models) # empty list for regressions
  r_squared <- data.frame(structure(matrix(nrow=length(DVs), ncol=length(models)), dimnames=list(DVs, models))) # r-squared empty df 
  for (m in models) { # iterate models 
    savedata <- data.frame(sapply(mplus_output[[m]]$savedata, as.numeric))
    # if (min(savedata$MPLUS_IN, na.rm = TRUE) == 0) index <- savedata$MPLUS_IN + 1 else index <- savedata$MPLUS_IN 
    if (is.null(DV_data) == FALSE) dm <- plyr::join(DV_data, savedata, by = index) else dm <- savedata # factor scores & outcome data together
    fac <- factors[[m]]
    dm[, sapply(fac, toupper)] <- scale(dm[, sapply(fac, toupper)]) # standardize & center factor scores
    # DVs <- sapply(DVs, function(y) { # remove invariable DVs
    #   yy <- toupper(paste0(strsplit(y, '')[[1]][1:min(length(strsplit(y, '')[[1]]), 8)], collapse=''))
    #   if (var(dm[is.na(dm[, yy]) == FALSE, yy]) != 0) y
    # })
    corrs <- data.frame(matrix(nrow=length(fac), ncol=length(DVs))) # empty df for correlations
    corrs <- structure(data.frame(matrix(nrow=length(fac), ncol=length(DVs), dimnames=list(fac, DVs))), names=DVs) # empty df for correlations
    coef_num <- ifelse(family != 'gaussian', length(fac), length(fac)+1)
    regs <- structure(data.frame(matrix(nrow=coef_num, ncol=length(DVs))), names=DVs) # empty df for regressions
    for (y in DVs) { # iterate factors 
      yy <- toupper(paste0(strsplit(y, '')[[1]][1:min(length(strsplit(y, '')[[1]]), 8)], collapse=''))
      if (yy %in% colnames(dm) == FALSE) next
      #### Regression ####
      if (tolower(family) != 'gaussian') { # if non-normal outcome 
        model <- stats::glm(paste0(yy, '~0+', paste0(sapply(fac, toupper), collapse = '+')), dm, family=family) # GLM results 
        model_null <- stats::glm(paste0(yy, '~0'), dm, family=family) # null model results
        result <- summary(model) # summary of model 
        r_squared[y, m] <- 1-logLik(model)/logLik(model_null) # McFadden's pseudo-R^2 
      } else { # if Gaussian outcome 
        result <- summary(stats::lm(formula=paste0(toupper(yy), "~", paste0(sapply(fac, toupper), collapse = "+")), data=dm))
        r_squared[y, m] <- result$r.squared # r-squared for DV & model
      }
      results_regressions_objects[[m]][[y]] <- result
      coefs <- coefficients(result)[, 'Estimate'] # coefficients
      coefs <- stats::make.link(coef_link)$linkinv(coefs)
      pVals <- coefficients(result)[, grepl('Pr[(]', colnames(coefficients(result)))] # p-values
      stars <- ifelse(pVals < 0.001, '***', ifelse(pVals < 0.01, '**', ifelse(pVals < 0.05, '*', ''))) # significance stars 
      regs[, y] <- paste0(sapply(coefs, function(c) format(round(c, 2), nsmall=2)), stars) # coefficients & significance stars in one column (for DV) 
      rownames(regs) <- rownames(coefficients(result))
      #### Correlations ####
      for (f in fac) { # iterate outcomes  
        corrs[f, y] <- format(round(data.frame(stats::cor(dm[, toupper(f)], dm[, toupper(yy)], use='na.or.complete')), 3), nsmall=3) # factor-DV correlation
      }
    }
    correlations[[m]] <- cbind(data.frame(Model = m, # model heading column
                                          Factor = rownames(corrs)), corrs) # model & factor columns + correlations  
    regressions[[m]] <- data.frame(Model = m, # model heading column
                                   Factor = rownames(coefficients(result)), sapply(regs, function(c) as.numeric(gsub('[*]', '', c)))) # without p-stars
    regressions_p[[m]] <- data.frame(Model = m, # model heading column
                                     Factor = rownames(coefficients(result)), regs) # model & factor columns + regressions  
  }
  correlations <- data.frame(data.table::rbindlist(correlations)) # bind the model correlations list together into 1 table
  regressions <- data.frame(data.table::rbindlist(regressions)) # bind the model correlations list together into 1 table
  regressions_p <- data.frame(data.table::rbindlist(regressions_p)) # bind the model correlations list together into 1 table
  for (x in unlist(factors)) correlations$Factor <- gsub(toupper(x), x, correlations$Factor)
  for (x in unlist(factors)) regressions$Factor <- gsub(toupper(x), x, regressions$Factor)
  for (x in unlist(factors)) regressions_p$Factor <- gsub(toupper(x), x, regressions_p$Factor)
  return(invisible(list(results_correlations=correlations,  results_regressions_p=regressions_p, results_regressions = regressions, results_r2 = r_squared,
                        results_regressions_objects = results_regressions_objects))) 
} 

# Analyses

## Factor Inter-Correlations
for (l in c('cfas', 'cfas_PTSD', 'bfs')) {
  o <- eval(parse(text = l)) # load object named after string l
  fs <- sapply(o$model_names, function(m) o$data[[m]][, o$fac_series[[m]]]) # factor scores for each model
  fb <- sapply(o$model_names, function(m) sapply(o$fac_series[[1]], function(f) stats::cor(o$data[[1]][, f], fs[[m]])))
  fw <- sapply(fs, function(f) stats::cor(f))
  b <- data.table::rbindlist(lapply(names(fb), function(m) data.frame(Model = m, Factor = o$fac_series[[m]], sapply(data.frame(format(round(fb[[m]], 3), nsmall = 3)), as.numeric))))
  w <- data.table::rbindlist(lapply(names(fw), 
                                    function(m) data.frame(Model = m, Factor = o$fac_series[[m]], sapply(data.frame(format(round(fw[[m]], 3), nsmall = 3)), as.numeric))), fill = T)
  eval(parse(text = paste0(l, '[[\'descriptives_factors\']][[\'correlations_within\']] <- w')))
  eval(parse(text = paste0(l, '[[\'descriptives_factors\']][[\'correlations_between\']] <- b')))
}

## Correlations & Regressions 
for (i in 1:length(DV_subsets)) { # iterate through subsets of DVs 
  link <- c('identity', 'log', 'identity')[i] # coefficient transform link (so exp() on original returned coefficients for i == 2)
  tables_cfas <- analyses_tables(DV_subsets[[i]], models = cfas$model_names, factors = cfas$fac_series, family = c('gaussian', 'binomial', 'gaussian')[i], 
                                 mplus_output = cfas$mplus_output, coef_link = link) # tables for CFAs
  vars <- DV_subsets[[i]][which(DV_subsets[[i]] != 'PTSD')]
  tables_cfas_PTSD <- analyses_tables(vars, models = cfas_PTSD$model_names, factors = cfas_PTSD$fac_series, family = c('gaussian', 'binomial', 'gaussian')[i],
                                      mplus_output = cfas_PTSD$mplus_output, coef_link = link) # PTSD subgroup
  tables_bfs <- analyses_tables(DV_subsets[[i]], models = bfs$model_names, factors = bfs$fac_series, family = c('gaussian', 'binomial', 'gaussian')[i],
                                mplus_output = bfs$mplus_output, coef_link = link) # bifactor tables 
  for (t in names(tables_cfas)) {
    cfas[[t]][[i]] <- tables_cfas[[t]] # iterate through output types
    cfas_PTSD[[t]][[i]] <- tables_cfas_PTSD[[t]] # iterate through output types
    bfs[[t]][[i]] <- tables_bfs[[t]] # iterate through output types
  }
}

## Name After DV Subsets
for (t in names(tables_cfas)) {
  names(cfas[[t]]) <- names(DV_subsets)
  names(cfas_PTSD[[t]]) <- names(DV_subsets)
  names(bfs[[t]]) <- names(DV_subsets)
}


## Make Directory (if needed) 
if ('R' %in% list.files('Results') == FALSE) system('mkdir Results/R') 

# ## Write Data
# openxlsx::write.xlsx(sapply(cfas[["mplus_output"]], function(x) x[['savedata']]), 'MPlus/data_cfas.xlsx', sheets = names(cfas[["mplus_output"]]))
# openxlsx::write.xlsx(sapply(cfas_PTSD[["mplus_output"]], function(x) x[['savedata']]), 'MPlus/data_cfas_PTSD.xlsx', sheets = names(cfas_PTSD[["mplus_output"]]))
# openxlsx::write.xlsx(sapply(bfs[["mplus_output"]], function(x) x[['savedata']]), 'MPlus/data_bfs.xlsx', sheets = names(bfs[["mplus_output"]]))

## Correlation 
openxlsx::write.xlsx(cfas$results_correlations, 'Results/R/CFA_results_correlations.xlsx')
openxlsx::write.xlsx(cfas_PTSD$results_correlations, 'Results/R/CFA_PTSD_results_correlations.xlsx', border = 'columns', fontName = 'Times', colWidths = 'auto')
openxlsx::write.xlsx(bfs$results_correlations, 'Results/R/Bifactor_results_correlations.xlsx')

## Regression 
openxlsx::write.xlsx(cfas$results_regressions_p, 'Results/R/CFA_results_regressions_p.xlsx')
openxlsx::write.xlsx(cfas_PTSD$results_regressions_p, 'Results/R/CFA_PTSD_results_regressions_p.xlsx')
openxlsx::write.xlsx(bfs$results_regressions_p, 'Results/R/Bifactor_results_regressions_p_Bifactor.xlsx')
openxlsx::write.xlsx(cfas$results_regressions, 'Results/R/CFA_results_regressions.xlsx')
openxlsx::write.xlsx(cfas_PTSD$results_regressions, 'Results/R/CFA_PTSD_results_regressions.xlsx')
openxlsx::write.xlsx(bfs$results_regressions, 'Results/R/Bifactor_results_regressions.xlsx')


