# Preprocessing for Function Space Optimization
# Moritz Feigl, 2019 - adjusted for wflow_sbm by Ruben Imhoff, 2022
#

rm(list = ls())

setwd("/gpfs/home4/aweerts")
library(feather)
library(tidyverse)
library(keras)
library(tensorflow)
Sys.setenv(RETICULATE_PYTHON = "/gpfs/home4/aweerts/.conda/envs/fso/bin/python")
source("Scripts/functions/CFG_functions.R")
use_condaenv("fso")
Sys.setenv(NUMEXPR_MAX_THREADS = 192)

# 1. Create Grammar --------------------------------------------------------------------
g <- grammar(tf = "numeric * <eq> + numeric, <eq> + numeric, <eq> ",
             eq = "<fs>, <eq><op><fs>, <eq><op>numeric, <fs><op>numeric",
             fs = "<sp>,  <f>(<sp>), <sp><op><sp>, numeric",
             f = "exp, log",
             op = "+, -, *, /",
             sp = "silt, sand, clay, oc, bd, ph, bl") # See also lines 117-121
#TODO: change this
spatial_predictor_variables <- c("silt", "sand", "clay", "oc", "bd", "ph", "bl")

# 2. Random sample grammar & simplify --------------------------------------------------
#TODO: change this
# function_df2 <- par.grammar.sampler(n = 5000,
#                                     cfgram = g,
#                                     max.depth = 3,
#                                     save_feather = FALSE,
#                                     parallel = TRUE,
#                                     no_cores = 192)

# TODO: uncomment this when done
function_df2 <- par.grammar.sampler(n = 5000000,
                                    cfgram = g,
                                    max.depth = 3,
                                    save_feather = FALSE,
                                    parallel = TRUE,
                                    no_cores = 192)

# take only unique functions
function_df2 <- function_df2[!duplicated(function_df2$Transfer_Function), ]
# save function_list as feather
write_feather(function_df2, "Data/functions_V2.feather")
function_df2$Transfer_Function_simple <- parlapply_simplify(funs = function_df2$Transfer_Function,
                                                            function_variables = spatial_predictor_variables,
                                                            no_cores = 192)
# remove NA functions
function_df2 <- function_df2[!is.na(function_df2$Transfer_Function_simple), ]
write_feather(function_df2, "Data/functions_simple_onlyfunctions.feather")

# 3. Put in numeric values from [-1.5, 1.5] ----------------------------------------------
numbers <- c(-1*seq(0.1, 1.5, 0.1), seq(0.1, 1.5, 0.1))
# Functions for randomly putting in numeric values from given range
num_input <- function(fun){
  while(length(grep("numeric", fun)) != 0){
    fun <- sub("numeric", sample(numbers, 1), fun)
  }
  return(fun)
}
nums_input <- function(fun){
  fun <- gsub(" ", "", fun)
  funs <- replicate(10, num_input(fun))
  funs <- gsub("+-", "-", funs, fixed = TRUE)
  funs <- gsub("--", "+", funs, fixed = TRUE)
  funs <- gsub("++", "+", funs, fixed = TRUE)
  funs <- gsub("-+", "-", funs, fixed = TRUE)
  return(funs)
}
num_df_input <- function(functions, numbers){
  number_list <- lapply(functions, nums_input)
  return(do.call(c, number_list))
}
# Create new function vector with 10 numeric inputs each
functions_simple_10_numerics <- num_df_input(function_df2$Transfer_Function_simple, numbers)
# take only unique functions
functions_simple_10_numerics_unique <- functions_simple_10_numerics[
  !duplicated(functions_simple_10_numerics)
]

# remove exotic functions ----------------------------------------------------------------
# * 6
all_id6 <- grep("6*", functions_simple_10_numerics_unique, fixed = TRUE)
id6 <- all_id6[!(all_id6 %in% grep(".6*", functions_simple_10_numerics_unique, fixed = TRUE))]
if (length(id6) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id6]
}
# * 5
all_id5 <- grep("5*", functions_simple_10_numerics_unique, fixed = TRUE)
id5 <- all_id5[!(all_id5 %in% grep(".5*", functions_simple_10_numerics_unique, fixed = TRUE))]
if (length(id5) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id5]
}
# ^5
idp5 <- grep("^5", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(idp5) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-idp5]
}
# ^4
idp4 <- grep("^4", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(idp4) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-idp4]
}
# ^3
idp3 <- grep("^3", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(idp3) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-idp3]
}
# * 3
all_id3 <- grep("3*", functions_simple_10_numerics_unique, fixed = TRUE)
id3 <- all_id3[!(all_id3 %in% grep(".3*", functions_simple_10_numerics_unique, fixed = TRUE))]
if (length(id3) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id3]
}
# * 4
all_id4 <- grep("4/*", functions_simple_10_numerics_unique, fixed = FALSE)
id4 <- all_id4[!(all_id4 %in% grep(".4", functions_simple_10_numerics_unique, fixed = TRUE))]
if (length(id4) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id4]
}
id42 <- grep("(4", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(id42) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id42]
}
id43 <- grep("-4", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(id43) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id43]
}
id44 <- grep("/4", functions_simple_10_numerics_unique, fixed = TRUE)
if (length(id44) > 0){
functions_simple_10_numerics_unique <- functions_simple_10_numerics_unique[-id43]
}

# save
write_feather(data.frame("TF_simple_10numerics" = functions_simple_10_numerics_unique, 
                         stringsAsFactors = FALSE),
              "Data/functions_simple_10_numerics_pp.feather")

# 4. Prepare text data as dictionary integers -------------------------------------------
funs <- functions_simple_10_numerics_unique

# Define dictionary
dict_from_list <- function(...){
  dict <- c(...)
  dictionary <- as.data.frame(matrix(NA, nrow = 1, ncol = length(dict)))
  names(dictionary) <- dict
  dictionary[1, ] <- 1:length(dict)
  return(dictionary)
}

dict_from_list_index2words <- function(...){
  dict <- c(...)
  dictionary <- as.data.frame(matrix(NA, nrow = 1, ncol = length(dict)))
  names(dictionary) <- 1:length(dict)
  dictionary[1, ] <- dict
  return(dictionary)
}

#TODO: change this
operators <- c('-', '(', ')', '+', "*", "/", "1")
f <- c("log", "exp")
vars <- c("silt", "sand", "clay", "oc", "bd", "ph", "bl")
nums <- c("^2", "0.8", "0.1", "0.2", "0.7", "1.4", "1.5", "1.3", "1.1", 
          "1.2", "0.9", "0.3", "0.5", "0.6", "0.4", "2", "3")
dict <- c(operators, f, vars, nums)
dictionary <- dict_from_list(operators, f, vars, nums)
dictionary_index2words <- dict_from_list_index2words(operators, f, vars, nums)
write_feather(dictionary_index2words, "Data/index2word_10numerics.feather")

# Keras tokenizing
prepare_text_for_tokenize <- function(text){
  # Replace "^2" with a placeholder
  text <- gsub("\\^2", "\\$", text)
  
  # Add spaces around operators
  operators = c("\\+", "\\-", "\\/", "\\*", "\\(", "\\), \\$")
  operator_regex <- paste(operators, collapse = "|")
  text <- stringr::str_replace_all(text, paste0('(', operator_regex, ')'), " \\1 ")
  
  text <- tokenizers::tokenize_words(text, strip_punct = FALSE)
  text <- sapply(text, 
                 FUN = function(x) paste0(c("", paste0(x, collapse = ";"), ""), collapse = ";"))

  # Replace the placeholder with "^2"
  text <- gsub("\\$", "\\^2", text)
  return(text)
}

#TODO: change this
# cl <- parallel::makeCluster(192)
prep_funs <- parallel::mclapply(X = funs,
                                FUN = prepare_text_for_tokenize,
                                       mc.cores = 192,
                                       mc.set.seed = TRUE)
# prep_funs <- parallel::parLapply(cl = cl, X = funs, prepare_text_for_tokenize)

# keras tokenizer with pre-defined dictionary
tt <- keras::text_tokenizer(filters = "\\;")
tt$word_index <- dictionary

funs_tokenized <- keras::texts_to_sequences(tt, prep_funs)
max_length <- max(lengths(funs_tokenized))
funs_padded <- keras::pad_sequences(funs_tokenized, maxlen = max_length)
feather::write_feather(data.frame(funs_padded), "Data/generator_data_simple_10numerics.feather")



