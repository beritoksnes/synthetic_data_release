rm(list=ls())

#####################################
## Run CVineSensitive estimation   ##
## To be called from Python script ##
#####################################

suppressWarnings(
  suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(jsonlite)
  })
)

# Functions ------------------------------------- ####
prep_data <- function(dt, var_types) {
  # Initialize data tables
  u_cont <- u_cat <- u_cat_1 <- NULL
  # Separate continuous and discrete columns
  col_names <- colnames(dt)
  cont_cols <- col_names[which(var_types=="c")]
  cat_cols  <- col_names[which(var_types=="d")]
  # Transform data to data.table
  dt <- data.table::as.data.table(dt)
  # Handle numerical columns 
  if (any(var_types == "c")) {
    u_cont <- dt[, lapply(.SD, function(x) ecdf(x)(x)*.N/(.N+1)), .SDcols=cont_cols]
  }
  # Handle categorical columns
  if (any(var_types == "d")) {
    u_cat <- dt[, lapply(.SD, function(x) ecdf(x)(x)*.N/(.N+1)), .SDcols=cat_cols]
    u_cat_1 <- dt[, lapply(.SD, function(x) ecdf(x)(x-1)*.N/(.N+1)), .SDcols=cat_cols]
    colnames(u_cat_1) <- paste0(colnames(u_cat), "_1")
  }
  # Return transformed data
  pobs <- cbind(u_cont, u_cat, u_cat_1)
  return(pobs)
}

.bicopest_sensitive <- function(data,
                                family,
                                rotation,
                                var_types=c("c","c"),
                                lmbda=0,
                                lower,
                                upper) {
  # Define objective function
  loglik <- function(par) {
    sum(log(rvinecopulib::dbicop(u=data,
                                 family=family,
                                 rotation=rotation,
                                 parameters=par,
                                 var_types=var_types)))
  }
  scaled_loglik <- function(par) {
    n <- dim(data)[1]
    loglik(par)/n
  }
  penalty <- function(par) {
    ktau <- rvinecopulib::par_to_ktau(family, rotation, par)
    lmbda * (ktau)^2/(1-(ktau)^2)
  }
  objective <- function(par) {
    -scaled_loglik(par) + penalty(par)
  }

  # Find initial value (utilize 1-1 relationship with par and ktau)
  tau <- cor(data[,1], data[,2], method="kendall")
  par_start <- rvinecopulib::ktau_to_par(family, tau)

  # Run estimation algorithm
  res <- optim(par=par_start,
               fn=objective,
               method="L-BFGS-B",
               lower=lower,
               upper=upper)
  par <- res$par

  # Return model estimate with log-likelihood value
  return(list(par=as.numeric(par),
              family=family,
              rotation=rotation,
              loglik=loglik(par)))
}


bicop_sensitive <- function(data, var_types=c("c", "c"), lmbda=0) {
  # Initialize familyset table
  family <- c("gaussian",
              "frank",
              "clayton", "clayton", "clayton", "clayton",
              "gumbel", "gumbel", "gumbel", "gumbel",
              "joe", "joe", "joe", "joe")
  
  rotation <- c(0,
                0, 
                0, 90, 180, 270,
                0, 90, 180, 270,
                0, 90, 180, 270)
  
  par_min <- c(-1+1e-10, # add small number to avoid boundary issues
               -35,
               1e-10, 1e-10, 1e-10, 1e-10,
               1, 1, 1, 1,
               1, 1, 1, 1)
  
  par_max <- c(1-1e-10, # subtract small number to avoid boundary issues
               35,
               28, 28, 28, 28,
               50, 50, 50, 50,
               30, 30, 30, 30)
  
  familyset_table <- data.table(family, rotation, par_min, par_max)
  
  familyset_table[, `:=` (theta_hat=0, tau_hat=0, aic=0)]
  
  # Estimate all families and compute log-likelihood value
  for (i in 1:nrow(familyset_table)) {
    family <- familyset_table[i, family]
    rotation <- familyset_table[i, rotation]
    lower <- familyset_table[i, par_min]
    upper <- familyset_table[i, par_max]
    res <- .bicopest_sensitive(data,
                              family,
                              rotation,
                              var_types=var_types,
                              lmbda=lmbda,
                              lower=lower,
                              upper=upper)
    familyset_table[i, `:=` (
      theta_hat=res$par,
      tau_hat=rvinecopulib::par_to_ktau(family, rotation, res$par),
      loglik=res$loglik,
      scaled_loglik=res$loglik/nrow(data),
      aic=-2*res$loglik+2*1
      )]
  }
  
  # Select the best family as the one with the lowest aic
  best_idx <- familyset_table[,which.min(aic)]
  if (familyset_table[best_idx, aic] > 0) {
    fit <- rvinecopulib::bicop_dist(family="indep", var_types=var_types)
  } else {
    fit <- rvinecopulib::bicop_dist(
      family=familyset_table[best_idx, family],
      rotation=familyset_table[best_idx, rotation],
      parameters=familyset_table[best_idx, theta_hat],
      var_types=var_types)
  }
  return(fit)
}

cvinecop_sensitive <- function(data, 
                               var_types, 
                               family_set="all", 
                               sensitive=NULL, 
                               lmbda=0) {
  # Prepare data
  n <- dim(data)[1]
  # Check if response is discrete
  discrete_response <- tail(var_types,1)=="d"
  if (discrete_response) {
    d <- ncol(data) - 1 # Account for the extra column required for discrete data
  } else {
    d <- ncol(data)
  }
  dt <- copy(data)
  dt <- as.data.table(dt)
  var_names <- colnames(dt)
  
  # Create matrices for C-vine estimation
  # R-vine matrix
  M <- matrix(d:1,d,d) 
  M[lower.tri(M)] <- 0
  M <- M[1:d,d:1]
  
  # Array for storing pseudo-observations (only one needed for C-vines!)
  V <- array(NA, dim=c(d,d,n))
  for (i in 1:d) {
    V[1,i,] <- data[[i]]
  }
  
  # Log-likelihood matrix
  loglik <- matrix(0, d, d)
  
  # Estimation
  pair_copulas <- vector("list", d-1)
  # Estimation of first tree
  k <- 1
  pcs <- vector("list", d-k)
  for (j in 1:(d-k)) {
    # Extract columns
    ind1 <- j
    ind2 <- d
    u1 <- data[[ind1]]
    u2 <- data[[ind2]]
    if (discrete_response) {
      u2_1 <- data[[d+1]]
      pc_data <- cbind(u1,u2,u2_1)
    } else {
      pc_data <- cbind(u1,u2)
    }
    # Select and estimate pair copula
    if (any(colnames(data)[c(ind1, ind2)] %in% sensitive)) {
      fit <- bicop_sensitive(data=pc_data, 
                             var_types=var_types[c(ind1,ind2)],
                             lmbda=lmbda)
    } else {
      fit <- rvinecopulib::bicop(data=pc_data, 
                                 var_types=var_types[c(ind1,ind2)],
                                 family_set=family_set)
    }
    pcs[[j]] <- fit
    loglik[k,j] <- sum(log(predict(fit, pc_data, what="pdf")))
    # Save pseudo-observations
    pobs <- rvinecopulib::hbicop(u=pc_data, 2, fit)
    V[k+1,j,] <- pobs
  }
  pair_copulas[[k]] <- pcs
  
  # Estimation of later trees
  for (k in 2:(d-1)) {
    pcs <- vector("list", d-k)
    for (j in 1:(d-k)) {
      # Check if any of the variables are sensitive
      ind1 <- M[d-j+1,j]
      ind2 <- M[k,j]
      # Extract columns
      u1 <- V[k,j,]
      u2 <- V[k,d-k+1,]
      pc_data <- cbind(u1,u2)
      # Select and estimate pair copula
      if (any(colnames(data)[c(ind1,ind2)] %in% sensitive)) {
        fit <- bicop_sensitive(data=pc_data, 
                               var_types=var_types[c(ind1,ind2)],
                               lmbda=lmbda)
      } else {
        fit <- rvinecopulib::bicop(data=pc_data, 
                                   var_types=var_types[c(ind1,ind2)],
                                   family_set=family_set)
      }
      pcs[[j]] <- fit
      loglik[k,j] <- sum(log(predict(fit, pc_data, what="pdf")))
      # Save pseudo-observations
      pobs <- rvinecopulib::hbicop(u=pc_data, 2, fit)
      V[k+1,j,] <- pobs
    }
    pair_copulas[[k]] <- pcs
  }
  # Return model object
  vinecop <- rvinecopulib::vinecop_dist(pair_copulas=pair_copulas, 
                                        structure=M, 
                                        var_types=var_types)
  return(vinecop)
}

vinecop_to_json <- function(vc) {
  d <- vc$structure$d
  vc_lst <- list("pair copulas"=list(), "structure"=list(), "var_types"=list())
  
  # Add pair copula information
  for (t in 1:(d-1)) {
    pcs <- list()
    for (e in 1:(d-t)) {
      pc <- vc$pair_copulas[[t]][[e]]
        pcs[[paste0("pc", e-1)]] <- list(
          "fam"=unlist(family_name[pc$family]),
          "npars"=pc$npars,
          "par"=list(data=as.list(pc$parameters),
                     shape=dim(pc$parameters)),
          "rot"=pc$rotation,
          "vt"=pc$var_types
        )
    }
    vc_lst[["pair copulas"]][[paste0("tree", t-1)]] <- pcs
  }
  
  # Add structure information
  structure <- vc$structure
  vc_lst$structure <- list(
    "array"=list("d"=d,
                 "data"=lapply(structure$struct_array, as.list),
                 "t"=structure$trunc_lvl),
    "order"=structure$order)
  
  # Add variable types
  vc_lst$var_types <- vc$var_types
  
  
  jsonlite::toJSON(vc_lst, pretty=FALSE, auto_unbox=TRUE, digits=NA)
}

family_name <- list("indep" = "Independence",
                    "gaussian" = "Gaussian",
                    "t" = "Student",
                    "clayton" = "Clayton",
                    "gumbel" = "Gumbel",
                    "frank" = "Frank",
                    "joe" = "Joe",
                    "bb1" = "BB1",
                    "bb6" = "BB6",
                    "bb7" = "BB7",
                    "bb8" = "BB8",
                    "tawn" = "Tawn", 
                    "tll" = "TLL")

# Estimation ------------------------------------ ####

# Read command line arguments
args <- commandArgs(trailingOnly=TRUE)

data <- read.csv(args[1]) %>% as.data.table()

params <- fromJSON(args[2])
var_types <- params$var_types
family_set <- params$family_set
sensitive <- params$sensitive
lmbda <- params$lmbda

model_path <- args[3]

# Estimate C-vine with penalty on sensitive dependencies
u <- prep_data(data, var_types)

fit <- cvinecop_sensitive(data = u, 
                          var_type = var_types,
                          family_set = family_set,
                          sensitive = sensitive,
                          lmbda = lmbda)

json_obj <- vinecop_to_json(fit)
write(json_obj, file=model_path)