#' @title Fit the binary Gaussian mixture model (GMM) on target data set by leveraging multiple source data sets under a transfer learning (TL) setting.
#'
#' @description Fit the binary Gaussian mixture model (GMM) on target data set by leveraging multiple source data sets under a transfer learning (TL) setting. This function implements the modified EM algorithm (Altorithm 4) proposed in Tian, Y., Weng, H., & Feng, Y. (2022).
#' @export
#' @param x design matrix of the target data set. Should be a \code{matrix} or \code{data.frame} object.
#' @param fitted_bar the output from \code{mtlgmm} function.
#' @param step_size step size choice in proximal gradient method to solve each optimization problem in the revised EM algorithm (Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)), which can be either "lipschitz" or "fixed". Default = "lipschitz".
#' \itemize{
#' \item lipschitz: \code{eta_w}, \code{eta_mu} and \code{eta_beta} will be chosen by the Lipschitz property of the gradient of objective function (without the penalty part). See Section 4.2 of Parikh, N., & Boyd, S. (2014).
#' \item fixed: \code{eta_w}, \code{eta_mu} and \code{eta_beta} need to be specified
#' }
#' @param eta_w step size in the proximal gradient method to learn w (Step 3 of Algorithm 4 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param eta_mu step size in the proximal gradient method to learn mu (Steps 4 and 5 of Algorithm 4 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param eta_beta step size in the proximal gradient method to learn beta (Step 7 of Algorithm 4 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param lambda_choice the choice of constants in the penalty parameter used in the optimization problems. See Algorithm 4 of Tian, Y., Weng, H., & Feng, Y. (2022), which can be either "fixed" or "cv". Default = "cv".
#' \itemize{
#' \item cv: \code{cv_nfolds}, \code{cv_upper}, and \code{cv_length} need to be specified. Then the C1 and C2 parameters will be chosen in all combinations in \code{exp(seq(log(cv_lower/10), log(cv_upper/10), length.out = cv_length))} via cross-validation. Note that this is a two-dimensional cv process, because we set \code{C1_w} = \code{C2_w}, \code{C1_mu} = \code{C1_beta} = \code{C2_mu} = \code{C2_beta} to reduce the computational cost.
#' \item fixed: \code{C1_w}, \code{C1_mu}, \code{C1_beta}, \code{C2_w}, \code{C2_mu}, and \code{C2_beta} need to be specified. See equations (19)-(24) in Tian, Y., Weng, H., & Feng, Y. (2022).
#' }
#' @param cv_nfolds the number of cross-validation folds. Default: 5
#' @param cv_upper the upper bound of \code{lambda} values used in cross-validation. Default: 5
#' @param cv_lower the lower bound of \code{lambda} values used in cross-validation. Default: 0.01
#' @param cv_length the number of \code{lambda} values considered in cross-validation. Default: 5
#' @param C1_w the initial value of C1_w. See equations (19) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.05
#' @param C1_mu the initial value of C1_mu. See equations (20) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C1_beta the initial value of C1_beta. See equations (21) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C2_w the initial value of C2_w. See equations (22) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.05
#' @param C2_mu the initial value of C2_mu. See equations (23) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C2_beta the initial value of C2_beta. See equations (24) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param kappa0 the decaying rate used in equation (19)-(24) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 1/3
#' @param tol maximum tolerance in all optimization problems. If the difference between last update and the current update is less than this value, the iterations of optimization will stop. Default: 1e-05
#' @param initial_method initialization method. This indicates the method to initialize the estimates of GMM parameters for each data set. Can be either "kmeans" or "EM".
#' \itemize{
#' \item kmeans: the initial estimates of GMM parameters will be generated from the single-task k-means algorithm. Will call \code{\link[stats]{kmeans}} function in \code{stats} package.
#' \item EM: the initial estimates of GMM parameters will be generated from the single-task EM algorithm. Will call \code{\link[mclust]{Mclust}} function in \code{mclust} package.
#' }
#' @param iter_max the maximum iteration number of the revised EM algorithm (i.e. the parameter T in Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 1000
#' @param iter_max_prox the maximum iteration number of the proximal gradient method. Default: 100
#' @param ncores the number of cores to use. Parallel computing is strongly suggested, specially when \code{lambda_choice} = "cv". Default: 1
#' @return A list with the following components.
#' \item{w}{the estimate of mixture proportion in GMMs for the target task. Will be a vector.}
#' \item{mu1}{the estimate of Gaussian mean in the first cluster of GMMs for the target task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{mu2}{the estimate of Gaussian mean in the second cluster of GMMs for the target task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{beta}{the estimate of the discriminant coefficient for the target task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{Sigma}{the estimate of the common covariance matrix for the target task. Will be a list, where each component represents the estimate for a task.}
#' \item{C1_w}{the initial value of C1_w.}
#' \item{C1_mu}{the initial value of C1_mu.}
#' \item{C1_beta}{the initial value of C1_beta.}
#' \item{C2_w}{the initial value of C2_w.}
#' \item{C2_mu}{the initial value of C2_mu.}
#' \item{C2_beta}{the initial value of C2_beta.}
#' @seealso \code{\link{mtlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'
#' Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and trends in Optimization, 1(3), 127-239.
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' ## Consider a transfer learning problem with 3 source tasks and 1 target task in the setting "MTL-1"
#' data_list_source <- data_generation(K = 3, outlier_K = 0, simulation_no = "MTL-1", h_w = 0,
#' h_mu = 0, n = 50)  # generate the source data
#' data_target <- data_generation(K = 1, outlier_K = 0, simulation_no = "MTL-1", h_w = 0.1,
#' h_mu = 1, n = 50)  # generate the target data
#' fit_mtl <- mtlgmm(x = data_list_source$data$x, C1_w = 0.05, C1_mu = 0.2, C1_beta = 0.2,
#' C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa = 1/3, initial_method = "EM",
#' trim = 0.1, lambda_choice = "fixed", step_size = "lipschitz")
#'
#' fit_tl <- tlgmm(x = data_target$data$x[[1]], fitted_bar = fit_mtl, C1_w = 0.05,
#' C1_mu = 0.2, C1_beta = 0.2, C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa0 = 1/3,
#' initial_method = "EM", ncores = 1, lambda_choice = "fixed", step_size = "lipschitz")
#'
#' \donttest{
#' # use cross-validation to choose the tuning parameters
#' # warning: can be quite slow, large "ncores" input is suggested!!
#' fit_tl <- tlgmm(x = data_target$data$x[[1]], fitted_bar = fit_mtl, kappa0 = 1/3,
#' initial_method = "EM", ncores = 2, lambda_choice = "cv", step_size = "lipschitz")
#' }


tlgmm <- function(x, fitted_bar, step_size = c("lipschitz", "fixed"), eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.1,
                  lambda_choice = c("fixed", "cv"), cv_nfolds = 5, cv_upper = 2, cv_lower = 0.01, cv_length = 5, C1_w = 0.05, C1_mu = 0.2,
                  C1_beta = 0.2, C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa0 = 1/3, tol = 1e-5,
                  initial_method = c("kmeans", "EM"), iter_max = 1000, iter_max_prox = 100, ncores = 1) {

  lambda_choice <- match.arg(lambda_choice)
  step_size <- match.arg(step_size)
  initial_method <- match.arg(initial_method)
  mtl_initial_mu1 <- fitted_bar$initial_mu1
  mtl_initial_mu2 <- fitted_bar$initial_mu2

  # basic parameters
  p <- ncol(x)
  n <- nrow(x)
  i <- 1 # just to avoid the error message "no visible binding for global variable 'i'"

  registerDoParallel(ncores)
  # cl <- makeCluster(ncores, outfile="")


  # initialization and alignment adjustment
  # ---------------------------------------------
  fitted_values <- initialize(list(x), initial_method)

  score1 <- score(rep(1,ncol(mtl_initial_mu1)+1), rep(2, ncol(mtl_initial_mu2)+1), mu1 = cbind(mtl_initial_mu1, fitted_values$mu1), mu2 = cbind(mtl_initial_mu2, fitted_values$mu2))
  score2 <- score(c(rep(1,ncol(mtl_initial_mu1)), 2), c(rep(2, ncol(mtl_initial_mu2)), 1), mu1 = cbind(mtl_initial_mu1, fitted_values$mu1), mu2 = cbind(mtl_initial_mu2, fitted_values$mu2))
  if (score2 < score1) { # swap two clusters of target data
    fitted_values <- alignment_swap(2, 1, initial_value_list = fitted_values)
  }


  # TL-EM
  # -------------------------
  w <- fitted_values$w
  mu1 <- fitted_values$mu1
  mu2 <- fitted_values$mu2
  beta <- fitted_values$beta
  Sigma <- fitted_values$Sigma[[1]]

  w.bar <- fitted_bar$w_bar
  mu1.bar <- fitted_bar$mu1_bar
  mu2.bar <- fitted_bar$mu2_bar
  beta.bar <- fitted_bar$beta_bar

  # we may need to define similar w.t etc...

  if (lambda_choice == "cv") {
    folds_index <- createFolds(1:n, k = cv_nfolds, list = TRUE)

    C_w <- exp(seq(log(cv_lower/10), log(cv_upper/10), length.out = cv_length))
    C_mu <- exp(seq(log(cv_lower), log(cv_upper), length.out = cv_length))
    C_matrix <- as.matrix(expand.grid(C_w, C_mu))

    emp_logL <- foreach(i = 1:nrow(C_matrix), .combine = "rbind", .packages = "mclust") %dopar% {
      C1_w <- C_matrix[i, 1]
      C2_w <- C_matrix[i, 1]
      C1_mu <- C_matrix[i, 2]
      C1_beta <- C_matrix[i, 2]
      C2_mu <- C_matrix[i, 2]
      C2_beta <- C_matrix[i, 2]

      C1_w.t <- C1_w
      C1_mu.t <- C1_mu
      C1_beta.t <- C1_beta

      C2_w.t <- C2_w
      C2_mu.t <- C2_mu
      C2_beta.t <- C2_beta

      sapply(1:cv_nfolds, function(j){
        x.train <- x[-folds_index[[j]], ]
        x.valid <- x[folds_index[[j]], ]
        n.train <- nrow(x.train)

        for (l in 1:iter_max) {
          gamma <- as.numeric(w/(w + (1-w)*exp(t(beta) %*% (t(x.train) - as.numeric(mu1 + mu2)/2))))
          if (l >= 2) {
            C1_w.t <- C1_w + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_w.t <- kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)

            C1_mu.t <- C1_mu + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_mu.t <- kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)

            C1_beta.t <- C1_beta + C1_mu.t + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_beta.t <- C2_mu.t + kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)
          }
          lambda.w <- C1_w.t*sqrt(p) + C2_w.t*n.train
          lambda.mu <- C1_mu.t*sqrt(p) + C2_mu.t*n.train
          lambda.beta <- C1_beta.t*sqrt(p) + C2_beta.t*n.train

          # w
          if (l == 1) {
            w.t <- w - w.bar
          }
          w.old.last.round <- w
          for (r in 1:iter_max_prox) {
            w.t.old <- w.t
            del <- w.t - eta_w*(w.t + w.bar - mean(gamma))
            w.t <- max(1-(eta_w*lambda.w/sqrt(n.train))/abs(del), 0)*del

            if (max(vec_norm(w.t - w.t.old)) <= tol) {
              break
            }
          }
          w.old <- w
          w <- w.t + w.bar

          # mu1
          if (l == 1) {
            mu1.t <- mu1 - mu1.bar
          }
          mu1.bar.old.last.round <- mu1.bar
          for (r in 1:iter_max_prox) {
            mu1.t.old <- mu1.t

            del <- mu1.t - eta_mu*colMeans(as.numeric(1-gamma)*(matrix(rep(mu1.t+mu1.bar, n.train), nrow = n.train, byrow = T) - x.train))
            mu1.t <- max(1-(eta_mu*lambda.mu/sqrt(n.train))/sqrt(sum(del^2)), 0)*del

            if (col_norm(mu1.t - mu1.t.old) <= tol) {
              break
            }
          }
          mu1.old <- mu1
          mu1 <- mu1.t + mu1.bar


          # mu2
          if (l == 1) {
            mu2.t <- mu2 - mu2.bar
          }
          mu2.bar.old.last.round <- mu2.bar
          for (r in 1:iter_max_prox) {
            mu2.t.old <- mu2.t

            del <- mu2.t - eta_mu*colMeans(as.numeric(gamma)*(matrix(rep(mu2.t+mu2.bar, n.train), nrow = n.train, byrow = T) - x.train))
            mu2.t <- max(1-(eta_mu*lambda.mu/sqrt(n.train))/sqrt(sum(del^2)), 0)*del

            if (col_norm(mu2.t - mu2.t.old) <= tol) {
              break
            }
          }
          mu2.old <- mu2
          mu2 <- mu2.t + mu2.bar

          # Sigma
          Sigma.old <- Sigma

          Sigma1 <- t(x.train - matrix(rep(mu1, n.train), nrow =  n.train, byrow = T)) %*% diag(1-as.numeric(gamma)) %*% (x.train - matrix(rep(mu1, n.train), nrow = n.train, byrow = T))
          Sigma2 <- t(x.train - matrix(rep(mu2, n.train), nrow =  n.train, byrow = T)) %*% diag(as.numeric(gamma)) %*% (x.train - matrix(rep(mu2, n.train), nrow = n.train, byrow = T))
          Sigma <- (Sigma1+Sigma2)/n.train

          # beta
          if (l == 1) {
            beta.t <- beta - beta.bar
          }
          beta.bar.old.last.round <- beta.bar

          if (step_size == "lipschitz") {
            eta_beta.list <- 1/(2*norm(Sigma, "2"))
          } else if (step_size == "fixed") {
            eta_beta.list <- eta_beta
          }

          for (r in 1:iter_max_prox) {
            beta.t.old <- beta.t

            eta_beta <- eta_beta.list
            del <- beta.t - eta_beta*(Sigma %*% (beta.t + beta.bar) - mu1 + mu2)
            beta.t <- max(1-(eta_beta*lambda.beta/sqrt(n.train))/sqrt(sum(del^2)), 0)*del

            if (col_norm(beta.t - beta.t.old) <= tol) {
              break
            }

          }

          beta.old <- beta
          beta <- beta.t + as.numeric(beta.bar)

          # check whether to terminate the interation process or not
          error <- max(vec_max_norm(w-w.old), col_norm(mu1 - mu1.old), col_norm(mu2 - mu2.old), col_norm(beta - beta.old),
                       norm(Sigma-Sigma.old, "2"))
          if (error <= tol) {
            break
          }
        }
        loss <- sum(log((1-w)*mclust::dmvnorm(data = x.valid, mean = mu1, sigma = Sigma) +
                          w*mclust::dmvnorm(data = x.valid, mean = mu2, sigma = Sigma)))
        sum(loss)
      })
    }

    C_w <- C_matrix[which.max(rowMeans(emp_logL)),1]
    C_mu <- C_beta <- C_matrix[which.max(rowMeans(emp_logL)),2]
    C1_w <- C2_w <- C_w
    C1_mu <- C1_beta <- C2_mu <- C2_beta <- C_mu
    lambda_choice <- "fixed" # run the algorithm again with optimal choice of parameters
  }

  if (lambda_choice == "fixed") {
    C1_w.t <- C1_w
    C1_mu.t <- C1_mu
    C1_beta.t <- C1_beta

    C2_w.t <- C2_w
    C2_mu.t <- C2_mu
    C2_beta.t <- C2_beta


    n <- nrow(x)

    for (l in 1:iter_max) {
      gamma <- as.numeric(w/(w + (1-w)*exp(t(beta) %*% (t(x) - (as.numeric(mu1 + mu2))/2))))
      if (l >= 2) {
        C1_w.t <- C1_w + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_w.t <- kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)

        C1_mu.t <- C1_mu + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_mu.t <- kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)

        C1_beta.t <- C1_beta + C1_mu.t + kappa0*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_beta.t <- C2_mu.t + kappa0*max(C2_w.t, C2_mu.t, C2_beta.t)
      }
      lambda.w <- C1_w.t*sqrt(p) + C2_w.t*n
      lambda.mu <- C1_mu.t*sqrt(p) + C2_mu.t*n
      lambda.beta <- C1_beta.t*sqrt(p) + C2_beta.t*n

      # w
      if (l == 1) {
        w.t <- w - w.bar
      }
      w.old.last.round <- w
      for (r in 1:iter_max_prox) {
        w.t.old <- w.t
        del <- w.t - eta_w*(w.t + w.bar - mean(gamma))
        w.t <- max(1-(eta_w*lambda.w/sqrt(n))/abs(del), 0)*del

        if (max(vec_norm(w.t - w.t.old)) <= tol) {
          break
        }
      }
      w.old <- w
      w <- w.t + w.bar

      # mu1
      if (l == 1) {
        mu1.t <- mu1 - mu1.bar
      }
      mu1.bar.old.last.round <- mu1.bar
      for (r in 1:iter_max_prox) {
        mu1.t.old <- mu1.t

        del <- mu1.t - eta_mu*colMeans(as.numeric(1-gamma)*(matrix(rep(mu1.t+mu1.bar, n), nrow = n, byrow = T) - x))
        mu1.t <- max(1-(eta_mu*lambda.mu/sqrt(n))/sqrt(sum(del^2)), 0)*del

        if (col_norm(mu1.t - mu1.t.old) <= tol) {
          break
        }
      }
      mu1.old <- mu1
      mu1 <- mu1.t + mu1.bar


      # mu2
      if (l == 1) {
        mu2.t <- mu2 - mu2.bar
      }
      mu2.bar.old.last.round <- mu2.bar
      for (r in 1:iter_max_prox) {
        mu2.t.old <- mu2.t

        del <- mu2.t - eta_mu*colMeans(as.numeric(gamma)*(matrix(rep(mu2.t+mu2.bar, n), nrow = n, byrow = T) - x))
        mu2.t <- max(1-(eta_mu*lambda.mu/sqrt(n))/sqrt(sum(del^2)), 0)*del

        if (col_norm(mu2.t - mu2.t.old) <= tol) {
          break
        }
      }
      mu2.old <- mu2
      mu2 <- mu2.t + mu2.bar

      # Sigma
      Sigma.old <- Sigma

      Sigma1 <- t(x - matrix(rep(mu1, n), nrow = n, byrow = T)) %*% diag(1-as.numeric(gamma)) %*% (x - matrix(rep(mu1, n), nrow = n, byrow = T))
      Sigma2 <- t(x - matrix(rep(mu2, n), nrow = n, byrow = T)) %*% diag(as.numeric(gamma)) %*% (x - matrix(rep(mu2, n), nrow = n, byrow = T))
      Sigma <- (Sigma1+Sigma2)/n

      # beta
      if (l == 1) {
        beta.t <- beta - beta.bar
      }
      beta.bar.old.last.round <- beta.bar

      if (step_size == "lipschitz") {
        eta_beta.list <- 1/(2*norm(Sigma, "2"))
      } else if (step_size == "fixed") {
        eta_beta.list <- eta_beta
      }

      for (r in 1:iter_max_prox) {
        beta.t.old <- beta.t

        eta_beta <- eta_beta.list
        del <- beta.t - eta_beta*(Sigma %*% (beta.t + beta.bar) - mu1 + mu2)
        beta.t <- max(1-(eta_beta*lambda.beta/sqrt(n))/sqrt(sum(del^2)), 0)*del

        if (col_norm(beta.t - beta.t.old) <= tol) {
          break
        }

      }

      beta.old <- beta
      beta <- beta.t + as.numeric(beta.bar)

      # check whether to terminate the interation process or not
      error <- max(vec_max_norm(w-w.old), col_norm(mu1 - mu1.old), col_norm(mu2 - mu2.old), col_norm(beta - beta.old),
                   norm(Sigma-Sigma.old, "2"))
      if (error <= tol) {
        break
      }
    }
  }


  stopImplicitCluster()
  # stopCluster(cl)


  return(list(w = w, mu1 = mu1, mu2 = mu2, beta = beta, Sigma = Sigma,
              C1_w = C1_w, C1_mu = C1_mu, C1_beta = C1_beta, C2_w = C2_w, C2_mu = C2_mu,
              C2_beta = C2_beta))


}
