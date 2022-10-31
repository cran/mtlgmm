#' @title Fit binary Gaussian mixture models (GMMs) on multiple data sets under a multi-task learning (MTL) setting.
#'
#' @description it binary Gaussian mixture models (GMMs) on multiple data sets under a multi-task learning (MTL) setting. This function implements the modified EM algorithm (Altorithm 1) proposed in Tian, Y., Weng, H., & Feng, Y. (2022).
#' @export
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom doParallel registerDoParallel stopImplicitCluster
#' @importFrom mclust dmvnorm Mclust mclustBIC
#' @importFrom stats runif rnorm kmeans rt uniroot quantile
#' @importFrom caret createFolds
#' @param x design matrices from multiple data sets. Should be a list, of which each component is a \code{matrix} or \code{data.frame} object, representing the design matrix from each task.
#' @param step_size step size choice in proximal gradient method to solve each optimization problem in the revised EM algorithm (Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)), which can be either "lipschitz" or "fixed". Default = "lipschitz".
#' \itemize{
#' \item lipschitz: \code{eta_w}, \code{eta_mu} and \code{eta_beta} will be chosen by the Lipschitz property of the gradient of objective function (without the penalty part). See Section 4.2 of Parikh, N., & Boyd, S. (2014).
#' \item fixed: \code{eta_w}, \code{eta_mu} and \code{eta_beta} need to be specified
#' }
#' @param eta_w step size in the proximal gradient method to learn w (Step 3 of Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param eta_mu step size in the proximal gradient method to learn mu (Steps 4 and 5 of Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param eta_beta step size in the proximal gradient method to learn beta (Step 9 of Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 0.1. Only used when \code{step_size} = "fixed".
#' @param lambda_choice the choice of constants in the penalty parameter used in the optimization problems. See Algorithm 1 of Tian, Y., Weng, H., & Feng, Y. (2022), which can be either "fixed" or "cv". Default: "cv".
#' \itemize{
#' \item cv: \code{cv_nfolds}, \code{cv_upper}, and \code{cv_length} need to be specified. Then the C1 and C2 parameters will be chosen in all combinations in \code{exp(seq(log(cv_lower/10), log(cv_upper/10), length.out = cv_length))} via cross-validation. Note that this is a two-dimensional cv process, because we set \code{C1_w} = \code{C2_w}, \code{C1_mu} = \code{C1_beta} = \code{C2_mu} = \code{C2_beta} to reduce the computational cost.
#' \item fixed: \code{C1_w}, \code{C1_mu}, \code{C1_beta}, \code{C2_w}, \code{C2_mu}, and \code{C2_beta} need to be specified. See equations (7)-(12) in Tian, Y., Weng, H., & Feng, Y. (2022).
#' }
#' @param cv_nfolds the number of cross-validation folds. Default: 5
#' @param cv_upper the upper bound of \code{lambda} values used in cross-validation. Default: 5
#' @param cv_lower the lower bound of \code{lambda} values used in cross-validation. Default: 0.01
#' @param cv_length the number of \code{lambda} values considered in cross-validation. Default: 5
#' @param C1_w the initial value of C1_w. See equations (7) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.05
#' @param C1_mu the initial value of C1_mu. See equations (8) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C1_beta the initial value of C1_beta. See equations (9) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C2_w the initial value of C2_w. See equations (10) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.05
#' @param C2_mu the initial value of C2_mu. See equations (11) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param C2_beta the initial value of C2_beta. See equations (12) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 0.2
#' @param kappa the decaying rate used in equation (7)-(12) in Tian, Y., Weng, H., & Feng, Y. (2022). Default: 1/3
#' @param tol maximum tolerance in all optimization problems. If the difference between last update and the current update is less than this value, the iterations of optimization will stop. Default: 1e-05
#' @param initial_method initialization method. This indicates the method to initialize the estimates of GMM parameters for each data set. Can be either "EM" or "kmeans". Default: "EM".
#' \itemize{
#' \item EM: the initial estimates of GMM parameters will be generated from the single-task EM algorithm. Will call \code{\link[mclust]{Mclust}} function in \code{mclust} package.
#' \item kmeans: the initial estimates of GMM parameters will be generated from the single-task k-means algorithm. Will call \code{\link[stats]{kmeans}} function in \code{stats} package.
#' }
#' @param alignment_method the alignment algorithm to use. See Section 2.4 of Tian, Y., Weng, H., & Feng, Y. (2022). Can either be "exhaustive" or "greedy". Default: when \code{length(x)} <= 10, "exhaustive" will be used, otherwise "greedy" will be used.
#' \itemize{
#' \item exhaustive: exhaustive search algorithm (Algorithm 2 in Tian, Y., Weng, H., & Feng, Y. (2022)) will be used.
#' \item greedy: greey label swapping algorithm (Algorithm 3 in Tian, Y., Weng, H., & Feng, Y. (2022)) will be used.
#' }
#' @param trim the proportion of trimmed data sets in the cross-validation procedure of choosing tuning parameters. Setting it to a non-zero small value can help avoid the impact of outlier tasks on the choice of tuning parameters. Default: 0.1
#' @param iter_max the maximum iteration number of the revised EM algorithm (i.e. the parameter T in Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: 1000
#' @param iter_max_prox the maximum iteration number of the proximal gradient method. Default: 100
#' @param ncores the number of cores to use. Parallel computing is strongly suggested, specially when \code{lambda_choice} = "cv". Default: 1
#' @return A list with the following components.
#' \item{w}{the estimate of mixture proportion in GMMs for each task. Will be a vector.}
#' \item{mu1}{the estimate of Gaussian mean in the first cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{mu2}{the estimate of Gaussian mean in the second cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{beta}{the estimate of the discriminant coefficient for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{Sigma}{the estimate of the common covariance matrix for each task. Will be a list, where each component represents the estimate for a task.}
#' \item{w_bar}{the center estimate of w. Numeric. See Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022). }
#' \item{mu1_bar}{the center estimate of mu1. Will be a vector. See Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022).}
#' \item{mu2_bar}{the center estimate of mu2. Will be a vector. See Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022).}
#' \item{beta_bar}{the center estimate of beta. Will be a vector. See Algorithm 1 in Tian, Y., Weng, H., & Feng, Y. (2022).}
#' \item{C1_w}{the initial value of C1_w.}
#' \item{C1_mu}{the initial value of C1_mu.}
#' \item{C1_beta}{the initial value of C1_beta.}
#' \item{C2_w}{the initial value of C2_w.}
#' \item{C2_mu}{the initial value of C2_mu.}
#' \item{C2_beta}{the initial value of C2_beta.}
#' \item{initial_mu1}{the well-aligned initial estimate of mu1 of different tasks. Useful for the alignment problem in transfer learning. See Section 3.4 in Tian, Y., Weng, H., & Feng, Y. (2022).}
#' \item{initial_mu2}{the well-aligned initial estimate of mu2 of different tasks. Useful for the alignment problem in transfer learning. See Section 3.4 in Tian, Y., Weng, H., & Feng, Y. (2022).}
#' @seealso \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'
#' Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and trends in Optimization, 1(3), 127-239.
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' library(mclust)
#' ## Consider a 5-task multi-task learning problem in the setting "MTL-1"
#' data_list <- data_generation(K = 5, outlier_K = 1, simulation_no = "MTL-1",
#' h_w = 0.1, h_mu = 1, n = 50)  # generate the data
#' fit <- mtlgmm(x = data_list$data$x, C1_w = 0.05, C1_mu = 0.2, C1_beta = 0.2,
#' C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa = 1/3, initial_method = "EM",
#' trim = 0.1, lambda_choice = "fixed", step_size = "lipschitz")
#'
#'
#' ## compare the performance with that of single-task estimators
#' # fit single-task GMMs
#' fitted_values <- initialize(data_list$data$x, "EM")  # initilize the estimates
#' L <- alignment(fitted_values$mu1, fitted_values$mu2,
#' method = "exhaustive")  # call the alignment algorithm
#' fitted_values <- alignment_swap(L$L1, L$L2,
#' initial_value_list = fitted_values)  # obtain the well-aligned initial estimates
#'
#' # fit a pooled GMM
#' x.comb <- Reduce("rbind", data_list$data$x)
#' fit_pooled <- Mclust(x.comb, G = 2, modelNames = "EEE")
#' fitted_values_pooled <- list(w = NULL, mu1 = NULL, mu2 = NULL, beta = NULL, Sigma = NULL)
#' fitted_values_pooled$w <- rep(fit_pooled$parameters$pro[1], length(data_list$data$x))
#' fitted_values_pooled$mu1 <- matrix(rep(fit_pooled$parameters$mean[,1],
#' length(data_list$data$x)), ncol = length(data_list$data$x))
#' fitted_values_pooled$mu2 <- matrix(rep(fit_pooled$parameters$mean[,2],
#' length(data_list$data$x)), ncol = length(data_list$data$x))
#' fitted_values_pooled$Sigma <- sapply(1:length(data_list$data$x), function(k){
#'   fit_pooled$parameters$variance$Sigma
#' }, simplify = FALSE)
#' fitted_values_pooled$beta <- sapply(1:length(data_list$data$x), function(k){
#'   solve(fit_pooled$parameters$variance$Sigma) %*%
#'   (fit_pooled$parameters$mean[,1] - fit_pooled$parameters$mean[,2])
#' })
#' error <- matrix(nrow = 3, ncol = 4, dimnames = list(c("Single-task-GMM","Pooled-GMM","MTL-GMM"),
#' c("w", "mu", "beta", "Sigma")))
#' error["Single-task-GMM", "w"] <- estimation_error(
#' fitted_values$w[-data_list$data$outlier_index],
#' data_list$parameter$w[-data_list$data$outlier_index], "w")
#' error["Pooled-GMM", "w"] <- estimation_error(
#' fitted_values_pooled$w[-data_list$data$outlier_index],
#' data_list$parameter$w[-data_list$data$outlier_index], "w")
#' error["MTL-GMM", "w"] <- estimation_error(
#' fit$w[-data_list$data$outlier_index],
#' data_list$parameter$w[-data_list$data$outlier_index], "w")
#'
#' error["Single-task-GMM", "mu"] <- estimation_error(
#' list(fitted_values$mu1[, -data_list$data$outlier_index],
#' fitted_values$mu2[, -data_list$data$outlier_index]),
#' list(data_list$parameter$mu1[, -data_list$data$outlier_index],
#' data_list$parameter$mu2[, -data_list$data$outlier_index]), "mu")
#' error["Pooled-GMM", "mu"] <- estimation_error(list(
#' fitted_values_pooled$mu1[, -data_list$data$outlier_index],
#' fitted_values_pooled$mu2[, -data_list$data$outlier_index]),
#' list(data_list$parameter$mu1[, -data_list$data$outlier_index],
#' data_list$parameter$mu2[, -data_list$data$outlier_index]), "mu")
#' error["MTL-GMM", "mu"] <- estimation_error(list(
#' fit$mu1[, -data_list$data$outlier_index],
#' fit$mu2[, -data_list$data$outlier_index]),
#' list(data_list$parameter$mu1[, -data_list$data$outlier_index],
#' data_list$parameter$mu2[, -data_list$data$outlier_index]), "mu")
#'
#' error["Single-task-GMM", "beta"]  <- estimation_error(
#' fitted_values$beta[, -data_list$data$outlier_index],
#' data_list$parameter$beta[, -data_list$data$outlier_index], "beta")
#' error["Pooled-GMM", "beta"] <- estimation_error(
#' fitted_values_pooled$beta[, -data_list$data$outlier_index],
#' data_list$parameter$beta[, -data_list$data$outlier_index], "beta")
#' error["MTL-GMM", "beta"] <- estimation_error(
#' fit$beta[, -data_list$data$outlier_index],
#' data_list$parameter$beta[, -data_list$data$outlier_index], "beta")
#'
#' error["Single-task-GMM", "Sigma"] <- estimation_error(
#' fitted_values$Sigma[-data_list$data$outlier_index],
#' data_list$parameter$Sigma[-data_list$data$outlier_index], "Sigma")
#' error["Pooled-GMM", "Sigma"] <- estimation_error(
#' fitted_values_pooled$Sigma[-data_list$data$outlier_index],
#' data_list$parameter$Sigma[-data_list$data$outlier_index], "Sigma")
#' error["MTL-GMM", "Sigma"] <- estimation_error(
#' fit$Sigma[-data_list$data$outlier_index],
#' data_list$parameter$Sigma[-data_list$data$outlier_index], "Sigma")
#'
#' error
#'
#' \donttest{
#' # use cross-validation to choose the tuning parameters
#' # warning: can be quite slow, large "ncores" input is suggested!!
#' fit <- mtlgmm(x = data_list$data$x, kappa = 1/3, initial_method = "EM", ncores = 2, cv_length = 5,
#' trim = 0.1, cv_upper = 2, cv_lower = 0.01, lambda = "cv", step_size = "lipschitz")
#' }
#'


mtlgmm <- function(x, step_size = c( "lipschitz", "fixed"), eta_w = 0.1, eta_mu = 0.1, eta_beta = 0.1,
                   lambda_choice = c("cv", "fixed"), cv_nfolds = 5, cv_upper = 5, cv_lower = 0.01, cv_length = 5, C1_w = 0.05, C1_mu = 0.2,
                   C1_beta = 0.2, C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa = 1/3, tol = 1e-5,
                   initial_method = c("EM", "kmeans"), alignment_method = ifelse(length(x) <= 10, "exhaustive", "greedy"),
                   trim = 0.1, iter_max = 1000, iter_max_prox = 100, ncores = 1) {

  lambda_choice <- match.arg(lambda_choice)
  step_size <- match.arg(step_size)
  initial_method <- match.arg(initial_method)

  # basic parameters
  p <- ncol(x[[1]])
  K <- length(x)
  n <- sapply(1:K, function(k){nrow(x[[k]])})
  i <- NULL # just to avoid the error message "no visible binding for global variable 'i'"

  registerDoParallel(ncores)
  # cl <- makeCluster(ncores, outfile="")


  # initialization and alignment adjustment
  # ---------------------------------------------
  fitted_values <- initialize(x, initial_method)
  if (alignment_method == "exhaustive") {
    L <- alignment(fitted_values$mu1, fitted_values$mu2, method = "exhaustive")
    fitted_values <- alignment_swap(L$L1, L$L2, initial_value_list = fitted_values)
  } else if (alignment_method == "greedy") {
    L <- alignment(fitted_values$mu1, fitted_values$mu2, method = "greedy")
    fitted_values <- alignment_swap(L$L1, L$L2, initial_value_list = fitted_values)
  }


  # MTL-EM
  # -------------------------
  w <- fitted_values$w
  mu1 <- fitted_values$mu1
  mu2 <- fitted_values$mu2
  beta <- fitted_values$beta
  Sigma <- fitted_values$Sigma


  if (lambda_choice == "cv") {
    folds_index <- sapply(1:K, function(k){
      createFolds(1:n[k], k = cv_nfolds, list = TRUE)
    }, simplify = FALSE)

    C_w <- exp(seq(log(cv_lower/10), log(cv_upper/10), length.out = cv_length))
    C_mu <- exp(seq(log(cv_lower), log(cv_upper), length.out = cv_length))
    C_matrix <- as.matrix(expand.grid(C_w, C_mu))

    emp_logL <- foreach(i = 1:nrow(C_matrix), .combine = "rbind") %dopar% {
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
        x.train <- sapply(1:K, function(k){
          x[[k]][-folds_index[[k]][[j]], ]
        }, simplify = FALSE)

        x.valid <- sapply(1:K, function(k){
          x[[k]][folds_index[[k]][[j]], ]
        }, simplify = FALSE)

        n.train <- sapply(1:K, function(k){nrow(x.train[[k]])})

        for (l in 1:iter_max) {
          gamma <- sapply(1:K, function(k){
            as.numeric(w[k]/(w[k] + (1-w[k])*exp(t(beta[, k]) %*% (t(x.train[[k]]) - (mu1[, k] + mu2[, k])/2))))
          }, simplify = FALSE)

          if (l >= 2) {
            C1_w.t <- C1_w + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_w.t <- kappa*max(C2_w.t, C2_mu.t, C2_beta.t)

            C1_mu.t <- C1_mu + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_mu.t <- kappa*max(C2_w.t, C2_mu.t, C2_beta.t)

            C1_beta.t <- C1_beta + C1_mu.t + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
            C2_beta.t <- C2_mu.t + kappa*max(C2_w.t, C2_mu.t, C2_beta.t)
          }
          lambda.w <- C1_w.t*sqrt(p+log(K)) + C2_w.t*max(n.train)
          lambda.mu <- C1_mu.t*sqrt(p+log(K)) + C2_mu.t*max(n.train)
          lambda.beta <- C1_beta.t*sqrt(p+log(K)) + C2_beta.t*max(n.train)

          # w
          if (l == 1) {
            w.bar <- mean(w)
            w.t <- w - w.bar
          }
          w.bar.old.last.round <- w.bar
          for (r in 1:iter_max_prox) {
            w.t.old <- w.t
            w.t <- sapply(1:K, function(k){
              del <- w.t[k] - eta_w*(w.t[k] + w.bar - mean(gamma[[k]]))
              max(1-(eta_w*lambda.w/sqrt(n.train[k]))/abs(del), 0)*del
            })


            w.bar.old <- w.bar
            w.bar <- sum(sapply(1:K, function(k){
              sum(gamma[[k]] - w.t[k])
            }))/sum(n.train)

            if (max(vec_max_norm(w.t - w.t.old), vec_max_norm(w.bar - w.bar.old)) <= tol) {
              break
            }
          }
          w.old <- w
          w <- w.t + w.bar

          # mu1
          if (l == 1) {
            mu1.bar <- rowMeans(mu1)
            mu1.t <- mu1 - mu1.bar
          }
          mu1.bar.old.last.round <- mu1.bar
          for (r in 1:iter_max_prox) {
            mu1.t.old <- mu1.t
            mu1.t <- sapply(1:K, function(k){
              del <- mu1.t[, k] - eta_mu*colMeans(as.numeric(1-gamma[[k]])*(matrix(rep(mu1.t[, k]+mu1.bar, n.train[k]), nrow = n.train[k], byrow = T) - x.train[[k]]))
              max(1-(eta_mu*lambda.mu/sqrt(n.train[k]))/sqrt(sum(del^2)), 0)*del
            })

            mu1.bar.old <- mu1.bar
            mu1.bar <- rowSums(sapply(1:K, function(k){
              colSums(as.numeric(1-gamma[[k]])*(x.train[[k]] - matrix(rep(mu1.t[, k], n.train[k]), nrow =  n.train[k], byrow = T)))
            }))/sum(1-Reduce("c", gamma))

            if (max(col_norm(mu1.t - mu1.t.old), vec_norm(mu1.bar - mu1.bar.old)) <= tol) {
              break
            }
          }
          mu1.old <- mu1
          mu1 <- mu1.t + mu1.bar


          # mu2
          if (l == 1) {
            mu2.bar <- rowMeans(mu2)
            mu2.t <- mu2 - mu2.bar
          }
          mu2.bar.old.last.round <- mu2.bar
          for (r in 1:iter_max_prox) {
            mu2.t.old <- mu2.t
            mu2.t <- sapply(1:K, function(k){
              del <- mu2.t[, k] - eta_mu*colMeans(as.numeric(gamma[[k]])*(matrix(rep(mu2.t[, k]+mu2.bar, n.train[k]), nrow = n.train[k], byrow = T) - x.train[[k]]))
              max(1-(eta_mu*lambda.mu/sqrt(n.train[k]))/sqrt(sum(del^2)), 0)*del
            })

            mu2.bar.old <- mu2.bar
            mu2.bar <- rowSums(sapply(1:K, function(k){
              colSums(as.numeric(gamma[[k]])*(x.train[[k]] - matrix(rep(mu2.t[, k], n.train[k]), nrow =  n.train[k], byrow = T)))
            }))/sum(Reduce("c", gamma))

            if (max(col_norm(mu2.t - mu2.t.old), vec_norm(mu2.bar - mu2.bar.old)) <= tol) {
              break
            }
          }
          mu2.old <- mu2
          mu2 <- mu2.t + mu2.bar

          # Sigma
          Sigma.old <- Sigma
          Sigma <- sapply(1:K, function(k){
            Sigma1 <- t(x.train[[k]] - matrix(rep(mu1[, k], n.train[k]), nrow =  n.train[k], byrow = T)) %*% diag(1-as.numeric(gamma[[k]])) %*% (x.train[[k]] - matrix(rep(mu1[, k], n.train[k]), nrow = n.train[k], byrow = T))
            Sigma2 <- t(x.train[[k]] - matrix(rep(mu2[, k], n.train[k]), nrow =  n.train[k], byrow = T)) %*% diag(as.numeric(gamma[[k]])) %*% (x.train[[k]] - matrix(rep(mu2[, k], n.train[k]), nrow = n.train[k], byrow = T))
            (Sigma1+Sigma2)/n.train[k]
          }, simplify = FALSE)


          # beta
          if (l == 1) {
            beta.bar <- rowMeans(beta)
            beta.t <- beta - beta.bar
          }
          beta.bar.old.last.round <- beta.bar

          if (step_size == "lipschitz") {
            eta_beta.list <- sapply(1:K, function(k){
              1/(2*norm(Sigma[[k]], "2"))
            })
          } else if (step_size == "fixed") {
            eta_beta.list <- rep(eta_beta, K)
          }

          for (r in 1:iter_max_prox) {
            beta.t.old <- beta.t
            beta.t <- sapply(1:K, function(k){
              eta_beta <- eta_beta.list[k]
              del <- beta.t[, k] - eta_beta*(Sigma[[k]] %*% (beta.t[, k] + beta.bar) - mu1[, k] + mu2[, k])
              max(1-(eta_beta*lambda.beta/sqrt(n.train[k]))/sqrt(sum(del^2)), 0)*del
            })

            Sigma.weighted.sum <- Reduce("+", sapply(1:K, function(k){
              n.train[k]*Sigma[[k]]
            }, simplify = FALSE))

            vector.weighted.sum <- rowSums(sapply(1:K, function(k){
              n.train[k]*(-Sigma[[k]]%*%beta.t[, k] + mu1[, k] - mu2[, k])
            }))


            beta.bar.old <- beta.bar
            beta.bar <- as.numeric(solve(Sigma.weighted.sum) %*% vector.weighted.sum)

            if (max(col_norm(beta.t - beta.t.old), vec_norm(beta.bar - beta.bar.old)) <= tol) {
              break
            }

          }

          beta.old <- beta
          beta <- beta.t + as.numeric(beta.bar)

          # check whether to terminate the interation process or not
          error <- max(vec_max_norm(w-w.old), col_norm(mu1 - mu1.old), col_norm(mu2 - mu2.old), col_norm(beta - beta.old),
                       max(sapply(1:K, function(k){norm(Sigma[[k]]-Sigma.old[[k]], "2")})),
                       abs(w.bar - w.bar.old.last.round), vec_norm(mu1.bar - mu1.bar.old.last.round),
                       vec_norm(mu2.bar - mu2.bar.old.last.round), vec_norm(beta.bar - beta.bar.old.last.round))
          if (error <= tol) {
            break
          }
        }
        loss <- sort(sapply(1:K, function(k){
          sum(log((1-w[k])*dmvnorm(data = x.valid[[k]], mean = mu1[, k], sigma = Sigma[[k]]) +
                    w[k]*dmvnorm(data = x.valid[[k]], mean = mu2[, k], sigma = Sigma[[k]])))
          # sum((1-w[k])*log(mclust::dmvnorm(data = x.valid[[k]], mean = mu1[, k], sigma = Sigma[[k]])) +
          #           w[k]*log(mclust::dmvnorm(data = x.valid[[k]], mean = mu2[, k], sigma = Sigma[[k]])))
        }), decreasing = FALSE)
        if (trim >= 0) {
          trim_ind <- quantile(1:K, c(trim, 1-trim), type = 1)
          sum(loss[-c(1:floor(trim_ind[1]), ceiling(trim_ind[2]):K)])
        } else {
          sum(loss)
        }
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


    for (l in 1:iter_max) {
      gamma <- sapply(1:K, function(k){
        as.numeric(w[k]/(w[k] + (1-w[k])*exp(t(beta[, k]) %*% (t(x[[k]]) - (mu1[, k] + mu2[, k])/2))))
      }, simplify = FALSE)

      if (l >= 2) {
        C1_w.t <- C1_w + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_w.t <- kappa*max(C2_w.t, C2_mu.t, C2_beta.t)

        C1_mu.t <- C1_mu + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_mu.t <- kappa*max(C2_w.t, C2_mu.t, C2_beta.t)

        C1_beta.t <- C1_beta + C1_mu.t + kappa*max(C1_w.t, C1_mu.t, C1_beta.t)
        C2_beta.t <- C2_mu.t + kappa*max(C2_w.t, C2_mu.t, C2_beta.t)
      }
      lambda.w <- C1_w.t*sqrt(p+log(K)) + C2_w.t*max(n)
      lambda.mu <- C1_mu.t*sqrt(p+log(K)) + C2_mu.t*max(n)
      lambda.beta <- C1_beta.t*sqrt(p+log(K)) + C2_beta.t*max(n)

      # w
      if (l == 1) {
        w.bar <- mean(w)
        w.t <- w - w.bar
      }
      w.bar.old.last.round <- w.bar
      for (r in 1:iter_max_prox) {
        w.t.old <- w.t
        w.t <- sapply(1:K, function(k){
          del <- w.t[k] - eta_w*(w.t[k] + w.bar - mean(gamma[[k]]))
          max(1-(eta_w*lambda.w/sqrt(n[k]))/abs(del), 0)*del
        })


        w.bar.old <- w.bar
        w.bar <- sum(sapply(1:K, function(k){
          sum(gamma[[k]] - w.t[k])
        }))/sum(n)

        if (max(vec_max_norm(w.t - w.t.old), vec_max_norm(w.bar - w.bar.old)) <= tol) {
          break
        }
      }
      w.old <- w
      w <- w.t + w.bar

      # mu1
      if (l == 1) {
        mu1.bar <- rowMeans(mu1)
        mu1.t <- mu1 - mu1.bar
      }
      mu1.bar.old.last.round <- mu1.bar
      for (r in 1:iter_max_prox) {
        mu1.t.old <- mu1.t
        mu1.t <- sapply(1:K, function(k){
          del <- mu1.t[, k] - eta_mu*colMeans(as.numeric(1-gamma[[k]])*(matrix(rep(mu1.t[, k]+mu1.bar, n[k]), nrow = n[k], byrow = T) - x[[k]]))
          max(1-(eta_mu*lambda.mu/sqrt(n[k]))/sqrt(sum(del^2)), 0)*del
        })

        mu1.bar.old <- mu1.bar
        mu1.bar <- rowSums(sapply(1:K, function(k){
          colSums(as.numeric(1-gamma[[k]])*(x[[k]] - matrix(rep(mu1.t[, k], n[k]), nrow =  n[k], byrow = T)))
        }))/sum(1-Reduce("c", gamma))

        if (max(col_norm(mu1.t - mu1.t.old), vec_norm(mu1.bar - mu1.bar.old)) <= tol) {
          break
        }
      }
      mu1.old <- mu1
      mu1 <- mu1.t + mu1.bar


      # mu2
      if (l == 1) {
        mu2.bar <- rowMeans(mu2)
        mu2.t <- mu2 - mu2.bar
      }
      mu2.bar.old.last.round <- mu2.bar
      for (r in 1:iter_max_prox) {
        mu2.t.old <- mu2.t
        mu2.t <- sapply(1:K, function(k){
          del <- mu2.t[, k] - eta_mu*colMeans(as.numeric(gamma[[k]])*(matrix(rep(mu2.t[, k]+mu2.bar, n[k]), nrow = n[k], byrow = T) - x[[k]]))
          max(1-(eta_mu*lambda.mu/sqrt(n[k]))/sqrt(sum(del^2)), 0)*del
        })

        mu2.bar.old <- mu2.bar
        mu2.bar <- rowSums(sapply(1:K, function(k){
          colSums(as.numeric(gamma[[k]])*(x[[k]] - matrix(rep(mu2.t[, k], n[k]), nrow =  n[k], byrow = T)))
        }))/sum(Reduce("c", gamma))

        if (max(col_norm(mu2.t - mu2.t.old), vec_norm(mu2.bar - mu2.bar.old)) <= tol) {
          break
        }
      }
      mu2.old <- mu2
      mu2 <- mu2.t + mu2.bar

      # Sigma
      Sigma.old <- Sigma
      Sigma <- sapply(1:K, function(k){
        Sigma1 <- t(x[[k]] - matrix(rep(mu1[, k], n[k]), nrow =  n[k], byrow = T)) %*% diag(1-as.numeric(gamma[[k]])) %*% (x[[k]] - matrix(rep(mu1[, k], n[k]), nrow = n[k], byrow = T))
        Sigma2 <- t(x[[k]] - matrix(rep(mu2[, k], n[k]), nrow =  n[k], byrow = T)) %*% diag(as.numeric(gamma[[k]])) %*% (x[[k]] - matrix(rep(mu2[, k], n[k]), nrow = n[k], byrow = T))
        (Sigma1+Sigma2)/n[k]
      }, simplify = FALSE)


      # beta
      if (l == 1) {
        beta.bar <- rowMeans(beta)
        beta.t <- beta - beta.bar
      }
      beta.bar.old.last.round <- beta.bar

      eta_beta.list <- sapply(1:K, function(k){
        1/(3*norm(Sigma[[k]], "2"))
      })
      for (r in 1:iter_max_prox) {
        beta.t.old <- beta.t
        beta.t <- sapply(1:K, function(k){
          eta_beta <- eta_beta.list[k]
          del <- beta.t[, k] - eta_beta*(Sigma[[k]] %*% (beta.t[, k] + beta.bar) - mu1[, k] + mu2[, k])
          max(1-(eta_beta*lambda.beta/sqrt(n[k]))/sqrt(sum(del^2)), 0)*del
        })

        Sigma.weighted.sum <- Reduce("+", sapply(1:K, function(k){
          n[k]*Sigma[[k]]
        }, simplify = FALSE))

        vector.weighted.sum <- rowSums(sapply(1:K, function(k){
          n[k]*(-Sigma[[k]]%*%beta.t[, k] + mu1[, k] - mu2[, k])
        }))


        beta.bar.old <- beta.bar
        beta.bar <- as.numeric(solve(Sigma.weighted.sum) %*% vector.weighted.sum)

        if (max(col_norm(beta.t - beta.t.old), vec_norm(beta.bar - beta.bar.old)) <= tol) {
          break
        }
        # print(max(col_norm(beta.t - beta.t.old), vec_norm(beta.bar - beta.bar.old)))

      }

      beta.old <- beta
      beta <- beta.t + as.numeric(beta.bar)

      # check whether to terminate the interation process or not
      error <- max(vec_max_norm(w-w.old), col_norm(mu1 - mu1.old), col_norm(mu2 - mu2.old), col_norm(beta - beta.old),
                   max(sapply(1:K, function(k){norm(Sigma[[k]]-Sigma.old[[k]], "2")})),
                   abs(w.bar - w.bar.old.last.round), vec_norm(mu1.bar - mu1.bar.old.last.round),
                   vec_norm(mu2.bar - mu2.bar.old.last.round), vec_norm(beta.bar - beta.bar.old.last.round))

      if (error <= tol) {
        break
      }
    }
  }


  stopImplicitCluster()
  # stopCluster(cl)


  return(list(w = w, mu1 = mu1, mu2 = mu2, beta = beta, Sigma = Sigma, w_bar = w.bar, mu1_bar = mu1.bar, mu2_bar = mu2.bar,
              beta_bar = beta.bar, C1_w = C1_w, C1_mu = C1_mu, C1_beta = C1_beta, C2_w = C2_w, C2_mu = C2_mu,
              C2_beta = C2_beta, initial_mu1 = fitted_values$mu1, initial_mu2 = fitted_values$mu2))


}
