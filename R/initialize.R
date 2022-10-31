#' @title Initialize the estimators of GMM parameters on each task.
#'
#' @description Initialize the estimators of GMM parameters on each task.
#' @export
#' @param x design matrices from multiple data sets. Should be a list, of which each component is a \code{matrix} or \code{data.frame} object, representing the design matrix from each task.
#' @param method initialization method. This indicates the method to initialize the estimates of GMM parameters for each data set. Can be either "EM" or "kmeans". Default: "EM".
#' \itemize{
#' \item EM: the initial estimates of GMM parameters will be generated from the single-task EM algorithm. Will call \code{\link[mclust]{Mclust}} function in \code{mclust} package.
#' \item kmeans: the initial estimates of GMM parameters will be generated from the single-task k-means algorithm. Will call \code{\link[stats]{kmeans}} function in \code{stats} package.
#' }
#' @return A list with the following components.
#' \item{w}{the estimate of mixture proportion in GMMs for each task. Will be a vector.}
#' \item{mu1}{the estimate of Gaussian mean in the first cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{mu2}{the estimate of Gaussian mean in the second cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{beta}{the estimate of the discriminant coefficient for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{Sigma}{the estimate of the common covariance matrix for each task. Will be a list, where each component represents the estimate for a task.}
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' ## Consider a 5-task multi-task learning problem in the setting "MTL-1"
#' data_list <- data_generation(K = 5, outlier_K = 1, simulation_no = "MTL-1", h_w = 0.1,
#' h_mu = 1, n = 50)  # generate the data
#' fit <- mtlgmm(x = data_list$data$x, C1_w = 0.05, C1_mu = 0.2, C1_beta = 0.2,
#' C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa = 1/3, initial_method = "EM",
#' trim = 0.1, lambda_choice = "fixed", step_size = "lipschitz")
#'
#' ## Initialize the estimators of GMM parameters on each task.
#' fitted_values_EM <- initialize(data_list$data$x,
#' "EM")  # initilize the estimates by single-task EM algorithm
#' fitted_values_kmeans <- initialize(data_list$data$x,
#' "EM")  # initilize the estimates by single-task k-means


initialize <- function(x, method = c("kmeans", "EM")) {
  method <- match.arg(method)
  K <- length(x)
  if (method == "kmeans") {
    fit_kmeans <- sapply(1:K, function(k){kmeans(x[[k]], centers = 2)}, simplify = FALSE)
    w_0 <- sapply(1:K, function(k){
      fit_kmeans[[k]]$size[2]/nrow(x[[k]])
    })

    mu1_0 <- sapply(1:K, function(k){
      fit_kmeans[[k]]$centers[1,]
    })

    mu2_0 <- sapply(1:K, function(k){
      fit_kmeans[[k]]$centers[2,]
    })

    Sigma_0 <- sapply(1:K, function(k){
      Sigma1_0 <- t(x[[k]][fit_kmeans[[k]]$cluster == 1, ] - matrix(rep(mu1_0[, k], each = sum(fit_kmeans[[k]]$cluster == 1)), ncol = ncol(x[[k]]))) %*%
        (x[[k]][fit_kmeans[[k]]$cluster == 1, ] - matrix(rep(mu1_0[, k], each = sum(fit_kmeans[[k]]$cluster == 1)), ncol = ncol(x[[k]])))
      Sigma2_0 <- t(x[[k]][fit_kmeans[[k]]$cluster == 2, ] - matrix(rep(mu2_0[, k], each = sum(fit_kmeans[[k]]$cluster == 2)), ncol = ncol(x[[k]]))) %*%
        (x[[k]][fit_kmeans[[k]]$cluster == 2, ] - matrix(rep(mu2_0[, k], each = sum(fit_kmeans[[k]]$cluster == 2)), ncol = ncol(x[[k]])))
      (Sigma1_0 + Sigma2_0)/nrow(x[[k]])
    }, simplify = FALSE)

    beta_0 <- sapply(1:K, function(k){
      solve(Sigma_0[[k]]) %*% (mu1_0[, k] - mu2_0[, k])
    })
  } else if (method == "EM") {
    fit_mcluster <- quiet(sapply(1:K, function(k){
      Mclust(x[[k]], G = 2, modelNames = "EEE")
    }, simplify = FALSE))

    w_0 <- sapply(1:K, function(k){
      fit_mcluster[[k]]$parameters$pro[1]
    })

    mu1_0 <- sapply(1:K, function(k){
      fit_mcluster[[k]]$parameters$mean[,1]
    })

    mu2_0 <- sapply(1:K, function(k){
      fit_mcluster[[k]]$parameters$mean[,2]
    })

    Sigma_0 <- sapply(1:K, function(k){
      fit_mcluster[[k]]$parameters$variance$Sigma
    }, simplify = FALSE)


    beta_0 <- sapply(1:K, function(k){
      solve(Sigma_0[[k]]) %*% (mu1_0[, k] - mu2_0[, k])
    })

  }

  return(list(w = w_0, mu1 = mu1_0, mu2 = mu2_0, Sigma = Sigma_0, beta = beta_0))
}
