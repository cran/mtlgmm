#' @title Clustering new observations based on fitted GMM estimators.
#'
#' @description Clustering new observations based on fitted GMM estimators, which is an empirical version of Bayes classifier. See equation (13) in Tian, Y., Weng, H., & Feng, Y. (2022).
#' @export
#' @param w the estimate of mixture proportion in the GMM. Numeric.
#' @param mu1 the estimate of Gaussian mean of the first cluster in the GMM. Should be a vector.
#' @param mu2 the estimate of Gaussian mean of the first cluster in the GMM. Should be a vector.
#' @param beta the estimate of the discriminant coefficient for the GMM. Should be a vector.
#' @param newx design matrix of new observations. Should be a matrix.
#' @return A vector of predicted labels of new observations.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'
#' @examples
#' set.seed(23, kind = "L'Ecuyer-CMRG")
#' ## Consider a 5-task multi-task learning problem in the setting "MTL-1"
#' data_list <- data_generation(K = 5, outlier_K = 1, simulation_no = "MTL-1", h_w = 0.1,
#' h_mu = 1, n = 50)  # generate the data
#' x_train <- sapply(1:length(data_list$data$x), function(k){
#'   data_list$data$x[[k]][1:50,]
#' }, simplify = FALSE)
#' x_test <- sapply(1:length(data_list$data$x), function(k){
#'   data_list$data$x[[k]][-(1:50),]
#' }, simplify = FALSE)
#' y_test <- sapply(1:length(data_list$data$x), function(k){
#'   data_list$data$y[[k]][-(1:50)]
#' }, simplify = FALSE)
#'
#' fit <- mtlgmm(x = x_train, C1_w = 0.05, C1_mu = 0.2, C1_beta = 0.2,
#' C2_w = 0.05, C2_mu = 0.2, C2_beta = 0.2, kappa = 1/3, initial_method = "EM",
#' trim = 0.1, lambda_choice = "fixed", step_size = "lipschitz")
#'
#' y_pred <- sapply(1:length(data_list$data$x), function(i){
#' predict_gmm(w = fit$w[i], mu1 = fit$mu1[, i], mu2 = fit$mu2[, i],
#' beta = fit$beta[, i], newx = x_test[[i]])
#' }, simplify = FALSE)
#' misclustering_error(y_pred[-data_list$data$outlier_index],
#' y_test[-data_list$data$outlier_index], type = "max")

predict_gmm <- function(w, mu1, mu2, beta, newx) {
  as.numeric(2-I(t(t(newx) - as.numeric(mu1+mu2)/2) %*% beta >= log(w/(1-w))))
}
