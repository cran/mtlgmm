#' @title Calculate the misclustering error given the predicted cluster labels.
#'
#' @description  Calculate the misclustering error given the predicted cluster labels.
#' @export
#' @param y_pred predicted cluster labels
#' @param y_test true cluster labels
#' @param type which type of the misclustering error rate to return. Can be either "max", "all", or "avg". Default: "max".
#' \itemize{
#' \item max: maximum of misclustering error rates on all tasks
#' \item all: a vector of misclustering error rates on each tasks
#' \item avg: average of misclustering error rates on all tasks
#' }
#' @return Depends on \code{type}.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{data_generation}}, \code{\link{predict_gmm}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'
#' @examples
#' set.seed(23, kind = "L'Ecuyer-CMRG")
#' ## Consider a 5-task multi-task learning problem in the setting "MTL-1"
#' data_list <- data_generation(K = 5, outlier_K = 1, simulation_no = "MTL-1", h_w = 0.1,
#' h_mu = 1, n = 100)  # generate the data
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


misclustering_error <- function(y_pred, y_test, type = c("max", "all", "avg")) {
  type <- match.arg(type)
  if (is.list(y_pred) && is.list(y_test)) {
    if (type == "max") {
      max(sapply(1:length(y_pred), function(k){
        min(mean(y_pred[[k]] != y_test[[k]]), mean((3-y_pred[[k]]) != y_test[[k]]))
      }))
    } else if (type == "all") {
      sapply(1:length(y_pred), function(k){
        min(mean(y_pred[[k]] != y_test[[k]]), mean((3-y_pred[[k]]) != y_test[[k]]))
      })
    } else if (type == "avg") {
      n.all <- length(Reduce("c", y_pred))
      wt <- sapply(1:length(y_pred), function(k){
        length(y_pred[[k]])/n.all
      })
      sum(wt*sapply(1:length(y_pred), function(k){
        min(mean(y_pred[[k]] != y_test[[k]]), mean((3-y_pred[[k]]) != y_test[[k]]))
      }))
    }
  } else {
    min(mean(y_pred != y_test), mean((3-y_pred) != y_test))
  }
}
