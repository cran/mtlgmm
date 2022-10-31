#' @title Caluclate the estimation error of GMM parameters under the MTL setting (the worst performance among all tasks).
#'
#' @description Caluclate the estimation error of GMM parameters under the MTL setting (the worst performance among all tasks). Euclidean norms are used.
#' @export
#' @param estimated_value estimate of GMM parameters. The form of input depends on the parameter \code{parameter}.
#' @param true_value true values of GMM parameters. The form of input depends on the parameter \code{parameter}.
#' @param parameter which parameter to calculate the estimation error for. Can be "w", "mu", "beta", or "Sigma".
#' \itemize{
#' \item w: the Gaussian mixture proportions. Both \code{estimated_value} and \code{true_value} require an input of a K-dimensional vector, where K is the number of tasks. Each element in the vector is an "w" (estimate or true value) for each task.
#' \item mu: Gaussian mean parameters. Both \code{estimated_value} and \code{true_value} require an input of a list of two p-by-K matrices, where p is the dimension of Gaussian distribution and K is the number of tasks. Each column of the matrix is a "mu1" or "mu2" (estimate or true value) for each task.
#' \item beta: discriminant coefficients. Both \code{estimated_value} and \code{true_value} require an input of a p-by-K matrix, where p is the dimension of Gaussian distribution and K is the number of tasks. Each column of the matrix is a "beta" (estimate or true value) for each task.
#' \item Sigma: Gaussian covariance matrices. Both \code{estimated_value} and \code{true_value} require an input of a list of K p-by-p matrices, where p is the dimension of Gaussian distribution and K is the number of tasks. Each matrix in the list is a "Sigma" (estimate or true value) for each task.
#' }
#' @return the largest estimation error among all tasks.
#' @note For examples, see examples in function \code{\link{mtlgmm}}.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'

estimation_error <- function(estimated_value, true_value, parameter = c("w", "mu", "beta", "Sigma")) {
  if (parameter == "w") {
    return(min(vec_max_norm(estimated_value-true_value), vec_max_norm(1-estimated_value-true_value)))
  } else if (parameter == "mu") {
    alm1 <- max(col_norm(estimated_value[[1]]-true_value[[1]]), col_norm(estimated_value[[2]]-true_value[[2]]))
    alm2 <- max(col_norm(estimated_value[[1]]-true_value[[2]]), col_norm(estimated_value[[2]]-true_value[[1]]))
    return(min(alm1, alm2))
  } else if (parameter == "beta") {
    alm1 <- col_norm(estimated_value-true_value)
    alm2 <- col_norm(estimated_value+true_value)
    return(min(alm1, alm2))
  } else if (parameter == "Sigma") {
    err <- max(sapply(1:length(estimated_value), function(k){
      norm(estimated_value[[k]]-true_value[[k]], "2")
    }))
    return(max(err))
  }
}
