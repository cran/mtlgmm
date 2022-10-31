#' @title Complete the alignment of initializations based on the output of function \code{\link{alignment_swap}}.
#'
#' @description Complete the alignment of initializations based on the output of function \code{\link{alignment_swap}}. This function is mainly for people to align the single-task initializations manually. The alignment procedure has been automatically implemented in function \code{mtlgmm} and \code{tlgmm}. So there is no need to call this function when fitting MTL-GMM or TL-GMM.
#' @export
#' @param L1 the component "L1" of the output from function \code{\link{alignment_swap}}
#' @param L2 the component "L2" of the output from function \code{\link{alignment_swap}}
#' @param initial_value_list the output from function \code{\link{initialize}}
#' @return A list with the following components (well-aligned).
#' \item{w}{the estimate of mixture proportion in GMMs for each task. Will be a vector.}
#' \item{mu1}{the estimate of Gaussian mean in the first cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{mu2}{the estimate of Gaussian mean in the second cluster of GMMs for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{beta}{the estimate of the discriminant coefficient for each task. Will be a matrix, where each column represents the estimate for a task.}
#' \item{Sigma}{the estimate of the common covariance matrix for each task. Will be a list, where each component represents the estimate for a task.}
#' @note For examples, see part "fit signle-task GMMs" of examples in function \code{\link{mtlgmm}}.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'

alignment_swap <- function(L1, L2, initial_value_list) {
  for (k in 1:length(L1)) {
    if (L1[k] == 2) {
      # adjust mu1 and mu2
      mu_cur <- initial_value_list$mu1[, k]
      initial_value_list$mu1[, k] <- initial_value_list$mu2[, k]
      initial_value_list$mu2[, k] <- mu_cur

      # adjust w
      initial_value_list$w[k] <- 1 - initial_value_list$w[k]

      # adjust beta
      initial_value_list$beta[, k] <- - initial_value_list$beta[, k]
    }
  }

  return(initial_value_list)
}
