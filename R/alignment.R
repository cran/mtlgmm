#' @title Align the initializations.
#' @description Align the initializations. This function implements the two alignment algorithms (Algorithms 2 and 3) in Tian, Y., Weng, H., & Feng, Y. (2022). This function is mainly for people to align the single-task initializations manually. The alignment procedure has been automatically implemented in function \code{mtlgmm} and \code{tlgmm}. So there is no need to call this function when fitting MTL-GMM or TL-GMM.
#' @export
#' @param mu1 the initializations for mu1 of all tasks. Should be a matrix of which each column is a mu1 estimate of a task.
#' @param mu2 the initializations for mu2 of all tasks. Should be a matrix of which each column is a mu2 estimate of a task.
#' @param method alignment method. Can be either "exhaustive" (Algorithm 2 in Tian, Y., Weng, H., & Feng, Y. (2022)) or "greedy" (Algorithm 3 in Tian, Y., Weng, H., & Feng, Y. (2022)). Default: "exhaustive"
#' @return the index of two clusters to become well-aligned, i.e. the "r_k" in Section 2.4.2 of Tian, Y., Weng, H., & Feng, Y. (2022). The output can be passed to function \code{\link{alignment_swap}} to obtain the well-aligned intializations.
#' @note For examples, see part "fit signle-task GMMs" of examples in function \code{\link{mtlgmm}}.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{data_generation}}, \code{\link{initialize}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'

alignment <- function(mu1, mu2, method = c("exhaustive", "greedy")) {
  method <- match.arg(method)

  if (method == "exhaustive") {
    L1.table <- as.matrix(expand.grid(rep(list(0:1), ncol(mu1))))
    L2.table <- 1 - L1.table
    L1.table <- L1.table + 1
    L2.table <- L2.table + 1
    score.list <- sapply(1:nrow(L1.table), function(i){
      score(L1.table[i, ], L2.table[i, ], mu1, mu2)
    })
    return(list(L1 = as.numeric(L1.table[which.min(score.list), ]), L2 = as.numeric(L2.table[which.min(score.list), ])))
  } else if (method == "greedy") {
    L1 <- rep(1, ncol(mu1))
    L2 <- rep(2, ncol(mu1))
    for (k in 1:length(L1)) {
      L1.new <- L1
      L2.new <- L2
      L1.new[k] <- L2[k]
      L2.new[k] <- L1[k]
      if (score(L1.new, L2.new, mu1, mu2) < score(L1, L2, mu1, mu2)) {
        L1 <- L1.new
        L2 <- L2.new
      }
    }
    return(list(L1 = L1, L2 = L2))
  }

}
