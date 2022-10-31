#' @title Generate data for simulations.
#'
#' @description Generate data for simulations. All models used in Tian, Y., Weng, H., & Feng, Y. (2022)) are implemented.
#' @export
#' @param K the number of tasks (data sets). Default: 10
#' @param outlier_K the number of outlier tasks. Default: 1
#' @param simulation_no simulation number in Tian, Y., Weng, H., & Feng, Y. (2022)). Can be "MTL-1", "MTL-2". Default = "MTL-1".
#' @param h_w the value of h_w. Default: 0.1
#' @param h_mu the value of h_mu. Default: 1
#' @param n the sample size of each task. Can be either an positive integer or a vector of length \code{K}. If it is an integer, then the sample size of all tasks will be the same and equal to \code{n}. If it is a vector, then the k-th number will be the sample size of the k-th task. Default: 50.
#' @return a list of two sub-lists "data" and "parameter". List "data" contains a list of design matrices \code{x}, a list of hidden labels \code{y}, and a vector of outlier task indices \code{outlier_index}. List "parameter" contains a vector \code{w} of mixture proportions, a matrix \code{mu1} of which each column is the GMM mean of the first cluster of each task, a matrix \code{mu2} of which each column is the GMM mean of the second cluster of each task, a matrix \code{beta} of which each column is the discriminant coefficient in each task, a list \code{Sigma} of covariance matrices for each task.
#' @seealso \code{\link{mtlgmm}}, \code{\link{tlgmm}}, \code{\link{predict_gmm}}, \code{\link{initialize}}, \code{\link{alignment}}, \code{\link{alignment_swap}}, \code{\link{estimation_error}}, \code{\link{misclustering_error}}.
#' @references
#' Tian, Y., Weng, H., & Feng, Y. (2022). Unsupervised Multi-task and Transfer Learning on Gaussian Mixture Models. arXiv preprint arXiv:2209.15224.
#'
#' @examples
#' data_list <- data_generation(K = 5, outlier_K = 1, simulation_no = "MTL-1", h_w = 0.1,
#' h_mu = 1, n = 50)


data_generation <- function(K = 10, outlier_K = 1, simulation_no = c("MTL-1", "MTL-2"), h_w = 0.1, h_mu = 1, n = 50) {
  simulation_no <- match.arg(simulation_no)
  if (simulation_no == "MTL-1") {
    outlier_type <- "parameter"
    generate_type <- "mu"
  } else if (simulation_no == "MTL-2") {
    outlier_type <- "dist"
    generate_type <- "beta_new"
  }


  eps <- 0.5

  if (generate_type == "mu") {
    outlier_index <- sample(K, size = outlier_K)
    w <- 0.5+runif(K, min = -h_w/2, max = h_w/2)
    w[outlier_index] <- 0.3 + runif(outlier_K, min = -0.1, max = 0.1)
    mu1 <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        z <- rnorm(5)
        z <- z/vec_norm(z)
        c(2,0,0,0,0) + h_mu*z
        # c(0.5,0.5,-1,-0.5,1) + h_mu*z
      } else {
        z <- rnorm(5)
        z <- z/vec_norm(z)
        0.1*z
        # c(2,-1,-1,1,2) + rnorm(5)/5
      }
    })

    mu2 <- -mu1

    Sigma <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        # eps <- rnorm(5, sd = 0.3)
        outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))})
      } else {
        # outer(1:5, 1:5, function(x,y){0.8^(I(x!=y)*1)})
        outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))})
      }
    }, simplify = FALSE)

    if (length(n) == 1) {
      n <- rep(n, K)
    }


    beta <- sapply(1:K, function(k){
      solve(Sigma[[k]]) %*% (mu1[, k] - mu2[, k])
    })

    y <- sapply(1:K, function(k){
      sample(1:2, size = n[k], replace = TRUE, prob = c(1-w[k], w[k]))
    }, simplify = FALSE)

    if (outlier_type == "parameter") {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        x
      }, simplify = FALSE)
    } else {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        if (!(k %in% outlier_index)) {
          x
        } else {
          (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) + (y[[k]]-1)*matrix(rt(n[k]*5, df = 4), ncol = 5)
          # x %*% Sigma[[k]] %*% Sigma[[k]] + rnorm(n[k]*5)
        }
      }, simplify = FALSE)
    }
  } else if (generate_type == "beta") {
    outlier_index <- sample(2:K, size = outlier_K)
    w <- 0.5+runif(K, min = -h_w/2, max = h_w/2)
    w[outlier_index] <- 0.3 + runif(outlier_K, min = -0.1, max = 0.1)

    beta <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        z <- rnorm(5)
        z <- z/vec_norm(z)
        # c(0.5,0.5,-1,-0.5,1) + h_mu*z
        c(3,0,0,0,0) + h_mu*z
      } else {
        z <- rnorm(5)
        z <- z/vec_norm(z)
        0.1*z
        # c(2,-1,-1,1,2) + rnorm(5)/5
      }
    })

    Sigma <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        rv <- sample(0:1, size = 1)
        if (rv == 0) {
          outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))})
        } else {
          # outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))})
          eps_vec <- c(0, 0.5, 1, -1, 2)
          outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))}) + eps*(eps_vec%*%t(eps_vec))
        }
      } else {
        # outer(1:5, 1:5, function(x,y){0.8^(I(x!=y)*1)})
        outer(1:5, 1:5, function(x,y){0.2^(abs(x-y))})
      }
    }, simplify = FALSE)

    mu1 <- sapply(1:K, function(k){
      Sigma[[k]]%*%beta[, k]
    })

    mu2 <- -mu1*0

    if (length(n) == 1) {
      n <- rep(n, K)
    }

    y <- sapply(1:K, function(k){
      sample(1:2, size = n[k], replace = TRUE, prob = c(1-w[k], w[k]))
    }, simplify = FALSE)

    if (outlier_type == "parameter") {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        x
      }, simplify = FALSE)
    } else {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        if (!(k %in% outlier_index)) {
          x
        } else {
          (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) + (y[[k]]-1)*matrix(rt(n[k]*5, df = 3), ncol = 5)
          # x %*% Sigma[[k]] %*% Sigma[[k]] + rnorm(n[k]*5)
        }
      }, simplify = FALSE)
    }
  } else if (generate_type == "beta_new") {
    outlier_index <- sample(2:K, size = outlier_K)
    w <- 0.5+runif(K, min = -h_w/2, max = h_w/2)
    w[outlier_index] <- 0.3 + runif(outlier_K, min = -0.1, max = 0.1)



    Sigma <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        if (k == 1){
          outer(1:5, 1:5, function(x,y){0.5^(abs(x-y))})
        } else {
          rv <- sample(0:1, size = 1)
          if (rv == 0) {
            outer(1:5, 1:5, function(x,y){0.5^(abs(x-y))})
          } else {
            hcomp <- function(epsilon) {
              Sigma0 <- outer(1:5, 1:5, function(x,y){0.5^(abs(x-y))})
              Sigma1 <- outer(1:5, 1:5, function(x,y){epsilon^(abs(x-y))})
              Delta <- as.numeric(solve(Sigma1) %*% (Sigma0 - Sigma1) %*% c(2.5,0,0,0,0))
              vec_norm(Delta)-h_mu
            }

            outer(1:5, 1:5, function(x,y){(uniroot(hcomp, interval=c(0.5,0.999))$root)^(abs(x-y))})
            # outer(1:5, 1:5, function(x,y){(rootSolve::multiroot(hcomp, start = 0.5)$root)^(abs(x-y))})
          }
        }
      } else {
        # outer(1:5, 1:5, function(x,y){0.8^(I(x!=y)*1)})
        outer(1:5, 1:5, function(x,y){0.5^(abs(x-y))})
      }
    }, simplify = FALSE)


    beta <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        if (k == 1) {
          c(2.5,0,0,0,0)
        } else {
          Delta <- as.numeric(solve(Sigma[[k]]) %*% (Sigma[[1]] - Sigma[[k]]) %*% c(2.5,0,0,0,0))
          c(2.5,0,0,0,0) + Delta
        }
      } else {
        z <- rnorm(5)
        z <- z/vec_norm(z)
        # 0.1*z
        c(-2.5,-2.5,-2.5,-2.5,-2.5)
      }
    })


    mu1 <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        Sigma[[k]]%*%beta[, k]
      } else {
        Sigma[[k]]%*%beta[, k]/2
      }
    })

    # mu2 <- -mu1
    mu2 <- sapply(1:K, function(k){
      if (!(k %in% outlier_index)) {
        numeric(5)
      } else {
        -Sigma[[k]]%*%beta[, k]/2
      }
    })

    # mu2 <- -mu1*0

    if (length(n) == 1) {
      n <- rep(n, K)
    }

    y <- sapply(1:K, function(k){
      sample(1:2, size = n[k], replace = TRUE, prob = c(1-w[k], w[k]))
    }, simplify = FALSE)

    if (outlier_type == "parameter") {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        x
      }, simplify = FALSE)
    } else {
      x <- sapply(1:K, function(k){
        x <- (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) +
          (y[[k]]-1)*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu2[, k])
        if (!(k %in% outlier_index)) {
          x
        } else {
          # matrix(rt(n[k]*5, df = 3), ncol = 5)
          (1-(y[[k]]-1))*t(t(matrix(rnorm(n[k]*5), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]) + (y[[k]]-1)*matrix(rt(n[k]*5, df = 4), ncol = 5)
          # (1-(y[[k]]-1))*t(t(matrix(runif(n[k]*5, min = -1, max = 1), ncol = 5) %*% chol(Sigma[[k]])) + mu1[, k]/2) + (y[[k]]-1)*t(t(matrix(runif(n[k]*5, min = -1, max = 1), ncol = 5) %*% chol(Sigma[[k]])) - mu1[, k]/2)
        }
      }, simplify = FALSE)
    }
  }



  return(list(data = list(x = x, y = y, outlier_index = outlier_index), parameter = list(w = w, mu1 = mu1, mu2 = mu2, beta = beta, Sigma = Sigma)))

}
