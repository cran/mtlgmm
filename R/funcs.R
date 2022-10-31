vec_norm <- function(x) {
  sqrt(sum(x^2))
}

col_norm <- function(x) {
  max(sapply(1:ncol(x), function(k){
    vec_norm(x[, k])
  }))
}

vec_max_norm <- function(x) {
  max(abs(x))
}

norm_vec <- function(x) {
  sqrt(sum(x^2))
}

score <- function(L1, L2, mu1, mu2) {
  mu_0 <- list(mu1, mu2)
  sum(sapply(1:length(L1), function(i){
    sapply(1:length(L1), function(j){
      vec_norm(mu_0[[L1[i]]][, i] - mu_0[[L1[j]]][, j])
    })
  })) +
    sum(sapply(1:length(L2), function(i){
      sapply(1:length(L2), function(j){
        vec_norm(mu_0[[L2[i]]][, i] - mu_0[[L2[j]]][, j])
      })
    }))
}


quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}


