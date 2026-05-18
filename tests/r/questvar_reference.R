args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("missing operation")
}

read_matrix <- function(path) {
  as.matrix(read.csv(path, header = FALSE, na.strings = c("NA", "NaN", "nan")))
}

read_vector <- function(path) {
  as.numeric(read.csv(path, header = FALSE, na.strings = c("NA", "NaN", "nan"))[[1]])
}

adjust_p <- function(p, method, n_tests = NULL) {
  if (method %in% c("none", "None", "NA", "NULL")) {
    return(p)
  }

  if (method %in% c("fdr", "fdr_bh")) {
    return(stats::p.adjust(p, method = "BH", n = if (is.null(n_tests)) length(p) else n_tests))
  }

  if (method %in% c("bonferroni", "holm", "hochberg", "BY")) {
    return(stats::p.adjust(p, method = method, n = if (is.null(n_tests)) length(p) else n_tests))
  }

  if (method == "qvalue") {
    if (!is.null(n_tests) && n_tests != length(p)) {
      stop("qvalue reference does not support n_tests override")
    }
    return(qvalue::qvalue(p = p)$qvalues)
  }

  stop(sprintf("unsupported correction method: %s", method))
}

run_ttest_tost <- function(workdir, paired_mode, eq_thr, df_thr, p_thr, correction) {
  s1 <- read_matrix(file.path(workdir, "s1.csv"))
  s2 <- read_matrix(file.path(workdir, "s2.csv"))
  paired <- identical(paired_mode, "paired")

  n_rows <- nrow(s1)
  log2fc <- numeric(n_rows)
  df_p <- numeric(n_rows)
  eq_lo_p <- numeric(n_rows)
  eq_up_p <- numeric(n_rows)

  for (i in seq_len(n_rows)) {
    x <- as.numeric(s1[i, ])
    y <- as.numeric(s2[i, ])

    tt <- stats::t.test(x, y, paired = paired, var.equal = FALSE, alternative = "two.sided")
    tost <- TOSTER::t_TOST(
      x = x,
      y = y,
      paired = paired,
      var.equal = FALSE,
      eqb = eq_thr,
      eqbound_type = "raw",
      alpha = p_thr
    )

    tab <- tost$TOST
    log2fc[i] <- mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE)
    df_p[i] <- unname(tt$p.value)
    eq_lo_p[i] <- unname(tab[2, "p.value"])
    eq_up_p[i] <- unname(tab[3, "p.value"])
  }

  eq_p <- pmax(eq_lo_p, eq_up_p)
  df_adjp <- adjust_p(df_p, correction)
  eq_lo_adjp <- adjust_p(eq_lo_p, correction)
  eq_up_adjp <- adjust_p(eq_up_p, correction)
  eq_adjp <- adjust_p(eq_p, correction)
  comb_p <- ifelse(abs(log2fc) < eq_thr, eq_p, df_p)
  comb_adjp <- adjust_p(comb_p, correction)
  status <- ifelse(
    eq_adjp < p_thr & abs(log2fc) < eq_thr,
    1,
    ifelse(df_adjp < p_thr & abs(log2fc) > df_thr, -1, 0)
  )

  result <- data.frame(
    log2fc = log2fc,
    df_p = df_p,
    df_adjp = df_adjp,
    eq_lo_p = eq_lo_p,
    eq_lo_adjp = eq_lo_adjp,
    eq_up_p = eq_up_p,
    eq_up_adjp = eq_up_adjp,
    eq_p = eq_p,
    eq_adjp = eq_adjp,
    comb_p = comb_p,
    comb_adjp = comb_adjp,
    status = status
  )
  write.csv(result, file.path(workdir, "result.csv"), row.names = FALSE, quote = FALSE)
}

run_p_adjust <- function(workdir, method, n_tests_raw) {
  p <- read_vector(file.path(workdir, "p.csv"))
  n_tests <- if (identical(n_tests_raw, "default")) NULL else as.integer(n_tests_raw)
  adjusted <- adjust_p(p, method, n_tests)
  write.csv(data.frame(adjusted = adjusted), file.path(workdir, "result.csv"), row.names = FALSE, quote = FALSE)
}

operation <- args[1]

suppressPackageStartupMessages({
  library(TOSTER)
  library(qvalue)
})

if (operation == "ttest_tost") {
  if (length(args) != 7) {
    stop("ttest_tost requires: workdir paired_mode eq_thr df_thr p_thr correction")
  }
  run_ttest_tost(
    workdir = args[2],
    paired_mode = args[3],
    eq_thr = as.numeric(args[4]),
    df_thr = as.numeric(args[5]),
    p_thr = as.numeric(args[6]),
    correction = args[7]
  )
} else if (operation == "p_adjust") {
  if (length(args) != 4) {
    stop("p_adjust requires: workdir method n_tests")
  }
  run_p_adjust(args[2], args[3], args[4])
} else {
  stop(sprintf("unknown operation: %s", operation))
}