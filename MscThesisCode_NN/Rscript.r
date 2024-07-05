library(onlineforecast)
library(quantreg)
library(readr)

X_full <- read_csv("X_for_residuals_DK1_2024-06-03.csv", col_names = FALSE)

y <- read_csv("Y_for_residuals_DK1_2024-06-03.csv", col_names = "y")

X_full <- X_full[1:500,c(1,2,4,5,10,12, 15, 17,20,23,25,27, 30,32,35,37,40,42,45,48,49)]
data <- cbind(X_full, y[1:500,1])
predictor_cols <- colnames(X_full)

formula_string <- paste("y ~", paste(predictor_cols, collapse = " + "))
formula <- as.formula(formula_string)

rq_fit <- rq(formula, tau = 0.5, data = data )

write.csv(rq_fit$coefficients, "rq_fit_coefficients.csv")
write.csv(rq_fit$residuals, "rq_fit_residuals.csv")
