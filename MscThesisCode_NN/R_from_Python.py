# command = '/usr/local/bin/R'
# arg = '--vanilla'
# path2script = 'Rscript.r'

# import subprocess
# retcode = subprocess.call([command, arg, path2script], shell=False)

# import subprocess

# process = subprocess.Popen(["R", "--vanilla"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

# process.stdin.write(b"library(onlineforecast) \n")
# process.stdin.write(b"library(quantreg) \n")
# process.stdin.write(b"library(readr) \n")
# process.stdin.write(b'X_full <- read_csv("X_for_residuals_DK1_2024-06-03.csv", col_names = FALSE) \n')
# process.stdin.write(b'y <- read_csv("Y_for_residuals_DK1_2024-06-03.csv", col_names = "y") \n')
# process.stdin.write(b'X_full <- X_full[1:500,c(1,2,4,5,10,12, 15, 17,20,23,25,27, 30,32,35,37,40,42,45,48,49)] \n')
# process.stdin.write(b'data <- cbind(X_full, y[1:500,1]) \n')
# process.stdin.write(b'predictor_cols <- colnames(X_full) \n')
# process.stdin.write(b'formula_string <- paste("y ~ 0+", paste(predictor_cols, collapse = " + ")) \n')
# process.stdin.write(b'formula <- as.formula(formula_string) \n')
# process.stdin.write(b'rq_fit <- rq(formula, tau = 0.5, data = data ) \n')
# process.stdin.write(b'write.csv(rq_fit$coefficients, "rq_fit_coefficients.csv") \n')
# process.stdin.write(b'write.csv(rq_fit$residuals, "rq_fit_residuals.csv") \n')

# process.stdin.close()

# output = process.stdout.read()

# print(output.decode())

# process.terminate()


def run_r_script(X_filename, Y_filename, tau):
    import subprocess
    process = subprocess.Popen(["R", "--vanilla"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
    r_script = f"""
    library(onlineforecast) 
    library(quantreg) 
    library(readr) 
    X_full <- read_csv("{X_filename}", col_names = FALSE, show_col_types = FALSE) 
    y <- read_csv("{Y_filename}", col_names = "y", show_col_types = FALSE) 
    X_full <- X_full[1:500,] # [1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]  
    data <- cbind(X_full, y[1:500,1]) 
    predictor_cols <- colnames(X_full) 
    formula_string <- paste("y ~ 0+", paste(predictor_cols, collapse = " + ")) 
    formula <- as.formula(formula_string) 
    rq_fit <- rq(formula, tau = {tau}, data = data ) 
    write.csv(rq_fit$coefficients, "rq_fit_coefficients.csv") 
    write.csv(rq_fit$residuals, "rq_fit_residuals.csv") 
    """
    
    for line in r_script.strip().split('\n'):
        process.stdin.write(line.encode('utf-8') + b"\n")

    process.stdin.close()

    output = process.stdout.read()
    # print(output.decode())

    process.terminate()

# Example usage
# run_r_script("X_for_residuals_DK1_2024-06-03.csv", "Y_for_residuals_DK1_2024-06-03.csv", 0.5)