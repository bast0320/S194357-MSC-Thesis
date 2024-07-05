import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ou_log_likelihood(params, data, dt):
    """Calculate the negative log likelihood of the Ornstein-Uhlenbeck process.
    
    Parameters:
    - params: Tuple containing the parameters (lambda, sigma).
    - data: The observed data.
    - dt: Time increment between observations.
    
    Returns:
    - The negative log likelihood.
    """
    lam, sigma = params
    n = len(data)
    X = np.array(data)
    
    # Calculate the mean and variance of X(t) given X(t-1)
    mean_X = X[:-1] * np.exp(-lam * dt)
    var_X = (sigma ** 2) * (1 - np.exp(-2 * lam * dt)) / (2 * lam)
    
    # Log likelihood for the normal distribution
    ll = np.sum(norm.logpdf(X[1:], loc=mean_X, scale=np.sqrt(var_X)))
    return -ll

def estimate_ou_parameters(data, dt):
    """Estimate the parameters of the Ornstein-Uhlenbeck process.
    
    Parameters:
    - data: The observed data.
    - dt: Time increment between observations.
    
    Returns:
    - The estimated parameters (lambda, sigma).
    """
    initial_guess = [0.5, 100]
    bounds = [(0, None), (0.001, None)]
    
    result = minimize(ou_log_likelihood, initial_guess, args=(data, dt), bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed. Try different initial guesses or check the data.")
    
def estimate_ou_parameters_multitry(data, dt):
    """Estimate the parameters of the Ornstein-Uhlenbeck process with multiple initial guesses.
    
    Parameters:
    - data: The observed data.
    - dt: Time increment between observations.
    
    Returns:
    - The estimated parameters (lambda, sigma).
    """
    initial_guesses = [
        [0.5, 100], [1.0, 200], [0.1, 50], [0.8, 150], [0.2, 75],
        [0.4, 120], [1.5, 250], [0.3, 60], [0.9, 180], [0.6, 130],
        [0.7, 110], [1.2, 220], [0.05, 40], [0.15, 90], [0.35, 140],
        [0.25, 65], [1.8, 300], [0.45, 160], [0.55, 170], [0.75, 190]
        ]
    bounds = [(0, None), (0.001, None)]
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"Trying initial guess {i+1}...")
        result = minimize(ou_log_likelihood, initial_guess, args=(data, dt), bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x
    
    raise ValueError("Optimization failed. Try different initial guesses or check the data.")
    
def simulate_ou_process(lam, sigma, dt, n_steps, initial_value, X):
    X[0] = 0# initial_value
    for t in range(1, n_steps):
        X[t] = X[t-1] * np.exp(-lam * dt) + sigma * np.sqrt((1 - np.exp(-2 * lam * dt)) / (2 * lam)) * np.random.normal()
    return X



def variogram_score_R(x, y, p=0.5):
    """
    Calculate the CRPS score for a given quantile.
    From the paper Energy and AI, Mathias
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0

    # Iterate through all pairs
    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
            score += (np.abs(y[i] - y[j])**p - Ediff)**2

    # Variogram score
    return score


import properscoring as ps
def calculate_crps(actuals, corrected_ensembles):
    crps = ps.crps_ensemble(actuals, corrected_ensembles)
    return np.mean(crps)

def calculate_qss(actuals, taqr_results, quantiles):
    qss_scores = multi_quantile_skill_score(actuals, taqr_results, quantiles)
    return np.mean(qss_scores)

def multi_quantile_skill_score(y_true, y_pred, quantiles):
    """
    Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.

    y_true: This is a 1D numpy array or list of the true observed values.
    y_pred: This is a 2D numpy array or list of lists of the predicted quantile values. The outer dimension should be the same length as y_true, and the inner dimension should be the number of quantiles.
    quantiles: This is a 1D numpy array or list of the quantile levels. It should be the same length as the inner dimension of y_pred.

    Parameters:
    y_true (numpy.array or list): True observed values. 1D array.
    y_pred (numpy.array or list of lists): Predicted quantile values. 2D array.
    quantiles (numpy.array or list): Quantile levels, between 0 and 1. 1D array.

    Returns:
    numpy.array: The QSS for each quantile forecast. 1D array.
    """
    # Convert y_pred to a numpy array
    y_pred = np.array(y_pred)

    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.T

    # print("shape y_pred: ", y_pred.shape)   
    # print("shape y_true: ", y_true.shape)
    # assert len(y_true) == len(y_pred[0]), "y_true and y_pred must be the same length" TODO: Should be kept active,... 
    assert all(0 <= q <= 1 for q in quantiles), "All quantiles must be between 0 and 1"
    assert len(quantiles) == len(y_pred), "Number of quantiles must match inner dimension of y_pred"

    N = len(y_true)
    scores = np.zeros(len(quantiles))

    # for y, y_hats in zip(y_true, y_pred):
    #     for i, (y_hat, q) in enumerate(zip(y_hats, quantiles)):
    #         if y >= y_hat:
    #             scores[i] += q * np.abs(y - y_hat)
    #         else:
    #             scores[i] += (1 - q) * np.abs(y - y_hat)

    # for y, y_hats in zip(y_true, y_pred):
    #     for i, (y_hat, q) in enumerate(zip(y_hats, quantiles)):
    #         E = (y - y_hat)

    #         if E > 0:
    #             scores[i] += q * E
    #         elif E < 0:
    #             scores[i] += (1 - q) * (-E)

    for i, q in enumerate(quantiles):
        E = y_true - y_pred[i]
        scores[i] = np.sum(np.where(E > 0, q * E, (1 - q) * -E))

    # for y, y_hats in zip(y_true, y_pred):
    #     for i, (y_hat, q) in enumerate(zip(y_hats, quantiles)):
    #         E = (y - y_hat)

    #         l = 0
    #         for j, e in enumerate(E):
    #             if e > 0:
    #                 l += q * e
    #             else:
    #                 l += (1 - q) * (-e)

    #         scores[i] = l

    return scores / N



from pipeline_start_to_finish import estimate_ou_parameters, simulate_ou_process, variogram_score_R, calculate_crps, calculate_qss

def SDE_subprocess(corrected_ensembles, actuals, quantiles, n_steps = 100, number_of_repeats = 100, print_ = False, plot_ = False, plotname = "TestPlot", latex_ = False):
    # assuming here that ensembles are and actuals are the test set from 75-100% of the data

    n_steps = int(len(actuals)*(15/25)) # 10 out of 25% , " which is the train set for the SDE"
    dt = 1
    residuals_dict = {}
    estimated_params_dict = {}
    percentile_value_dict = {}
    simulated_data_dict = {}
    for q in quantiles:
        percentile_value = corrected_ensembles.quantile(q, axis=1)
        percentile_value_dict[q] = percentile_value
        residuals = actuals.loc[percentile_value.index] - percentile_value.values
        residuals_dict[q] = residuals
        estimated_params_dict[q] = estimate_ou_parameters(residuals[:n_steps], dt)
        lam, sigma = estimated_params_dict[q]
        sigma = np.sqrt(sigma)

        simulated_data_dict[q] = simulate_ou_process(lam, sigma, dt, n_steps = len(actuals)-n_steps, initial_value = 0, X = residuals[n_steps:])

    median = percentile_value_dict[quantiles[0]][n_steps:]
    sde_corrected_ensembles = pd.DataFrame()
    variogram_score_sde_corrected_list = []
    variogram_score = []
    crps_sde_corrected_list = []
    crps_score = []
    qss_sde_corrected_list = []
    qss_score = []

    for i in range(number_of_repeats):
        for q in quantiles:
            residuals = residuals_dict[q]
            lam, sigma = estimated_params_dict[q]
            sigma = np.sqrt(sigma)
            simulated_data_dict[q] = simulate_ou_process(lam, sigma, dt, n_steps = len(actuals)-n_steps, initial_value = 0, X = residuals[n_steps:])
            sde_corrected_ensembles[str(q)] = percentile_value_dict[q].values[n_steps:] + (residuals_dict[q][n_steps:] + simulated_data_dict[q])
        
        variogram_score_sde_corrected_list.append(variogram_score_R(sde_corrected_ensembles.values, actuals.loc[median.index].values, p=0.5))
        crps_sde_corrected_list.append(calculate_crps(actuals.loc[median.index].values, sde_corrected_ensembles.values))
        qss_sde_corrected_list.append(calculate_qss(actuals.loc[median.index].values, sde_corrected_ensembles.values, quantiles = quantiles))
    variogram_score.append(variogram_score_R(corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values, actuals.loc[median.index].values, p=0.5))
    crps_score.append(calculate_crps(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values))
    qss_score.append((calculate_qss(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values, quantiles = quantiles)))

    if print_:
        print("Variogram score comparison:")
        print(f"  SDE corrected: {np.mean(np.array(variogram_score_sde_corrected_list))}")
        print(f"  Previous: {float(variogram_score[0])}")
        
        print("\nQSS comparison:")
        print(f"  SDE corrected ensembles: {np.mean(np.array(qss_sde_corrected_list))}")
        print(f"  Previous corrected ensembles: {np.mean(np.array(qss_score))}")

        print("\nCRPS comparison:")
        print(f"  SDE corrected ensembles: {np.mean(np.array(crps_sde_corrected_list))}")
        print(f"  Previous corrected ensembles: {float(crps_score[0])}")

    if latex_:
        # Calculate means
        variogram_score_sde_corrected_mean = np.mean(np.array(variogram_score_sde_corrected_list))
        crps_sde_corrected_mean = np.mean(np.array(crps_sde_corrected_list))
        qss_sde_corrected_mean = np.mean(np.array(qss_sde_corrected_list))

        # Extract previous scores
        variogram_score_prev = float(variogram_score[0])
        crps_score_prev = float(crps_score[0])
        qss_score_prev = np.mean(np.array(qss_score))

        # LaTeX table content
        latex_table = f"""
        \\begin{{table}}[h!]
            \\centering
            \\begin{{tabular}}{{|c|c|c|}}
                \\hline
                & \\textbf{{SDE Corrected}} & \\textbf{{Previous}} \\\\
                \\hline
                \\textbf{{Variogram Score}} & {variogram_score_sde_corrected_mean:.4f} & {variogram_score_prev:.4f} \\\\
                \\hline
                \\textbf{{CRPS}} & {crps_sde_corrected_mean:.4f} & {crps_score_prev:.4f} \\\\
                \\hline
                \\textbf{{QSS}} & {qss_sde_corrected_mean:.4f} & {qss_score_prev:.4f} \\\\
                \\hline
            \\end{{tabular}}
            \\caption{{Comparison of scores for SDE corrected and previous ensembles {plotname}}}
            \\label{{table:scores_comparison}}
        \\end{{table}}
        """

        # Save to .tex file
        with open(f"latex_output/3scores_SDE_comparison_{plotname}.tex", "w") as file:
            file.write(latex_table)



    if number_of_repeats > 10 and plot_ == True:

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: Variogram scores
        sns.histplot(data=pd.DataFrame(np.array(variogram_score_sde_corrected_list)), kde=True, label="SDE corr. Variogram scores", bins=30, ax=axes[0])
        axes[0].axvline(np.mean(np.array(variogram_score_sde_corrected_list)), color="black", label="SDE corr. Mean Variogram Score")
        axes[0].axvline(variogram_score, color="red", label="Prev. Variogram Score")
        axes[0].set_xlabel("Variogram score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Variogram score for SDE corrected ensembles")
        axes[0].legend()

        # Subplot 2: QSS scores
        sns.histplot(data=pd.DataFrame(np.mean(np.array(qss_sde_corrected_list), axis = 1) ) , kde=True, label="SDE corr. QSS scores", bins=20, ax=axes[1])
        axes[1].axvline(np.mean(np.mean(np.array(qss_sde_corrected_list), axis = 1)), color="black", label="SDE corr. Mean QSS Score")
        axes[1].axvline(np.mean(np.array(qss_score)), label="Prev. QSS Score", color="red")
        axes[1].set_xlabel("QSS score")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("QSS score for SDE corrected ensembles")
        axes[1].legend()

        # Subplot 3: CRPS scores
        sns.histplot(data=pd.DataFrame(np.array(crps_sde_corrected_list)), kde=True, label="SDE corr. CRPS scores", bins=30, ax=axes[2])
        axes[2].axvline(np.mean(np.array(crps_sde_corrected_list)), color="black", label="SDE corr. Mean CRPS Score")
        axes[2].axvline(crps_score, label="Prev. CRPS Score", color="red")
        axes[2].set_xlabel("CRPS score")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("CRPS score for SDE corrected ensembles")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(f"figures/{plotname}.pdf")
        plt.show()

    return variogram_score_sde_corrected_list, variogram_score , crps_sde_corrected_list, crps_score, qss_sde_corrected_list, qss_score 



def SDE_subprocess_skill_score(corrected_ensembles, actuals, quantiles, n_steps=100, number_of_repeats=100, print_=False, plot_=False, plotname="TestPlot", latex_=False, plot_pacf_acf=False):
    
    '''
    This is the much more final version that can output acf and pacf plots, alongside plots and latex tables. Calculates the Skill score for CRPS and QSS as well as the Variogram score.
    
    '''
    
    # assuming here that ensembles are and actuals are the test set from 75-100% of the data
    # import pandas as pd
    # import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_pacf
    n_steps = int(len(actuals)*(15/25)) # 10 out of 25% , " which is the train set for the SDE"
    dt = 1
    residuals_dict = {}
    estimated_params_dict = {}
    percentile_value_dict = {}
    simulated_data_dict = {}
    for q in quantiles:
        percentile_value = corrected_ensembles.quantile(q, axis=1)
        percentile_value_dict[q] = percentile_value
        residuals = actuals.loc[percentile_value.index] - percentile_value.values
        residuals_dict[q] = residuals
        estimated_params_dict[q] = estimate_ou_parameters(residuals[:n_steps], dt)
        lam, sigma = estimated_params_dict[q]
        sigma = np.sqrt(sigma)

        simulated_data_dict[q] = simulate_ou_process(lam, sigma, dt, n_steps=len(actuals)-n_steps, initial_value=0, X=residuals[n_steps:])

    median = percentile_value_dict[quantiles[0]][n_steps:]
    sde_corrected_ensembles = pd.DataFrame()
    variogram_score_sde_corrected_list = []
    crps_sde_corrected_list = []
    qss_sde_corrected_list = []

    for i in range(number_of_repeats):
        for q in quantiles:
            residuals = residuals_dict[q]
            lam, sigma = estimated_params_dict[q]
            sigma = np.sqrt(sigma)
            simulated_data_dict[q] = simulate_ou_process(lam, sigma, dt, n_steps=len(actuals)-n_steps, initial_value=0, X=residuals[n_steps:])
            sde_corrected_ensembles[str(q)] = percentile_value_dict[q].values[n_steps:] + simulated_data_dict[q] # (residuals_dict[q][n_steps:] + simulated_data_dict[q])
        
        variogram_score_sde_corrected_list.append(variogram_score_R(sde_corrected_ensembles.values, actuals.loc[median.index].values, p=0.5))
        crps_sde_corrected_list.append(calculate_crps(actuals.loc[median.index].values, sde_corrected_ensembles.values))
        qss_sde_corrected_list.append(calculate_qss(actuals.loc[median.index].values, sde_corrected_ensembles.values, quantiles=quantiles))
    
    variogram_score = variogram_score_R(corrected_ensembles.loc[median.index].quantile(quantiles, axis=1).T.values, actuals.loc[median.index].values, p=0.5)
    crps_score = calculate_crps(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis=1).T.values)
    qss_score = calculate_qss(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis=1).T.values, quantiles=quantiles)

    variogram_score_sde_corrected_mean = np.mean(np.array(variogram_score_sde_corrected_list))
    crps_sde_corrected_mean = np.mean(np.array(crps_sde_corrected_list))
    qss_sde_corrected_mean = np.mean(np.array(qss_sde_corrected_list))

    variogram_score_prev = float(variogram_score)
    crps_score_prev = float(crps_score)
    qss_score_prev = float(np.mean(np.array(qss_score)))


    # plot the acf of the residuals
    # for q in quantiles:
    if plot_pacf_acf:

        residuals = residuals_dict[0.5]
        percentile_value = sde_corrected_ensembles.quantile(0.5, axis=1)
        residuals_sde = actuals.loc[percentile_value.index] - percentile_value.values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot autocorrelation of actuals
        pd.plotting.autocorrelation_plot(actuals.loc[percentile_value.index], ax=ax1, color="black", linewidth=2).set_xlim([0, 100])
        line3, = ax1.plot([], [], color="black", label='Actuals', linewidth=2)

        pd.plotting.autocorrelation_plot(corrected_ensembles.quantile(0.5, axis=1).T.values, ax=ax1, color="orange", linestyle="--", linewidth=2).set_xlim([0, 100])
        line4, = ax1.plot([], [], color="orange", label='Corrected Ensembles', linewidth=1.5, linestyle="--")

        pd.plotting.autocorrelation_plot(sde_corrected_ensembles.loc[percentile_value.index].quantile(0.5, axis=1), ax=ax1, color="green", linestyle="--", linewidth=2).set_xlim([0, 100])
        line5, = ax1.plot([], [], color="green", label='SDE Corrected Ensembles', linewidth=1.5, linestyle="--")

        ax1.set_title('Autocorrelation Plots')
        ax1.legend()

        # Plot partial autocorrelation of actuals
        plot_pacf(actuals.loc[percentile_value.index], ax=ax2, color="black", linewidth=2, lags=10)
        line3_pacf, = ax2.plot([], [], color="black", label='Actuals', linewidth=2)

        plot_pacf(corrected_ensembles.quantile(0.5, axis=1).T.values, ax=ax2, color="orange", linestyle="--", linewidth=2, lags=10)
        line4_pacf, = ax2.plot([], [], color="orange", label='Corrected Ensembles', linewidth=1.5, linestyle="--")

        plot_pacf(sde_corrected_ensembles.loc[percentile_value.index].quantile(0.5, axis=1), ax=ax2, color="green", linestyle="--", linewidth=2, lags=10)
        line5_pacf, = ax2.plot([], [], color="green", label='SDE Corrected Ensembles', linewidth=1.5, linestyle="--")

        ax2.set_title('Partial Autocorrelation Plots')
        ax2.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"figures/ACF_PACF_25062024_combined_{plotname}.pdf")
        plt.show()

    # print(qss_score_prev, "qss_score_prev")

    variogram_skill_score_list =  (np.array(variogram_score_sde_corrected_list) / variogram_score_prev)
    crps_skill_score_list =  (np.array(crps_sde_corrected_list) / crps_score_prev)
    qss_skill_score_list =  ((np.array(qss_sde_corrected_list)) / qss_score_prev)

    variogram_skill_score_mean =  (variogram_score_sde_corrected_mean / variogram_score_prev)
    crps_skill_score_mean =  (crps_sde_corrected_mean / crps_score_prev)
    qss_skill_score_mean =  (qss_sde_corrected_mean / qss_score_prev)

    if print_:
        print("Skill Scores comparison:")
        print(f"  Variogram Skill Score: {variogram_skill_score_mean:.4f}")
        print(f"  CRPS Skill Score: {crps_skill_score_mean:.4f}")
        print(f"  QSS Skill Score: {qss_skill_score_mean:.4f}")

    if latex_:
        latex_table = f"""
        \\begin{{table}}[h!]
            \\centering
            \\begin{{tabular}}{{|c|c|}}
                \\hline
                & \\textbf{{Skill Score}}  \\\\
                \\hline
                \\textbf{{Variogram Skill Score}} & {variogram_skill_score_mean:.4f}  \\\\
                \\hline
                \\textbf{{CRPS Skill Score}} & {crps_skill_score_mean:.4f}  \\\\
                \\hline
                \\textbf{{QSS Skill Score}} & {qss_skill_score_mean:.4f} \\\\
                \\hline
            \\end{{tabular}}
            \\caption{{Comparison of skill scores for SDE corrected and previous ensembles {plotname}}}
            \\label{{table:skill_scores_comparison}}
        \\end{{table}}
        """

        # Save to .tex file
        with open(f"latex_output/3skill_scores_SDE_comparison_v2_{plotname}.tex", "w") as file:
            file.write(latex_table)

    if number_of_repeats > 10 and plot_:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: Variogram Skill Scores
        sns.histplot(data=pd.DataFrame(variogram_skill_score_list), kde=True, label="SDE corr. Variogram Skill Scores", bins=30, ax=axes[0])
        axes[0].axvline(variogram_skill_score_mean, color="black", label="SDE corr. Mean Variogram Skill Score")
        axes[0].axvline(1, color="red", label="Skill Score Baseline (1)")
        axes[0].set_xlabel("Variogram Skill Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Variogram Skill Score for SDE corrected ensembles")
        axes[0].legend()

        # Subplot 2: QSS Skill Scores
        sns.histplot(data=pd.DataFrame(np.mean(np.array(qss_skill_score_list), axis = 1) ), kde=True, label="SDE corr. QSS Skill Scores", bins=20, ax=axes[1])
        axes[1].axvline(qss_skill_score_mean, color="black", label="SDE corr. Mean QSS Skill Score")
        axes[1].axvline(1, label="Skill Score Baseline (1)", color="red")
        axes[1].set_xlabel("QSS Skill Score")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("QSS Skill Score for SDE corrected ensembles")
        axes[1].legend()

        # Subplot 3: CRPS Skill Scores
        sns.histplot(data=pd.DataFrame(crps_skill_score_list), kde=True, label="SDE corr. CRPS Skill Scores", bins=30, ax=axes[2])
        axes[2].axvline(crps_skill_score_mean, color="black", label="SDE corr. Mean CRPS Skill Score")
        axes[2].axvline(1, label="Skill Score Baseline (1)", color="red")
        axes[2].set_xlabel("CRPS Skill Score")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("CRPS Skill Score for SDE corrected ensembles")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(f"figures/{plotname}_v2.pdf")
        plt.show()

    return variogram_skill_score_list, crps_skill_score_list, qss_skill_score_list


def save_optimal_model(corrected_ensembles_, actuals_, plotname_, quantiles = [0.1, 0.5, 0.9]):
    iter = 0
    
    variogram_skill_score_list, crps_skill_score_list, qss_skill_score_list  = SDE_subprocess_skill_score(corrected_ensembles_, 
                        actuals_, 
                        quantiles, n_steps = 100, number_of_repeats = 100, print_ = True, 
                        plot_ = True, plotname=plotname_,
                        latex_ = True)


    if np.mean(np.array(variogram_skill_score_list)) < 1 and np.mean(np.array(crps_skill_score_list)) < 1 and np.mean((np.array(qss_skill_score_list))) < 1:
        print("SDE corrected ensembles are better than previous ensembles")
    else:
        while np.mean(np.array(variogram_skill_score_list)) > 1 or np.mean(np.array(crps_skill_score_list)) > 1 or np.mean((np.array(qss_skill_score_list))) > 1:
            variogram_skill_score_list, crps_skill_score_list, qss_skill_score_list = SDE_subprocess_skill_score(corrected_ensembles_, 
                            actuals_, 
                            quantiles, n_steps = 100, number_of_repeats = 100, print_ = True, 
                            plot_ = True, plotname=plotname_,
                            latex_ = True)
            iter += 1
            print("Iteration: ", iter)
            if iter > 10:
                break



def save_optimal_model_v2(corrected_ensembles_, actuals_, plotname_):
    iter = 0
    best_skill_score_sum = float('inf')
    best_variogram_skill_score_list = []
    best_crps_skill_score_list = []
    best_qss_skill_score_list = []

    while iter < 10:
        variogram_skill_score_list, crps_skill_score_list, qss_skill_score_list  = SDE_subprocess_skill_score(
            corrected_ensembles_, actuals_, quantiles, n_steps=100, number_of_repeats=100, print_=True, 
            plot_=False, plotname=plotname_, latex_=True
        )

        variogram_skill_score_mean = np.mean(np.array(variogram_skill_score_list))
        crps_skill_score_mean = np.mean(np.array(crps_skill_score_list))
        qss_skill_score_mean = np.mean(np.array(qss_skill_score_list))
        skill_score_sum = variogram_skill_score_mean + crps_skill_score_mean + qss_skill_score_mean

        if skill_score_sum < best_skill_score_sum:
            best_skill_score_sum = skill_score_sum
            best_variogram_skill_score_list = variogram_skill_score_list
            best_crps_skill_score_list = crps_skill_score_list
            best_qss_skill_score_list = qss_skill_score_list

        if variogram_skill_score_mean < 1 and crps_skill_score_mean < 1 and qss_skill_score_mean < 1:
            print("SDE corrected ensembles are better than previous ensembles")
            return best_variogram_skill_score_list, best_crps_skill_score_list, best_qss_skill_score_list

        iter += 1
        print("Iteration: ", iter)

    print("Could not find a model that satisfies the criteria within 10 iterations.")
    print("Returning the best model found based on the sum of mean skill scores.")
    return best_variogram_skill_score_list, best_crps_skill_score_list, best_qss_skill_score_list