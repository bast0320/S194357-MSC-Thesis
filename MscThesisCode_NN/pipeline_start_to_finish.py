import numpy as np
import properscoring as ps
import tensorflow as tf
from functions_for_TAQR import one_step_quantile_prediction
import pandas as pd
from multimodel_quantile_loss_network_working_TDS_article_functions import *

# Pipeline:
# - Load the data, based on string input => DATA
# - With the ensembles, true data, we need to train the network on 80% of the data => TRAINED MODEL
# - We then out of sample use the trained model to correct the ensembles => CORRECTED ENSEMBLES
# - We then use the corrected ensembles to run the TAQR algo => TAQR RESULTS for e.g. 5 quantiles
# - We then calculate the QSS, CRPS, and Variogram score for the TAQR results => SCORES
# _ We could consider also just calculating scores for the corrected ensembles...
# - We then save the scores, and the corrected ensembles, and the TAQR results, and the TRAINED MODEL
# - We then repeat this for all data sources, and then we can start to look at the results

def load_data(data_area, type, subset_size = 18720):


    # TODO optimize this loading, but whatever for now...

    if data_area == "DK1" and type == "OnshoreWindPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK1_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1_fixed.pkl")
        Y = actuals["OnshoreWindPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy() # 18720 is obv a quick fix! TODO: Fix this
        X = ensembles.iloc[:subset_size].to_numpy()
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)

    elif data_area == "DK1" and type == "SolarPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK1_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_solar_DK1_fixed_v2.pkl")

        # find the common index
        set1 = actuals["SolarPower"].index
        set2 = ensembles.index
        common_index = set1.intersection(set2)
        actuals = actuals.loc[common_index]
        ensembles = ensembles.loc[common_index]
        Y = actuals["SolarPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy()
        X = ensembles.iloc[:subset_size].to_numpy()
        # remove 0s
        X = X[Y >= 0.9,:]
        Y = Y[(Y >= 0.9)]
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)

    elif data_area == "DK2" and type == "OnshoreWindPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK2_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK2_fixed.pkl")
        Y = actuals["OnshoreWindPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy()
        X = ensembles.iloc[:subset_size].to_numpy()
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)

    elif data_area == "DK2" and type == "SolarPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK2_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_solar_DK2_fixed_v2.pkl")

        set1 = actuals["SolarPower"].index
        set2 = ensembles.index
        common_index = set1.intersection(set2)
        actuals = actuals.loc[common_index]
        ensembles = ensembles.loc[common_index]

        Y = actuals["SolarPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy()
        X = ensembles.iloc[:subset_size].to_numpy()
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)

    elif data_area == "DK1" and type == "OffshoreWindPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK1_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK1_fixed.pkl")
        Y = actuals["OffshoreWindPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy()
        X = ensembles.iloc[:subset_size].to_numpy()
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)

    elif data_area == "DK2" and type == "OffshoreWindPower":
        actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK2_fixed.pkl")
        ensembles = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK2_fixed.pkl")
        Y = actuals["OffshoreWindPower"][:subset_size]
        idx = Y.index
        Y = Y.to_numpy()
        X = ensembles.iloc[:subset_size].to_numpy()
        X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)



    # Add more conditions for different data sources "elif"
    else:
        raise ValueError(f"Unsupported data source: {data_area} and type: {type}")
    return Y, X, X_y, idx

def train_ffnn(ensembles, actuals, X_y, epochs_input, x_val, y_val, data_name):
    # Preprocess the data

    # set nans to 0
    ensembles[np.isnan(ensembles)] = 0
    actuals[np.isnan(actuals)] = 0
    X_y[np.isnan(X_y)] = 0
    x_val[np.isnan(x_val)] = 0
    y_val[np.isnan(y_val)] = 0

    max_val = np.max(X_y)
    print("max_val: ", max_val)

    x = tf.convert_to_tensor(ensembles/max_val)
    y = tf.convert_to_tensor(actuals/max_val)
    X_y = tf.convert_to_tensor(X_y/max_val)
    x_val = tf.convert_to_tensor(x_val/max_val)
    y_val = tf.convert_to_tensor(y_val/max_val)

    # print("Min and max of x: ", np.min(x.numpy()), np.max(x.numpy()))
    # print("Min and max of y: ", np.min(y.numpy()), np.max(y.numpy()))
    # print("Min and max of x_val: ", np.min(x_val.numpy()), np.max(x_val.numpy()))
    # print("Min and max of y_val: ", np.min(y_val.numpy()), np.max(y_val.numpy()))

    # Define the quantiles
    quantiles = np.linspace(0.01, 0.99, 50)

    # Train the model
    model = train_model(quantiles=quantiles, epochs=epochs_input, lr=1e-3, batch_size=50, x=x, y=X_y, x_val = x_val, y_val = y_val, data_name=data_name)
    return model

def train_lstm(ensembles, actuals, X_y, epochs_input):
    # Preprocess the data
    x = tf.convert_to_tensor(ensembles)
    y = tf.convert_to_tensor(actuals)
    X_y = tf.convert_to_tensor(X_y)

    # Define the quantiles
    quantiles = np.linspace(0.01, 0.99, 50)

    # Train the model
    model = train_model_lstm_2(quantiles=quantiles, epochs=epochs_input, lr=1e-3, batch_size=50, x=x, y=X_y)
    return model

def correct_ensembles(model, ensembles_out_of_sample):
    # Convert ensembles to tensor
    x_test = tf.convert_to_tensor(ensembles_out_of_sample)

    # Use the trained model to correct the ensembles
    corrected_ensembles = model(x_test)
    return corrected_ensembles.numpy()

def run_taqr(corrected_ensembles, actuals, quantiles, n_init, n_full):
    

    # Clean for NaNs
    actuals[np.isnan(actuals)] = 0

    taqr_results = []
    for q in quantiles:
        print("running TAQR for quantile: ", q)
        y_pred, _, _ = one_step_quantile_prediction(corrected_ensembles, actuals, n_init=n_init, n_full=n_full, quantile=q, already_correct_size = True)
        taqr_results.append(y_pred)

    return taqr_results

def calculate_qss(actuals, taqr_results, quantiles):
    qss_scores = multi_quantile_skill_score(actuals, taqr_results, quantiles)
    return qss_scores

def calculate_crps(actuals, corrected_ensembles):
    crps = ps.crps_ensemble(actuals, corrected_ensembles)
    return np.mean(crps)

def calculate_variogram_score(corrected_ensembles, actuals):
    variogram_score = variogram_score_R(corrected_ensembles, actuals, p=0.5)
    return variogram_score / len(actuals)

def save_results(data_source, qss_scores, crps_score, variogram_score, corrected_ensembles, taqr_results, model, idx_to_save, actuals_out_of_sample):
    # Save the scores
    import datetime as dt
    today = dt.datetime.today().strftime('%Y-%m-%d')

    np.save(f"results/{data_source}_qss_scores.npy", qss_scores)
    np.save(f"results/{data_source}_crps_score.npy", crps_score)
    np.save(f"results/{data_source}_variogram_score.npy", variogram_score)

    np.save(f"results/{data_source}_actuals_out_of_sample.npy", actuals_out_of_sample)

    # Save the corrected ensembles
    
    df_corrected_ensembles = pd.DataFrame(corrected_ensembles, index=idx_to_save)
    df_corrected_ensembles.to_csv(f"results/{data_source}_corrected_ensembles_{today}.csv")

    # Save the TAQR results
    np.save(f"results/{data_source}_taqr_results.npy", taqr_results)

    # Save the trained model
    model.save(f"saved_models/{data_source}_model_{today}.keras")

def minmax_scale(X):
    min_X = np.min(X)
    max_X = np.max(X)
    return (X - min_X) / (max_X - min_X)

from R_from_Python import run_r_script

def pipeline(data_area, type, type_of_nn = "FFNN", epochs = 100):
    # Load the data based on the data_source string
    actuals, ensembles, X_y , idx = load_data(data_area=data_area, type=type)

    # Train the model on 80% of the data
    train_size = int(0.8 * len(actuals))

    if type_of_nn.lower() == "lstm":
        timesteps = [0,1,2,6,12,24,48]
        Xs, X_Ys = create_dataset3(ensembles, X_y, timesteps)
        if np.isnan(Xs).any():
            print("Xs has NaNs")
        if np.isnan(X_Ys).any():
            print("X_Ys has NaNs")

        Xs[np.isnan(Xs).any(axis=(1,2))] = 0
        X_Ys[np.isnan(X_Ys).any(axis=1)] = 0

        # min-max Xs and X_Ys
        XY_s_max_train = np.max(X_Ys[:train_size])

        X_Ys_scaled_train = (X_Ys[:train_size]) / XY_s_max_train
        Xs_scaled_train = (Xs[:train_size]) / XY_s_max_train

        X_Ys_scaled_validation = (X_Ys[train_size:(train_size+200)]) / XY_s_max_train
        Xs_scaled_validation = (Xs[train_size:(train_size+200)]) / XY_s_max_train

        print("min and max of X_Ys: ", np.min(X_Ys), np.max(X_Ys))
        print("min and max of Xs: ", np.min(Xs), np.max(Xs))

        model = train_model_lstm(quantiles=np.linspace(0.05, 0.95, 20).round(3), epochs=epochs, 
                                   lr=1e-3, batch_size=20, 
                                   x=tf.convert_to_tensor(Xs_scaled_train), 
                                   y=tf.convert_to_tensor(X_Ys_scaled_train),
                                   x_val = tf.convert_to_tensor(Xs_scaled_validation),
                                   y_val = tf.convert_to_tensor(X_Ys_scaled_validation),
                                   n_timesteps=timesteps,
                                   data_name=f"{data_area}_{type}_{type_of_nn}_{epochs}")

    elif type_of_nn.lower() == "ffnn":
        model = train_ffnn(ensembles[:train_size], actuals[:train_size], X_y[:train_size], epochs_input=epochs, 
                           x_val = ensembles[train_size:(train_size+200)], y_val = X_y[train_size:(train_size+200)], 
                           data_name=f"{data_area}_{type}_{type_of_nn}_{epochs}")
        trained_idx = idx[:train_size]

    # Use the trained model to correct the ensembles
    if type_of_nn.lower() == "lstm":
     
        # corrected_ensembles = correct_ensembles(model, Xs[train_size:])
        # print("SHAPE OF XS: ", Xs[train_size:].shape)
        min_Xs_test = np.min(Xs[:train_size])
        max_Xs_test = np.max(Xs[:train_size])
        Xs_scaled_test = (Xs[train_size:]) / XY_s_max_train

        corrected_ensembles = model(Xs_scaled_test)

        # Reverse transform...       
        corrected_ensembles = corrected_ensembles * (XY_s_max_train) # - min_Xs_test)) + min_Xs_test

        actuals_out_of_sample = actuals[train_size:]
        # actuals_out_of_sample = actuals_out_of_sample[timesteps:]
        
        actuals_out_of_sample = (actuals_out_of_sample) # - min_XY_s) / (max_XY_s - min_XY_s)

    elif type_of_nn.lower() == "ffnn":
        corrected_ensembles = correct_ensembles(model, ensembles[train_size:])
        actuals_out_of_sample = actuals[train_size:]
    
    test_idx = idx[train_size:]

    

    # Run the TAQR algorithm with the corrected ensembles
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_full = len(actuals_out_of_sample)
    n_init = int(0.25*n_full)
    print("n_init, n_full: ",  n_init, n_full)

    # plt.figure(figsize=(12,8))
    # plt.plot(corrected_ensembles, color = "grey", alpha=0.5)
    # plt.title(f"Corrected ensembles with {type_of_nn} and {epochs} epochs")  
    # plt.plot(actuals_out_of_sample, color = "black", label = "Actuals")
    # plt.legend()
    # plt.title("Corrected ensembles")
    # plt.show()


    # plt.figure(figsize=(12,8))
    # plt.imshow(corrected_ensembles, aspect="auto", cmap="viridis")  
    # plt.colorbar()
    # plt.title("Corrected ensembles cmap")
    # plt.show()
    if type_of_nn.lower() == "lstm":
        try:
            taqr_results = run_taqr(corrected_ensembles, actuals_out_of_sample, quantiles, n_init, n_full)
            actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

        except:
            print("Error in run_taqr due to LSTM")
            actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

            taqr_results = [0,0,0,0,0]
            qss_scores = 0
    else:
        taqr_results = run_taqr(corrected_ensembles, actuals_out_of_sample, quantiles, n_init, n_full)
        actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]
            # qss_scores = calculate_qss(actuals_out_of_sample, taqr_results, quantiles)
        # except:
        print("TAQR succesfull!")
        # X = corrected_ensembles[:n_full, :]
        # Y = actuals_out_of_sample[:n_full]
        # X_for_residuals = X[:n_init, :]
        # Y_for_residuals = Y[:n_init]
        import datetime as dt
        today = dt.datetime.today().strftime('%Y-%m-%d')
        # save them for to be used in rq... X_for_residuals and Y_for_residuals
        # np.savetxt(f"X_for_residuals_{data_area}_{today}.csv", X_for_residuals, delimiter=",")
        # np.savetxt(f"Y_for_residuals_{data_area}_{today}.csv", Y_for_residuals, delimiter=",")
        #actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

            # taqr_results = [0,0,0,0,0]
        #qss_scores = 0

    corrected_ensembles = corrected_ensembles[(n_init+1):(n_full-1)]
    idx_to_save = test_idx[(n_init+1):(n_full-1)]

    corrected_ensembles_sorted = np.sort(corrected_ensembles, axis=1)
    n,m = corrected_ensembles_sorted.shape
    corrected_ensembles_sorted_selected = corrected_ensembles_sorted[:, [int(x) for x in np.linspace(0.05, 0.95, 20).round(3)*m]]

    qss_scores = calculate_qss(actuals_out_of_sample, corrected_ensembles_sorted_selected,  np.linspace(0.05, 0.95, 20).round(3))

    # Calculate scores (QSS, CRPS, Variogram) for the TAQR results
    
    crps_score = calculate_crps(actuals_out_of_sample, corrected_ensembles)
    variogram_score = calculate_variogram_score(corrected_ensembles, actuals_out_of_sample)

    data_source = f"{data_area}_{type}_{type_of_nn}"
    # Save the scores, corrected ensembles, TAQR results, and trained model
    save_results(data_source, qss_scores, crps_score, variogram_score, corrected_ensembles, taqr_results, model, idx_to_save, actuals_out_of_sample)



import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

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

# def estimate_ou_parameters(data, dt):
#     """Estimate the parameters of the Ornstein-Uhlenbeck process.
    
#     Parameters:
#     - data: The observed data.
#     - dt: Time increment between observations.
    
#     Returns:
#     - The estimated parameters (lambda, sigma).
#     """
#     initial_guess = [0.5, 100]
#     bounds = [(0, None), (0.001, None)]
    
#     result = minimize(ou_log_likelihood, initial_guess, args=(data, dt), bounds=bounds, method='L-BFGS-B')
    
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("Optimization failed. Try different initial guesses or check the data.")


from scipy.optimize import minimize

def estimate_ou_parameters(data, dt):
    """Estimate the parameters of the Ornstein-Uhlenbeck process.
    
    Parameters:
    - data: The observed data.
    - dt: Time increment between observations.
    
    Returns:
    - The estimated parameters (lambda, sigma).
    """
    initial_guesses = [
        [0.5, 100],   # Current initial guess
        [1.0, 50],    # First alternative guess
        [0.1, 200]    # Second alternative guess
    ]
    bounds = [(0, None), (0.001, None)]
    
    for guess in initial_guesses:
        result = minimize(ou_log_likelihood, guess, args=(data, dt), bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return result.x
    
    # If all attempts fail
    raise ValueError("Optimization failed. Tried different initial guesses, but all failed. Check the data or consider different initial guesses.")

def simulate_ou_process(lam, sigma, dt, n_steps, initial_value, X):
    X[0] = 0# initial_value
    for t in range(1, n_steps):
        # print(t)
        X[t] = X[t-1] * np.exp(-lam * dt) + sigma * np.sqrt((1 - np.exp(-2 * lam * dt)) / (2 * lam)) * np.random.normal()
    return X


def SDE_subprocess(corrected_ensembles, actuals, quantiles, n_steps = 100, number_of_repeats = 100, print_ = False):
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

        # Generate the process
        # print("Simulating OU process for quantile: ", q, " with parameters: ", lam, sigma, dt)
        # print("actuals, n_steps, ", len(actuals), n_steps)
        # print(residuals.shape)
        simulated_data_dict[q] = simulate_ou_process(lam, sigma, dt, n_steps = len(actuals)-n_steps, initial_value = 0, X = residuals[n_steps:])

    
    sde_corrected_ensembles = pd.DataFrame()
    for q in quantiles:
        print(simulated_data_dict[q].shape)
        sde_corrected_ensembles[str(q)] = percentile_value_dict[q].values[n_steps:] + (residuals_dict[q][n_steps:] + simulated_data_dict[q])
    
    median = percentile_value_dict[quantiles[0]][n_steps:]
    variogram_score_sde_corrected = variogram_score_R(sde_corrected_ensembles.values, actuals.loc[median.index].values, p=0.5)
    variogram_score_ = variogram_score_R(corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values, actuals.loc[median.index].values, p=0.5)

    crps_sde_corrected = calculate_crps(actuals.loc[median.index].values, sde_corrected_ensembles.values)
    crps_ = calculate_crps(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values)


    qss_sde_corrected = calculate_qss(actuals.loc[median.index].values, sde_corrected_ensembles.values, quantiles = quantiles)
    qss_score = (calculate_qss(actuals.loc[median.index].values, corrected_ensembles.loc[median.index].quantile(quantiles, axis = 1).T.values, quantiles = quantiles))




    if print_:
        print("Variogram score sde corrected: ", variogram_score_sde_corrected )
        print("vs.")
        print("Variogram score: ", variogram_score_)
        print("CRPS for 3 part SDE corrected ensembles: ", crps_sde_corrected)
        print("vs.")
        print("CRPS for prev. corrected ensembles: ", crps_)
        print("QSS for SDE corrected ensembles: ", qss_sde_corrected)
        print("vs.")
        print("QSS for prev. corrected ensembles: ", qss_score)
    return variogram_score_sde_corrected, variogram_score_, crps_sde_corrected, crps_, qss_sde_corrected, qss_score


def pipeline_SDE(data_area, type, type_of_nn = "FFNN", epochs = 100):
    # Load the data based on the data_source string
    actuals, ensembles, X_y , idx = load_data(data_area=data_area, type=type )

    # Train the model on 80% of the data
    train_size = int(0.75 * len(actuals))

    if type_of_nn.lower() == "lstm":
        timesteps = [0,1,2,6,12,24,48]
        Xs, X_Ys = create_dataset3(ensembles, X_y, timesteps)
        if np.isnan(Xs).any():
            print("Xs has NaNs")
        if np.isnan(X_Ys).any():
            print("X_Ys has NaNs")

        Xs[np.isnan(Xs).any(axis=(1,2))] = 0
        X_Ys[np.isnan(X_Ys).any(axis=1)] = 0

        # Scaling
        XY_s_max_train = np.max(X_Ys[:train_size])

        X_Ys_scaled_train = (X_Ys[:train_size]) / XY_s_max_train
        Xs_scaled_train = (Xs[:train_size]) / XY_s_max_train

        # Validation set
        X_Ys_scaled_validation = (X_Ys[train_size:(train_size+200)]) / XY_s_max_train
        Xs_scaled_validation = (Xs[train_size:(train_size+200)]) / XY_s_max_train

        print("min and max of X_Ys: ", np.min(X_Ys), np.max(X_Ys))
        print("min and max of Xs: ", np.min(Xs), np.max(Xs))

        model = train_model_lstm(quantiles=np.linspace(0.05, 0.95, 20).round(3), epochs=epochs, 
                                   lr=1e-3, batch_size=20, 
                                   x=tf.convert_to_tensor(Xs_scaled_train), 
                                   y=tf.convert_to_tensor(X_Ys_scaled_train),
                                   x_val = tf.convert_to_tensor(Xs_scaled_validation),
                                   y_val = tf.convert_to_tensor(X_Ys_scaled_validation),
                                   n_timesteps=timesteps)

    elif type_of_nn.lower() == "ffnn":
        model = train_ffnn(ensembles[:train_size], actuals[:train_size], 
                           X_y[:train_size], epochs_input=epochs, 
                           x_val = ensembles[train_size:(train_size+200)], 
                           y_val = X_y[train_size:(train_size+200)])
        trained_idx = idx[:train_size]

    # Use the trained model to correct the ensembles
    if type_of_nn.lower() == "lstm":
        min_Xs_test = np.min(Xs[:train_size])
        max_Xs_test = np.max(Xs[:train_size])
        Xs_scaled_test = (Xs[train_size:]) / XY_s_max_train

        corrected_ensembles = model(Xs_scaled_test)

        # Reverse transform...       
        corrected_ensembles = corrected_ensembles * (XY_s_max_train) # - min_Xs_test)) + min_Xs_test

        actuals_out_of_sample = actuals[train_size:]
        # actuals_out_of_sample = actuals_out_of_sample[timesteps:]
        
        actuals_out_of_sample = (actuals_out_of_sample) # - min_XY_s) / (max_XY_s - min_XY_s)

    elif type_of_nn.lower() == "ffnn":
        corrected_ensembles = correct_ensembles(model, ensembles[train_size:])
        actuals_out_of_sample = actuals[train_size:]
    
    test_idx = idx[train_size:]

    

    # Run the TAQR algorithm with the corrected ensembles
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_full = len(actuals_out_of_sample)
    n_init = int(0.25*n_full)
    print("n_init, n_full: ",  n_init, n_full)

    # plt.figure(figsize=(12,8))
    # plt.plot(corrected_ensembles, color = "grey", alpha=0.5)
    # plt.title(f"Corrected ensembles with {type_of_nn} and {epochs} epochs")  
    # plt.plot(actuals_out_of_sample, color = "black", label = "Actuals")
    # plt.legend()
    # plt.title("Corrected ensembles")
    # plt.show()


    # plt.figure(figsize=(12,8))
    # plt.imshow(corrected_ensembles, aspect="auto", cmap="viridis")  
    # plt.colorbar()
    # plt.title("Corrected ensembles cmap")
    # plt.show()
    if type_of_nn.lower() == "lstm":
        try:
            taqr_results = run_taqr(corrected_ensembles, actuals_out_of_sample, quantiles, n_init, n_full)
            actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

        except:
            print("Error in run_taqr due to LSTM")
            actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

            taqr_results = [0,0,0,0,0]
            qss_scores = 0
    else:
        taqr_results = run_taqr(corrected_ensembles, actuals_out_of_sample, quantiles, n_init, n_full)
        actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]
        print("TAQR succesfull!")
        import datetime as dt
        today = dt.datetime.today().strftime('%Y-%m-%d')

    corrected_ensembles = corrected_ensembles[(n_init+1):(n_full-1)]
    idx_to_save = test_idx[(n_init+1):(n_full-1)]

    corrected_ensembles_sorted = np.sort(corrected_ensembles, axis=1)
    n,m = corrected_ensembles_sorted.shape
    corrected_ensembles_sorted_selected = corrected_ensembles_sorted[:, [int(x) for x in np.linspace(0.05, 0.95, 20).round(3)*m]]

    variogram_score_sde_corrected, variogram_score_, crps_sde_corrected, crps_, qss_sde_corrected, qss_score = SDE_subprocess(corrected_ensembles, actuals, quantiles, n_steps = 100, number_of_repeats = 100, print_ = False)

    qss_scores = calculate_qss(actuals_out_of_sample, corrected_ensembles_sorted_selected,  np.linspace(0.05, 0.95, 20).round(3))

    crps_score = calculate_crps(actuals_out_of_sample, corrected_ensembles)
    variogram_score = calculate_variogram_score(corrected_ensembles, actuals_out_of_sample)

    data_source = f"{data_area}_{type}_{type_of_nn}"
    # Save the scores, corrected ensembles, TAQR results, and trained model
    save_results(data_source, qss_scores, crps_score, variogram_score, corrected_ensembles, taqr_results, model, idx_to_save, actuals_out_of_sample)


def load_and_present_results(data_area, type, type_of_nn = "FFNN"):
    data_source = f"{data_area}_{type}_{type_of_nn}"
    qss_scores = np.load(f"results/{data_source}_qss_scores.npy")
    crps_score = np.load(f"results/{data_source}_crps_score.npy")
    variogram_score = np.load(f"results/{data_source}_variogram_score.npy")
    taqr_results = np.load(f"results/{data_source}_taqr_results.npy")
    actuals_out_of_sample = np.load(f"results/{data_source}_actuals_out_of_sample.npy")

    corrected_ensembles = pd.read_csv(f"results/{data_source}_corrected_ensembles_2024-05-27.csv") # TODO: FIX THIS
    idx_out_of_sample = pd.to_datetime(corrected_ensembles["HourDK"])

    # Load
    Y, X, _, idx = load_data(data_area, type)
    # idx_for_actuals = np.arange(3200+200, 3200+200+598) # TODO FIX THIS

    # Find the index we need to index into numpy array based on the dates in the two index. 
    idx_for_actuals = np.where(np.isin(idx, idx_out_of_sample))[0]

    X = X[idx_for_actuals,:]
    Y = Y[idx_for_actuals]
    X_sorted = np.sort(X, axis=1)
    n,m = X_sorted.shape
    X_selected = X_sorted[:, [int(x) for x in np.linspace(0.05, 0.95, 20).round(3)*m]]

    # calculate QSS, CRPS, and Variogram scores for Y and X:
    qss_scores_org = calculate_qss(Y, X_selected.T, np.linspace(0.05, 0.95, 20).round(3))
    crps_score_org = calculate_crps(Y, X)
    variogram_score_org = calculate_variogram_score(X, Y)
    
    y_true = Y

    print("------------------------")
    print(f"QSS scores for {data_source}: {qss_scores}. Mean: {np.mean(qss_scores):.3f}")
    print(f"CRPS score for {data_source}: {crps_score:.3f}")
    print(f"Variogram score for {data_source}: {variogram_score:.3f}")
    print("COMPARED TO: ")
    print(f"Original QSS scores for {data_source}: {qss_scores_org}. Mean: {np.mean(qss_scores_org):.3f}")
    print(f"Original CRPS score for {data_source}: {crps_score_org:.3f}")
    print(f"Original Variogram score for {data_source}: {variogram_score_org:.3f}")
    print("------------------------")




    plt.figure()
    for i, q in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        plt.plot(taqr_results[i], label=f"Quantile: {q}", color="grey", alpha=0.5)
    plt.plot(actuals_out_of_sample, label="True", color="black")
    plt.legend()
    plt.title(f"TAQR results for {data_source}")
    plt.show()


    plt.figure()
    corrected_ensembles.plot(legend=False, alpha=0.3, color="grey")
    # plt.plot(corrected_ensembles, color = "grey", alpha=0.5)
    plt.plot(actuals_out_of_sample[0:], color = "black", label = "Actuals")  # TODO, check... 
    plt.legend()
    plt.title("Corrected ensembles")
    plt.show()

    return qss_scores, crps_score, variogram_score, corrected_ensembles, taqr_results

import numpy as np
import torch

def create_data_dict(corrected_ensembles, actuals, train_ratio=0.8):
    # Convert NumPy arrays to PyTorch tensors
    corrected_ensembles_tensor = torch.from_numpy(corrected_ensembles)
    actuals_tensor = torch.from_numpy(actuals).unsqueeze(1)  # Add an extra dimension for consistency

    # Get the number of samples
    num_samples = corrected_ensembles_tensor.shape[0]

    # Calculate the number of training samples
    num_train_samples = int(train_ratio * num_samples)

    # Create the data dictionary
    data_dict = {
        'train_input': corrected_ensembles_tensor[:num_train_samples],
        'test_input': corrected_ensembles_tensor[num_train_samples:],
        'train_label': actuals_tensor[:num_train_samples],
        'test_label': actuals_tensor[num_train_samples:]
    }

    return data_dict

from kan import KAN, create_dataset
import torch

def fit_KAN_to_corrected_ensembles(data_area, type, type_of_nn):
    
    model = KAN(width=[52,10,1], grid=5, k = 3)
    actuals, _, _, idx = load_data(data_area, type)
    _, _, _, corrected_ensembles, _  = load_and_present_results(data_area, type, type_of_nn = type_of_nn)

    # remove nans
    corrected_ensembles[np.isnan(corrected_ensembles)] = 0
    actuals[np.isnan(actuals)] = 0


    # Create the data dictionary
    data_dict = create_data_dict(corrected_ensembles, actuals, train_ratio=0.8)
    # print(data_dict)
    # results = model.train(data_dict, opt="LBFGS", steps=100)

    # # model.plot(beta=10)
    # train_losses = results['train_loss']
    # test_losses = results['test_loss']

    grids = np.array([5,10,20,40])

    train_losses = []
    test_losses = []
    steps = 50
    k = 3

    for i in range(grids.shape[0]):
        print(f"Grid: {grids[i]}")
        if i == 0:
            model = KAN(width=[52,10,1], grid=grids[i], k=k)
        if i != 0:
            model = KAN(width=[52,10,1], grid=grids[i], k=k).initialize_from_another_model(model, data_dict['train_input'])
        results = model.train(data_dict, opt="LBFGS", steps=steps, stop_grid_update_step=30)
        train_losses += results['train_loss']
        test_losses += results['test_loss']

    plt.figure()
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.ylabel('RMSE')
    plt.xlabel('step')
    plt.yscale('log')
    plt.show()



    
import time

if __name__ == "__main__":

    start = time.time()
    print("Starting pipeline")
    # pipeline("DK1", "OnshoreWindPower", "lstm", epochs = 30)
    # print("Done with DK1 OnshoreWindPower. Time used: ", time.time() - start)
    # pipeline("DK1", "OnshoreWindPower", "ffnn", epochs = 20)
    # pipeline("DK2", "OnshoreWindPower", "lstm", epochs = 200)
    # print("Done with DK2 OnshoreWindPower")
    # pipeline("DK2", "OffshoreWindPower", "lstm", epochs = 200)
    # print("Done with DK2 OffshoreWindPower")
    # print("Time used: ", time.time() - start)

    # try:
    #     pipeline("DK1", "SolarPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK1 SolarPower ffnn: {e}")

    # try:
    #     pipeline("DK2", "SolarPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK2 SolarPower ffnn: {e}")

    # try:
    #     pipeline("DK1", "OnshoreWindPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK1 OnshoreWindPower ffnn: {e}")

    # try:
    #     pipeline("DK1", "OffshoreWindPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK1 OffshoreWindPower ffnn: {e}")

    # try:
    #     pipeline("DK2", "OnshoreWindPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK2 OnshoreWindPower ffnn: {e}")

    # try:
    #     pipeline("DK2", "OffshoreWindPower", "ffnn", epochs=500)
    # except Exception as e:
    #     print(f"Error with DK2 OffshoreWindPower ffnn: {e}")

    # try:
    #     print("Time used: ", time.time() - start)
    # except Exception as e:
    #     print(f"Error with printing time for ffnn: {e}")

    # try:
    #     pipeline("DK1", "SolarPower", "lstm", epochs=200)
    # except Exception as e:
    #     print(f"Error with DK1 SolarPower lstm: {e}")

    # try:
    #     pipeline("DK2", "SolarPower", "lstm", epochs=200)
    # except Exception as e:
    #     print(f"Error with DK2 SolarPower lstm: {e}")

    try:
        pipeline("DK1", "OnshoreWindPower", "lstm", epochs=200)
    except Exception as e:
        print(f"Error with DK1 OnshoreWindPower lstm: {e}")

    try:
        pipeline("DK1", "OffshoreWindPower", "lstm", epochs=200)
    except Exception as e:
        print(f"Error with DK1 OffshoreWindPower lstm: {e}")

    try:
        pipeline("DK2", "OnshoreWindPower", "lstm", epochs=200)
    except Exception as e:
        print(f"Error with DK2 OnshoreWindPower lstm: {e}")

    try:
        pipeline("DK2", "OffshoreWindPower", "lstm", epochs=200)
    except Exception as e:
        print(f"Error with DK2 OffshoreWindPower lstm: {e}")

    try:
        print("Time used: ", time.time() - start)
    except Exception as e:
        print(f"Error with printing time for lstm: {e}")


   

    # pipeline("DK1", "SolarPower", "ffnn", epochs = 100)
    # load_and_present_results("DK1", "SolarPower", "ffnn")
    # load_and_present_results("DK2", "SolarPower", "ffnn")

    # data_source = "DK1_OnshoreWindPower_ffnn"
    # qss_scores = np.load(f"results/{data_source}_qss_scores.npy")
    # crps_score = np.load(f"results/{data_source}_crps_score.npy")
    # variogram_score = np.load(f"results/{data_source}_variogram_score.npy")
    # taqr_results = np.load(f"results/{data_source}_taqr_results.npy")
    # print("TAQR results: ", taqr_results)


    # load in some data
    # actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK1_fixed.pkl")
    # ensembles_DK1_onshorewindpower = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1_fixed.pkl")
    # ensembles_DK1_offshorewindpower = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK1_fixed.pkl")
    # ensembles_DK1_solarpower = pd.read_pickle("loaded_variables/ensembles_solar_DK1_fixed_v2.pkl")

    # Corrected_ensembles_DK1_OffshoreWindPower_lstm = pd.read_csv("results/DK1_OffshoreWindPower_lstm_corrected_ensembles_2024-05-20.csv")
    # Corrected_ensembles_DK1_OffshoreWindPower_lstm.set_index("HourDK", inplace=True)
    # Corrected_ensembles_DK1_OnshoreWindPower_lstm = pd.read_csv("results/DK1_OnshoreWindPower_lstm_corrected_ensembles_2024-05-20.csv")
    # Corrected_ensembles_DK1_OnshoreWindPower_lstm.set_index("HourDK", inplace=True)
    # Corrected_ensembles_DK1_OnshoreWindPower_lstm.index = pd.to_datetime(Corrected_ensembles_DK1_OnshoreWindPower_lstm.index)
    # Y_DK1_OnshoreWindPower = actuals["OnshoreWindPower"]
    # Y_DK1_OffshoreWindPower = actuals["OffshoreWindPower"]
    # Y_DK1_SolarPower = actuals["SolarPower"]
    # quantiles = [0.25, 0.5, 0.75]
    # print(SDE_subprocess(Corrected_ensembles_DK1_OffshoreWindPower_lstm, Y_DK1_OffshoreWindPower.loc[Corrected_ensembles_DK1_OffshoreWindPower_lstm.index], quantiles, n_steps = 100, number_of_repeats = 100, print_ = True))