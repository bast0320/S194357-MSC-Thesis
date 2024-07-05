# %%
import numpy as np
from scipy.integrate import nquad

def emcdf(x, u):
    return np.mean(np.all(x <= u, axis=1))

def intCrps(d, igd, l, u, reltol=0.01, max_reltol=0.1):
    converged = False
    while not converged:
        if np.sum(l < u) == d:
            try:
                int_result, abserr = nquad(igd, ranges=list(zip(l, u)), opts={'epsrel': reltol})
                ifail = 0 if abserr < reltol else 1
                if ifail == 0:
                    converged = True
                else:
                    reltol += 0.01
            except Exception as e:
                print(f"Integration error: {e}")
                reltol += 0.01
            if reltol > max_reltol:
                converged = True
        else:
            int_result = 0
            ifail = 1
            converged = True
    return int_result, ifail

def crps(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    l = np.min(x, axis=0)
    u = np.max(x, axis=0)
    m, k = x.shape

    def igdL(*u):
        return emcdf(x, u) ** 2

    def igdU(*u):
        return (emcdf(x, u) - 1) ** 2

    intL, ifailL = intCrps(k, igdL, l, y)
    intU, ifailU = intCrps(k, igdU, y, u)

    if ifailL == 0 and ifailU == 0:
        return intL + intU
    else:
        return np.nan
#%%    

# Example usage
y = np.array([1, 2, 3, 4, 5])  # Actual observations
x = np.array([[0.5, 1.5, 2.5, 3.5, 4.8],  # Lower quantiles
                      [1.5, 2.5, 3.4, 4.5, 5.5]])  # Upper quantiles

score = crps(x, y)
print("CRPS score:", score)

# %%
# plot a Beta(2,5) (𝑦 = 0.264, as a green dot)
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import beta

# a, b = 2, 5
# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

# x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
# y = beta.pdf(x, a, b)

# plt.plot(x, y, 'r-', lw=5, alpha=0.6, label='beta pdf')
# plt.axvline(x=mean, color='k', linestyle='--', label='Mean')
# plt.axvline(x=mean + var, color='b', linestyle='--', label='Mean + Variance')
# plt.axvline(x=mean - var, color='b', linestyle='--', label='Mean - Variance')

# # plot median
# plt.axvline(x=beta.ppf(0.5, a, b), color='g', linestyle='--', label='Median')

# # now add a beta 1,5 to the plot in green
# a1,b1 = 1,5
# mean1, var1, skew1, kurt1 = beta.stats(a1, b1, moments='mvsk')
# x1 = np.linspace(beta.ppf(0.01, a1, b1), beta.ppf(0.99, a1, b1), 100)
# y1 = beta.pdf(x1, a1, b1)
# plt.plot(x1, y1, 'g-', lw=5, alpha=0.6, label='beta pdf, wrong')

# # plot median
# plt.axvline(x=beta.ppf(0.5, a1, b1), color='g', linestyle='--', label='Median')

# plt.axvline(x=0.264, color='g', linestyle='--', label='y = 0.264')
# plt.legend()
# plt.show()
# %%




# Example usage
# y = np.array([3, 5, 7, 9, 11])  # Actual observations
# x = np.array([[0.2, 1.5, 2.5, 3.5, 4.8],  # Lower quantiles
#                       [1.5, 2.5, 3.4, 4.5, 5.5]])  # Predicted quantiles for a specific q

# # Assume we are evaluating the 0.5 quantile
# q = 0.5
# score = variogram_score_R(x, y, p=0.5)
# print("Variogram score:", score)
 