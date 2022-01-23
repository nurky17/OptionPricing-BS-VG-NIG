import math
from scipy.integrate import quad
import scipy.special as scsp
from scipy.optimize import curve_fit
from mpmath import besselk
import numpy as np
import joblib

# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

# # import R's utility package
# utils = rpackages.importr('utils')
#
# # select a mirror for R packages
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('GeneralizedHyperbolic')
GH = rpackages.importr('GeneralizedHyperbolic')

rlog = robjects.r['log']

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm

def d1(S,K,T,r,sigma):
    #q = 0.0165
    return np.array((np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T)), dtype='float64')

def d2(S,K,T,r,sigma):
    return np.array(d1(S,K,T,r,sigma)-sigma*np.sqrt(T), dtype='float64')

## data is a dataframe with 3 columns, first one with values of strike price K, second one of time maturity T, and third one the actual call price values ##

def Call_BS(data, sigma, S, r):
    #S = 1243.73; r = 0.05-0.0165
    #S = 4535.43; r = 0.021;
    return np.array([S * norm.cdf(d1(S, data[i][0], data[i][1], r, sigma)) - data[i][0] * np.exp(-r * data[i][1]) * norm.cdf(d2(S, data[i][0], data[i][1], r, sigma)) for i in range(data.shape[0])], dtype='float64')

## the following functions are used for calculating option price under VG as given in Madan, D. B., Carr, P. P., and Chang, E. C. (1998).  The variance gamma processand option pricing.Review of Finance, 2(1):79–105 ##
def integrand(up, alpha, beta, gamma, x, y):
    #print('dosao do integrala')
    return np.array((up**(alpha-1))*((1-up)**(gamma-alpha-1))*((1-up*x)**(-beta))*np.exp(up*y))

def gammas(alpha, gamma):
    #print('dosao do gamma')
    return scsp.gamma(gamma)/(scsp.gamma(alpha)*scsp.gamma(gamma-alpha))

def phi(alpha, beta, gamma, x, y):
    #print('dosao do phi')
    return np.array(gammas(alpha, gamma) * quad(integrand, 0, 1, args=(alpha, beta, gamma, x, y))[0], dtype='float64')

def Psi(a, b, gamma):
    c = abs(a) * math.sqrt(2 + b**2);  u = b / math.sqrt(2 + b**2);
    first_summand = c**(gamma + 1/2) * np.exp(np.sign(a)*c)*(1+u)**gamma / ( math.sqrt(2 * math.pi) * scsp.gamma(gamma) * gamma ) * besselk(gamma + 1/2, c) * phi(gamma, 1 - gamma, 1 + gamma, (1 + u)/2, -np.sign(a) * c * (1+u))
    second_summand = - np.sign(a) * c**(gamma + 1/2) * np.exp(np.sign(a)*c)*(1+u)**(1 + gamma) / ( math.sqrt(2 * math.pi) * scsp.gamma(gamma) * (1 + gamma) ) * besselk(gamma - 1/2, c) * phi(1 + gamma, 1 - gamma, 2 + gamma, (1 + u)/2, -np.sign(a) * c * (1+u))
    third_summand = np.sign(a) * c**(gamma + 1/2) * np.exp(np.sign(a)*c)*(1+u)**gamma / ( math.sqrt(2 * math.pi) * scsp.gamma(gamma) * gamma ) * besselk(gamma - 1/2, c) * phi(gamma, 1 - gamma, 1 + gamma, (1 + u)/2, -np.sign(a) * c * (1+u))
    #print('dosao do psi')
    return np.array(first_summand + second_summand + third_summand, dtype='float64')

def Call_VG(data, sigma, nu, theta, S, r):
    # data: 3d np.array, something like [[(K1, T1, market_price1),(K2, T2, market_price2),...]]
    # data[i][0]: K, i.e. strike price
    # data[i][1]: T, i.e. time to maturity
    # data[i][2]: real market prices
    zeta = -theta / sigma**2
    s = sigma / np.sqrt(1 + (theta/sigma)**2 * nu/2)
    c1 = nu * (zeta * s + s)**2 / 2
    c2 = nu * (zeta * s)**2 / 2
    return np.array([S * Psi(1/s * (np.log(S/data[i][0]) + r*data[i][1] + data[i][1]/nu * np.log((1-c1)/(1-c2))) * np.sqrt((1-c1)/nu), (zeta * s + s) * np.sqrt(nu/(1 - c1)), data[i][1]/nu) - data[i][0] * np.exp(- r * data[i][1]) * Psi(1/s * (np.log(S/data[i][0]) + r*data[i][1] + data[i][1]/nu * np.log((1-c1)/(1-c2))) * np.sqrt((1 - c2)/nu), zeta*(s**2) * np.sqrt(nu/(1-c2)), data[i][1]/nu) for i in range(data.shape[0])], dtype='float64')

## the following option price formula under NIG is to be found in Albrecher,  H.  and  Predota,  M.  (2004).   On  Asian  option  pricing  for  NIG  L ́evyprocesses.Journal of computational and applied mathematics, 172(1):153–168. ##
def Call_NIG(data, alpha, beta, delta, mu, S, r):
    # data: 3d np.array, something like [[(K1, T1, market_price1),(K2, T2, market_price2),...]]
    # data[i][0]: K, i.e. strike price
    # data[i][1]: T, i.e. time to maturity
    theta = - beta - 0.5 - (mu-r)/(2*delta)*np.sqrt((4*delta**2*alpha**2)/((mu-r)**2+delta**2)-1)
    if not (alpha >=0.5 and abs(mu)<= delta*np.sqrt(2*alpha-1) and theta >= - alpha - beta and theta <= alpha - beta - 1):
        return None
    else:
        return np.array([S * (1.0 - GH.pnig(rlog(float(data[i][0]/S))[0], alpha=float(alpha), beta=float(beta+ theta + 1), delta=float(delta*data[i][1]), mu=float(mu*data[i][1]))[0]) - data[i][0] * np.exp(- r * data[i][1]) * ( 1.0 - GH.pnig(rlog(float(data[i][0]/S))[0], alpha=float(alpha), beta=float(beta + theta), delta=float(delta*data[i][1]), mu=float(mu*data[i][1]))[0]) for i in range(data.shape[0])], dtype='float64')


## the example data, used in the Master thesis ##
S = 4535.43; r = 0.021
optimization_data = joblib.load('options_training_data.joblib')

## fitting BS ##
fit_BS, nesto_BS = curve_fit(lambda x, sigma : Call_BS(x, sigma, S, r), optimization_data[:, :2], optimization_data[:, 2], (0.5), bounds=((0), (1)))
print('\n BS risk neutral parameters are:')
print(fit_BS)
sigma_BS = fit_BS[0]

## fitting NIG ##
fit_NIG, nesto_NIG = curve_fit(lambda x, alpha, beta, delta, mu : Call_NIG(x, alpha, beta, delta, mu, S, r), optimization_data[:, :2], optimization_data[:, 2],
                       (5, -0.5, 0.5, 0.2), bounds=((0.5, -100, 0.0001, -0.8), (100, 100, 100, 100)))
print('\n NIG risk neutral parameters are:')
print(fit_NIG)
alpha_NIG, beta_NIG, delta_NIG, mu_NIG = fit_NIG[0], fit_NIG[1], fit_NIG[2], fit_NIG[3]

## fitting VG ##
fit_VG, nesto_VG = curve_fit(lambda x, sigma, nu, theta : Call_VG(x, sigma, nu, theta, S, r), optimization_data[:, :2], optimization_data[:, 2],
                       (1, 1, 1), bounds=((0.00001, 0.0001, -10), (10, 10, 10)))
print('\n VG risk neutral parameters are:')
print(fit_VG)
sigma_VG, nu_VG, theta_VG = fit_VG[0], fit_VG[1], fit_VG[2]

## put this to be True if you want print MSE and R^2 errors ##
print_errors = False

if print_errors == True:
    print('MSE BS: %.5f, R^2 BS: %.5f' % (
        mean_squared_error(optimization_data[:, 2], Call_BS(optimization_data, sigma_BS, S, r)),
        r2_score(optimization_data[:, 2], Call_BS(optimization_data, sigma_BS, S, r))))


    print('MSE NIG: %.5f, R^2 NIG: %.5f' % (mean_squared_error(optimization_data[:, 2],
                                                                               Call_NIG(optimization_data,
                                                                                        alpha_NIG, beta_NIG, delta_NIG, mu_NIG, S, r)),
                                                            r2_score(optimization_data[:, 2],
                                                                     Call_NIG(optimization_data, alpha_NIG, beta_NIG, delta_NIG, mu_NIG, S, r))))


    print('MSE VG: %.5f, R^2 VG: %.5f' % (mean_squared_error(optimization_data[:, 2],
                                                                               Call_VG(optimization_data,
                                                                                        sigma_VG, nu_VG, theta_VG, S, r)),
                                                            r2_score(optimization_data[:, 2],
                                                                     Call_VG(optimization_data, sigma_VG, nu_VG, theta_VG, S, r))))

## plot preferences ##
plt.rcParams["axes.labelpad"] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rc('legend',fontsize=14)
plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle="--", color='grey', alpha=0.4, lw = 0.5)


## BS ##
plt.figure(figsize=(9.5,6))
plt.scatter(optimization_data[:, 0], optimization_data[:, 2], marker='o', label='real data')
plt.scatter(optimization_data[:, 0], Call_BS(optimization_data, sigma_BS, S, r), marker='x', label='BS')
plt.xlabel('strike')
plt.ylabel('call price')
plt.grid()
plt.legend()
plt.show()

## VG ##
plt.figure(figsize=(9.5,6))
plt.scatter(optimization_data[:, 0], optimization_data[:, 2], marker='o', label='real data')
plt.scatter(optimization_data[:, 0], Call_VG(optimization_data, sigma_VG, nu_VG, theta_VG, S, r), marker='x', label='VG')
plt.xlabel('strike')
plt.ylabel('call price')
plt.grid()
plt.legend()
plt.show()

## NIG ##
plt.figure(figsize=(9.5,6))
plt.scatter(optimization_data[:, 0], optimization_data[:, 2], marker='o', label='real data')
plt.scatter(optimization_data[:, 0], Call_NIG(optimization_data, alpha_NIG, beta_NIG, delta_NIG, mu_NIG, S, r), marker='x', label='NIG')
plt.xlabel('strike')
plt.ylabel('call price')
plt.grid()
plt.legend()
plt.show()