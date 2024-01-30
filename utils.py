import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
from numpy.linalg import solve
import logging
logging.basicConfig(level = logging.DEBUG)
from scipy.stats import moment, norm

def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])

    # arrays for storing prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
    
    return S, v



def fleishman(b, c, d):
    """calculate the variance, skew and kurtois of a Fleishman distribution
    F = -c + bZ + cZ^2 + dZ^3, where Z ~ N(0,1)
    """
    b2 = b * b
    c2 = c * c
    d2 = d * d
    bd = b * d
    var = b2 + 6*bd + 2*c2 + 15*d2
    skew = 2 * c * (b2 + 24*bd + 105*d2 + 2)
    kurt = 24 * (bd + c2 * (1 + b2 + 28*bd) + 
                 d2 * (12 + 48*bd + 141*c2 + 225*d2))
    return (var, skew, kurt)

def flfunc(b, c, d, skew, kurtosis):
    """
    Given the fleishman coefficients, and a target skew and kurtois
    this function will have a root if the coefficients give the desired skew and kurtosis
    """
    x,y,z = fleishman(b,c,d)
    return (x - 1, y - skew, z - kurtosis)

def flderiv(b, c, d):
    """
    The deriviative of the flfunc above
    returns a matrix of partial derivatives
    """
    b2 = b * b
    c2 = c * c
    d2 = d * d
    bd = b * d
    df1db = 2*b + 6*d
    df1dc = 4*c
    df1dd = 6*b + 30*d
    df2db = 4*c * (b + 12*d)
    df2dc = 2 * (b2 + 24*bd + 105*d2 + 2)
    df2dd = 4 * c * (12*b + 105*d)
    df3db = 24 * (d + c2 * (2*b + 28*d) + 48 * d**3)
    df3dc = 48 * c * (1 + b2 + 28*bd + 141*d2)
    df3dd = 24 * (b + 28*b * c2 + 2 * d * (12 + 48*bd + 
                  141*c2 + 225*d2) + d2 * (48*b + 450*d))
    return np.matrix([[df1db, df1dc, df1dd],
                      [df2db, df2dc, df2dd],
                      [df3db, df3dc, df3dd]])

def newton(a, b, c, skew, kurtosis, max_iter=25, converge=1e-5):
    """Implements newtons method to find a root of flfunc."""
    f = flfunc(a, b, c, skew, kurtosis)
    for i in range(max_iter):
        if max(map(abs, f)) < converge:
            break
        J = flderiv(a, b, c)
        delta = -solve(J, f)
        (a, b, c) = delta + (a,b,c)
        f = flfunc(a, b, c, skew, kurtosis)
    return (a, b, c)


def fleishmanic(skew, kurt):
    """Find an initial estimate of the fleisman coefficients, to feed to newtons method"""
    c1 = 0.95357 - 0.05679 * kurt + 0.03520 * skew**2 + 0.00133 * kurt**2
    c2 = 0.10007 * skew + 0.00844 * skew**3
    c3 = 0.30978 - 0.31655 * c1
    #logging.debug("inital guess {},{},{}".format(c1,c2,c3))
    return (c1, c2, c3)


def fit_fleishman_from_sk(skew, kurt):
    """Find the fleishman distribution with given skew and kurtosis
    mean =0 and stdev =1
    
    Returns None if no such distribution can be found
    """
    if kurt < -1.13168 + 1.58837 * skew**2:
        return None
    a, b, c = fleishmanic(skew, kurt)
    coef = newton(a, b, c, skew, kurt)
    return(coef)

def fit_fleishman_from_standardised_data(data):
    """Fit a fleishman distribution to standardised data."""
    skew = moment(data,3)
    kurt = moment(data,4)
    coeff = fit_fleishman_from_sk(skew,kurt)
    return coeff

def describe(data):
    """Return summary statistics of as set of data"""
    mean = sum(data)/len(data)
    var = moment(data,2)
    skew = moment(data,3)/var**1.5
    kurt = moment(data,4)/var**2
    return (mean,var,skew,kurt)

def generate_fleishman(a,b,c,d,N=100):
    """Generate N data items from fleishman's distribution with given coefficents"""
    Z = norm.rvs(size=N)
    F = a + Z*(b +Z*(c+ Z*d))
    return F




def name_ticker_pair():
    pairs = {"silver":"SI=F", 
             "bac":"BAC", 
             "citi":"C", 
             "corn":"ZC=F", 
             "euro":"EURUSD=X", 
             "gold":"GC=F", 
             "iyr":"IYR", 
             "oil":"CL=F", 
             "pound":"GBPUSD=X", 
             "soybns":"ZS=F", 
             "tr5yr":"^FVX", 
             "tr10yr":"^TNX", 
             "wheat":"ZW=F", 
             "yen":"JPY=X"}
    return pairs


def SimuStockPath(s0: float, alpha: float, sigma: float, T: float, Nsim: int, stratified: bool) -> np.ndarray:
  '''
  s0: spot price
  alpha: instantaneous mean rate
  sigma: volatility
  T: time to maturity
  Nsim: numbers of simuation path
  stratified: using stratified sampling method or not
  '''
  if stratified:
      unif_samples = np.random.uniform(size=(Nsim))
      stratified_samples = (np.arange(1, Nsim + 1) - unif_samples) / Nsim
      z = norm.ppf(stratified_samples)
      np.random.shuffle(z)

  else:
      z = np.random.standard_normal((Nsim))

  prices = np.zeros((Nsim))
  for n in np.arange(Nsim):
      drift = (alpha - 0.5 * sigma**2) * T
      diffusion = sigma * np.sqrt(T) * z[n]

      prices[n] =  s0 * np.exp(drift + diffusion)

  return prices

def EuropeanOptionPrice(s0: float,alpha: float, sigma: float, r: float, T: float, Nsim: int , k: float, call: bool, stratified: bool) -> float:
    '''
    s0: spot price
    alpha: instantaneous mean rate; should equal to risk free rate
    sigma: volatility
    r: risk free rate
    T: time to maturity
    Nsim: numbers of simuation path
    k: strike
    call: option type
    stratified: using stratified sampling method or not
    '''
    price = SimuStockPath(s0, alpha, sigma, T, Nsim, stratified)

    if call:
      payoff = np.maximum(price - k, 0)
    else:
      payoff = np.maximum(k - price, 0)

    pv_payoff = payoff * np.exp(-r * T)

    option_price = np.mean(pv_payoff)

    return option_price