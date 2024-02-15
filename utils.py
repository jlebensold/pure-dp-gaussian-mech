import numpy as np
from functools import partial
from autodp import mechanism_zoo as mz
import mpmath as mp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

# Differential Privacy Analysis -- Exante
def gnmax_exante_mvn(sigma:float=0.08, d:int=3, GS=1/1000, delta=1e-5, **kwargs):
    """Privacy Loss for Report Noisy Max under the circumstance when:
    - User’s contribute to every query dimension
    - Query are statistical and bounded, such that:
    - Each dimension reports a value between 0 and 1.
    - Noise is Gaussian
    
    GS_q = \sqrt{d} / N
    
    Args:
        sigma (float, optional): 
        d (int, optional): dimensions. Defaults to 3.
        GS (int): Global Sensitivity.

    Returns:
        float: (eps, delta) privacy loss
    """
    GS_q = np.sqrt(d) * GS 
    return mz.ExactGaussianMechanism(sigma=sigma/GS_q).get_approxDP(delta)

def abovethreshold_exante(sigmaX:float=0.08, sigmaZ:float=1., rho:float=0.5, GS:float=1, delta=1/1000, c=1, **kwargs):
    return mz.GaussianSVT_Mechanism(dict(
        c=c,
        sigma=sigmaX,
        sigma_nu=sigmaZ,
        Delta=GS,
        margin=rho
    ), rdp_c_1=True).get_approxDP(delta)

# Differential Privacy Analysis -- Expost

def mpmath_mvn(d, sigma, y, dps=50):
    """
    Compute the multivariate normal density function using numerical integration.

    This function evaluates the multivariate normal density function using numerical integration
    provided by the mpmath library.

    Parameters:
        d (int): Dimensionality of the multivariate normal distribution.
        sigma (float): Standard deviation of the distribution.
        y (float): Value(s) at which to evaluate the density function.
        dps (int, optional): Digits of precision for the calculation. Defaults to 50.

    Returns:
        float: The computed log-density value.
    """
    mp.mp.dps = dps
    def F(z, sigma, y, d):
        phi = mp.npdf(z, sigma=sigma) 
        p = mp.ncdf(z + y, sigma=sigma)
        return phi * mp.power(p, d)

    f = partial(F, sigma=sigma, y=y, d=d)
    return mp.log(mp.quad(f, [-mp.inf, mp.inf]))


def gnmax_expost_mvn(sigma, d, GS, a=0., b=1., dps=100, **kwargs):
    """computes the ex-post privacy loss for GNMax

    Args:
        sigma (float): noise multiplier. 
        d (int): dimensionality the query. 
        GS (float): global sensitivity. This value is multiplied by two in order to account for the upper bound between two queries.
        a (float): lb. Defaults to 0.
        b (float): ub. Defaults to 1.
        dps (int, optional): digits of precision. Defaults to 100.

    Returns:
        float: GNMax ex-post privacy loss
    """
    # set the accuracy
    mp.mp.dps = dps
    y = a - b
    alpha = mpmath_mvn(d=d - 1, sigma=sigma, y=y + 2*GS, dps=dps)
    beta = mpmath_mvn(d=d - 1, sigma=sigma, y=y, dps=dps)
    result = float(alpha - beta)
    return result

def abovethreshold_mvn(Delta, t, boty, topy, rho, sigmaX=2, sigmaZ=1, **kwargs):
    """
    Compute the probability of a multivariate normal distribution being above a threshold.

    This function calculates the probability of a multivariate normal distribution being above a specified threshold
    using numerical integration provided by the mpmath library.

    Parameters:
        Delta (float): Threshold value.
        t (float): Stopping time.
        boty (float): Value of the lower threshold.
        topy (float): Value of the upper threshold.
        rho (float): Threshold.
        sigmaX (float, optional): Standard deviation of the distribution X. Defaults to 2.
        sigmaZ (float, optional): Standard deviation of the distribution Z. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the integration routine.

    Returns:
        float: The computed probability value.
    """
    def F(x):
        bot_part = mp.ncdf(sigmaX * x + rho - boty + Delta, sigma=sigmaZ)
        top_part = mp.ncdf(- sigmaX * x - rho + topy + Delta, sigma=sigmaZ)
        phi = mp.npdf(x, mu=0)
        return phi*mp.exp((t) * mp.log(bot_part)) * top_part
    return mp.quad(F, [-mp.inf, mp.inf])

def abovethreshold_expost(sigmaX, sigmaZ, t, rho, GS, boty, topy,  dps=20, **kwargs):
    """
    Compute the ex-post above threshold

    This function calculates the exponentiated posterior probability of a multivariate normal distribution
    being above a specified threshold at time t.

    Parameters:
        sigmaX (float): Standard deviation of the distribution X.
        sigmaZ (float): Standard deviation of the distribution Z.
        t (int): Stopping time.
        rho (float): Threshold.
        GS (float): Global Sensitivty.
        boty (float): Value of the lower threshold.
        topy (float): Value of the upper threshold.
        dps (int, optional): Digits of precision for the calculation. Defaults to 20.

    Returns:
        float: The computed exponentiated posterior value.
    """
    # set the accuracy
    mp.mp.dps = dps

    alpha = abovethreshold_mvn(Delta=GS, t=t - 1, boty=boty, topy=topy, rho=rho, sigmaX=sigmaX, sigmaZ=sigmaZ, dps=dps)
    beta = abovethreshold_mvn(Delta=0, t=t - 1, boty=boty, topy=topy, rho=rho, sigmaX=sigmaX, sigmaZ=sigmaZ, dps=dps)
    return float(mp.log(alpha) - mp.log(beta))

# Monte Carlo Estimation

def mc_cond_mvn_stable(d:int, sigma:float, y:float, n_samples:int=100_000_000, samples=None):
    """
    Generate a Monte Carlo estimate of the density using the standard survival function.

    Parameters:
        d (int): Dimensionality of the multivariate normal distribution.
        sigma (float): Standard deviation of the distribution.
        y (float): Value(s) at which to evaluate the density function.
        n_samples (int, optional): Number of samples to generate. Defaults to 100000000.
        samples (np.ndarray, optional): Pre-generated samples to use. Defaults to None.

    Returns:
        np.ndarray: Array of computed density values.
    """
    if samples is None:
        samples = np.random.normal(0, sigma, n_samples)

    return d * np.log(norm.sf((y - samples)/sigma))

def mc_cond_expost_stable(d, sigma, Delta, y=1, n_samples=None):
    """
    Compute the ex-post privacy for GNMax using Monte Carlo estimates.

    Parameters:
        d (int): Dimensionality of the multivariate normal distribution.
        sigma (float): Standard deviation of the distribution.
        Delta (float): Value of Delta.
        y (float, optional): Value(s) at which to evaluate the density function. Defaults to 1.
        dps (int, optional): Digits of precision for the calculation. Defaults to 25.
        n_samples (int, optional): Number of samples to generate. Defaults to None.

    Returns:
        float: The estimated GNMax privacy loss.
    """
    a = mc_cond_mvn_stable(d, sigma, y - 2*Delta, samples=np.random.normal(0, sigma, n_samples))
    b = mc_cond_mvn_stable(d, sigma, y, samples=np.random.normal(0, sigma, n_samples))
    return logsumexp(a) - logsumexp(b)

def calc_sigmas(sigma):
    ''' Calculate the noise multiplier for thresholds and queries resp.
    returns:
        (sigmaX, sigmaZ)
    '''
    return sigma, np.sqrt(3) * sigma