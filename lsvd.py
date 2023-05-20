import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit

def mohr_circle(r, l):
  sigma = np.linspace(l-r, l+r, 1000)
  tau = (r ** 2 - (sigma - l) ** 2) ** 0.5
  return sigma, tau

def rad_lamb(sigma_3_, sigma_1_):
  r_, l_ = list(), list()
  for i, _ in enumerate(sigma_3_):
    r = (sigma_1_[i] - sigma_3_[i]) / 2
    l = sigma_3_[i] + r
    r_.append(r)
    l_.append(l)
  return np.array(r_), np.array(l_)

def linear_LSVD(r, l):
  r_mean = np.mean(r)
  l_mean = np.mean(l)
  
  def eq_12(beta_1):
    p1 = (beta_1 ** 2 + 1) ** 0.5
    beta_0 = np.mean(r) * p1 - beta_1 * np.mean(l)
    sum_0 = beta_1 * np.sum(l ** 2 + r ** 2) + len(r) * beta_0 * (l_mean - r_mean * beta_1 / (beta_1 ** 2 + 1) ** 0.5) - (2 * beta_1 ** 2 + 1) * (np.sum(r*l)) / ((beta_1 ** 2 + 1) ** 0.5)
    
    return sum_0
  
  beta_1 = opt.newton(eq_12, x0= 1, tol= 1e-10)
  p1 = (beta_1 ** 2 + 1) ** 0.5
  beta_0 = r_mean * p1 - beta_1 * l_mean
  return np.arctan(beta_1) * 360 / 2 / np.pi, beta_0

def log_LSVD(r, l):
  n = len(l)

  def virtual_displacement(sigma_i, args):
    sum_0 = - args[2] / (sigma_i + args[3]) + (-sigma_i + args[1]) / (args[0] ** 2 - (sigma_i - args[1]) ** 2) ** 0.5
    return sum_0
  
  def loga(sigma, A, B, C):
    return A * np.log(sigma + B) + C

  def eq(beta_1, final= False):
    sigma = np.zeros(n)
    for i in range(n):
      sigma_i = bisection(virtual_displacement, l[i] - r[i], l[i], error= 1e-5, args=[r[i], l[i], beta_1[0], beta_1[1]])
      sigma[i] = sigma_i
    
    tau = (r ** 2 - (sigma - l) ** 2) ** 0.5
    sol = curve_fit(loga, sigma, tau, method="lm")

    beta = sol[0]

    tau_hat = beta[0] * np.log(sigma + beta[1]) + beta[2]

    sum_0 = np.sum((tau - tau_hat) ** 2)

    if final == True:
      return beta
    return sum_0

  p = opt.minimize(eq, x0= [min(r), 0, 0], method="powell", bounds= ([0, max(r)], [1-min(l-r), max(l)], [-max(l)*5, max(l)*5] ), tol= 1e-5)
  beta = eq(p.x, final= True)
  return beta


def bisection(f, a, b, error=1e-6, args= None):
  if f(a, args) * f(b, args) > 0:
    raise ValueError("f(a) and f(b) must have opposite signs.")
  while (b - a) / 2 > error:
    c = (a + b) / 2
    if f(c, args) == 0:
      return c
    elif f(a, args) * f(c, args) < 0:
      b = c
    else:
      a = c
  return (a + b) / 2

def LSVD(s_1, s_3, typef):
  x, y = rad_lamb(s_3, s_1)
  sigmoid, tauoid = mohr_circle(x, y)

  x_1 = np.linspace(0.001, max(x+y)*1.5, 1000)
  if typef == "linear":
    a = linear_LSVD(x,y)
    plt.plot(x_1, np.tan(a[0] * 2 * np.pi / 360) * x_1 + a[1], linewidth= 2, color= "black", linestyle= "-", label= "Linear LSVD")
    print(f"τ= tan({a[0]})  σ + {a[1]}")
  elif typef == "log":
    d = log_LSVD(x,y)
    y_3 = d[0] * np.log(x_1 + d[1]) + d[2]
    plt.plot(x_1, y_3, linewidth= 2, color= "black", linestyle= "-", label= "Log LSVD")	
    print(f"τ= {d[0]}  ln(σ + {d[1]}) + {d[2]}")
  else:
    raise TypeError("Not a valid model")


  plt.plot(sigmoid, tauoid, color= "gray", linewidth= 1, alpha= 1)

  plt.legend()
  plt.xlim(0, max(x+y))
  plt.ylim(0, max(x)/2+max(y)/2)
  plt.show()
  return 0

if __name__ == "__main__":
  """
  s3 : Confinement pressure
  s1 : Confinement pressure + Deviator stress
  typef : Type of model, linear or log
  """


  s3 = [50.9, 203.9, 407.8, 815.7, 917.7, 1019.7]     
  s1 = [1019.7, 1570.3, 2253.5, 3293.6, 3528.2, 3681.1] 
  LSVD(s1, s3, typef = "log")