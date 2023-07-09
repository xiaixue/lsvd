import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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

  def eq(beta, final= False):
    beta_1 = beta[0]
    beta_2 = beta[1]
    beta_0 = beta[2]
    sigma = np.zeros(n)
    for i in range(n):
      try:
        sigma_i = bisection(virtual_displacement, l[i] - r[i], l[i], error= 1e-5, args=[r[i], l[i], beta_1, beta_2])
      except:
        sigma_i = max(l) ** 2
      sigma[i] = sigma_i
    
    tau = (r ** 2 - (sigma - l) ** 2) ** 0.5

    tau_hat = beta_1 * np.log(sigma + beta_2) + beta_0

    sum_0 = np.sum((tau - tau_hat) ** 2)

    if final == True:
      return beta
    return sum_0

  p = opt.minimize(eq, x0= [min(r), -1, 0], method="powell", bounds= ([0, max(r)], [0, max(l)], [-max(l)*5, max(l)*5] ), tol= 1e-5)
  return p.x

def parabolic_LSVD(r, l):
  n = len(l)

  def virtual_displacement(sigma_i, args):
    sum_0 = - args[2] / (2 * sigma_i ** 0.5) + (-sigma_i + args[1]) / (args[0] ** 2 - (sigma_i - args[1]) ** 2) ** 0.5
    return sum_0

  def eq(beta, final= False):
    beta_1 = beta[0]
    beta_0 = beta[1]

    sigma = np.zeros(n)
    for i in range(n):
      try:
        sigma_i = bisection(virtual_displacement, l[i] - r[i], l[i], error= 1e-5, args=[r[i], l[i], beta_1])
      except:
        return max(l) ** 2
      sigma[i] = sigma_i
    
    tau = (r ** 2 - (sigma - l) ** 2) ** 0.5
    tau_hat = beta_1 * sigma ** 0.5 + beta_0
    sum_0 = np.sum((tau - tau_hat) ** 2)

    if final == True:
      return beta
    return sum_0

  p = opt.minimize(eq, x0= [min(r), 0], method="powell", bounds= ([0, max(r)], [-max(l), max(l)]), tol= 1e-5)

  return p.x

def power_LSVD(r, l):
  n = len(l)
  
  def virtual_displacement(sigma_i, args):
    sum_0 = (-sigma_i + args[1]) / ((args[0] ** 2 - (sigma_i - args[1]) ** 2) ** 0.5) - args[2] * args[4] * (sigma_i + args[3]) ** (args[4] - 1)
    return sum_0

  def eq(beta, final= False):
    beta_1 = beta[0]
    beta_2 = beta[1]
    beta_3 = beta[2]
    beta_0 = beta[3]

    sigma = np.zeros(n)
    for i in range(n):
      arguments = [r[i], l[i], beta_1, beta_2, beta_3, beta_0]
      try:
        sigma_i = bisection(virtual_displacement, l[i] - r[i], l[i], error= 1e-5, args=arguments)
      except:
        sigma_i = max(l) ** 2
      sigma[i] = sigma_i
    
    tau = (r ** 2 - (sigma - l) ** 2) ** 0.5

    tau_hat = beta_1 * (sigma + beta_2) ** beta_3 + beta_0

    sum_0 = np.sum((tau - tau_hat) ** 2)
    if final == True:
      return beta
    return sum_0
  
  p = opt.minimize(eq, x0= [min(r), 0, 0.5, 0], method="powell", bounds= ([0, max(r)], [0, max(l)], [0.01, 0.999], [-max(l), max(l)]), tol= 1e-5)

  return p.x

def polynomic_LSVD(r, l, deg):
  n = len(l)
  coeffs = [0] * (deg+1)
  
  def virtual_displacement(sigma_i, args):
    der_cir = (-sigma_i + args[1]) / ((args[0] ** 2 - (sigma_i - args[1]) ** 2) ** 0.5)
    der_pol = 0
    for i in range(1, deg+1):
      der_pol += i * args[2+i] * sigma_i ** (i-1)
    return der_cir - der_pol

  def eq(beta):
    sum_0 = 0
    for i in range(n):
      arguments = [r[i], l[i]] + list(beta)
      try:
        sigma_i = bisection(virtual_displacement, l[i] - 0.99*r[i], l[i] + 0.99*r[i], error= 1e-5, args=arguments)
      except Exception as e:
        sigma_i = l[i]
        sum_0 += max(l) ** 2

      tau = (r[i] ** 2 - (sigma_i - l[i]) ** 2) ** 0.5
      s_hat = 0
      for i in range(deg+1):
        s_hat += beta[i] * sigma_i ** i
      sum_0 += (tau - s_hat) ** 2
    return sum_0
  
  p = opt.minimize(eq, x0= coeffs, method="powell", tol= 1e-5)
  return p.x

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

def CTPAC(r,l):
  n = len(l)
  x_coords = []
  y_coords = []
  
  for i in range(n):
    if i == n-1:
      return x_coords, y_coords
    
    radii_diff = r[i+1] - r[i]
    centr_diff = l[i+1] - l[i]
    theta = np.arcsin(radii_diff/centr_diff)
    
    x1 = l[i] - r[i] * np.sin(theta)
    x2 = l[i+1] - r[i+1] * np.sin(theta)
    y1 = r[i] * np.cos(theta)
    y2 = r[i+1] * np.cos(theta)
    x_coords.append(x1)
    x_coords.append(x2)
    y_coords.append(y1)
    y_coords.append(y2)
  
  return x_coords, y_coords


def LSVD(s_1, s_3, typef):
  x, y = rad_lamb(s_3, s_1)
  sigmoid, tauoid = mohr_circle(x, y)

  x_1 = np.linspace(0.001, max(x+y)*1.5, 1000)
  if typef == "linear":
    a = linear_LSVD(x,y)
    plt.plot(x_1, np.tan(a[0] * 2 * np.pi / 360) * x_1 + a[1], linewidth= 2, color= "black", linestyle= "-", label= "Linear LSVD")
    print(f"s= tan({a[0]})  σ + {a[1]}")
  elif typef == "log":
    d = log_LSVD(x,y)
    y_3 = d[0] * np.log(x_1 + d[1]) + d[2]
    plt.plot(x_1, y_3, linewidth= 2, color= "black", linestyle= "-", label= "Log LSVD")	
    print(f"s= {d[0]}  ln(σ + {d[1]}) + {d[2]}")
  elif typef == "power":
    d = power_LSVD(x,y)
    y_3 = d[0] * (x_1 + d[1]) ** d[2] + d[3]
    plt.plot(x_1, y_3, linewidth= 2, color= "black", linestyle= "-", label= "Power LSVD")	
    print(f"s= {d[0]} (σ + {d[1]}) ^ {d[2]}+ {d[3]}")
  elif typef == "parabolic":
    d = parabolic_LSVD(x,y)
    y_3 = d[0] * x_1 ** 0.5 + d[1]
    plt.plot(x_1, y_3, linewidth= 2, color= "black", linestyle= "-", label= "Parabolic LSVD")	
    print(f"s= {d[0]}  σ^0.5 + {d[1]}")
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
  typef : Type of model, linear, log, parabolic, power
  """


  s_3 = [50.9, 203.9, 407.8, 815.7, 917.7, 1019.7]     
  s_1 = [1019.7, 1570.3, 2253.5, 3293.6, 3528.2, 3681.1] 
  
  #s_3 = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  #s_1 = [60, 100, 122, 154, 193, 221, 253, 275, 310, 323, 346, 361]
  
  LSVD(s_1, s_3, typef = "linear")
  LSVD(s_1, s_3, typef = "log")
  LSVD(s_1, s_3, typef = "parabolic")
  LSVD(s_1, s_3, typef = "power")
