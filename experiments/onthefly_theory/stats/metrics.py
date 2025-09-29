from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y, p): return float(mean_squared_error(y, p, squared=False))
def mae(y, p):  return float(mean_absolute_error(y, p))
