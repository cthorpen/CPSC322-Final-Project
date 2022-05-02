import statistics as stats

def compute_slope_intercept(x, y):
    meanx = stats.mean(x)
    meany = stats.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den 
    b = meany - m * meanx
    return m, b 

def compute_correlation_coefficient(x, y):
    num = 0
    for i in range(len(x)):
        num += (x[i] - stats.mean(x)) * (y[i]  -stats.mean(y))
    denom_1 = 0
    denom_2 = 0
    for i in range(len(x)):
        denom_1 += (x[i] - stats.mean(x)) ** 2
        denom_2 += (y[i] - stats.mean(y)) ** 2
    denom = (denom_1 * denom_2) ** (1/2)
    r = num/denom
    return r
    
def compute_covariance(x, y):
    num = 0
    for i in range(len(x)):
        num += (x[i] - stats.mean(x)) * (y[i]  -stats.mean(y))
    return num/len(x)