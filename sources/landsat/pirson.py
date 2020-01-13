import numpy as np
import math

def pearson(x1, x2):
    if len(x1) != len(x2):
        return 0
    avgX1 = np.average(x1)
    avgX2 = np.average(x2)
    sumX1=sum([(i - avgX1)**2.0 for i in x1])
    sumX2=sum([(i - avgX2)**2.0 for i in x2])
    a = 0
    for i in range(0, len(x1)):
        a += (x1[i] - avgX1)* (x2[i] - avgX2)
    p = 0
    if(sumX1*sumX2 != 0):
        p = a / (math.sqrt(sumX1*sumX2))
    return p