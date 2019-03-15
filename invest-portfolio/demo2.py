import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#模拟生成两个的权重
def weight(n):
    w = np.random.random(n)
    return w/sum(w)
#定义计算收益、风险、夏普率的函数
def portfolio_test(w):
    r1,r2 = 0.08,0.12
    sigma1,sigma2 = 0.12,0.25
    rho1,rho2,rho3,rho4,rho5 = -1,-0.5,0,0.5,1
    p_mean = r1*w[0]+r2*w[1]
    p_var1 = (w[0]**2)*(sigma1**2)+(w[1]**2)*(sigma2**2)+2*w[0]*w[1]*rho1*sigma1*sigma2
    p_var2 = (w[0]**2)*(sigma1**2)+(w[1]**2)*(sigma2**2)+2*w[0]*w[1]*rho2*sigma1*sigma2
    p_var3 = (w[0]**2)*(sigma1**2)+(w[1]**2)*(sigma2**2)+2*w[0]*w[1]*rho3*sigma1*sigma2
    p_var4 = (w[0]**2)*(sigma1**2)+(w[1]**2)*(sigma2**2)+2*w[0]*w[1]*rho4*sigma1*sigma2
    p_var5 = (w[0]**2)*(sigma1**2)+(w[1]**2)*(sigma2**2)+2*w[0]*w[1]*rho5*sigma1*sigma2
    p_sigma1 = np.sqrt(p_var1)
    p_sigma2= np.sqrt(p_var2)
    p_sigma3 = np.sqrt(p_var3)
    p_sigma4 = np.sqrt(p_var4)
    p_sigma5 = np.sqrt(p_var5)
    return w[0],w[1],p_mean,p_sigma1,p_sigma2,p_sigma3,p_sigma4,p_sigma5