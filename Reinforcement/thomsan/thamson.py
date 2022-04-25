import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data=pd.read_csv("Ads_CTR_Optimisation.csv")

N=data.shape[0]
d=data.shape[1]

# print(data.shape[0],data.shape[1],data.shape)

numbers_of_rewar_1=np.zeros(10)
numbers_of_rewar_0=np.zeros(10)
ad_selected=[]
total_reward=0
# print(numbers_of_rewar_1)

for i in range(N):
    ad=0
    max_random=0
    for j in range(d):
        rand_beta=random.betavariate(numbers_of_rewar_1[j]+1,numbers_of_rewar_0[j]+1)
        if rand_beta>max_random:
            max_random=rand_beta
            ad=j
    ad_selected.append(ad)
    reward=data.values[i,ad]
    if reward==0 : numbers_of_rewar_0[ad]+=1
    else :  numbers_of_rewar_1[ad]+=1
    total_reward+=1


plt.hist(ad_selected)
plt.show()