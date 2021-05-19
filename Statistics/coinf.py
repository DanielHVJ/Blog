import numpy as np

#probability of heads vs. tails. This can be changed.
probability = .5
#num of flips required. This can be changed.
n = 10000

#initiate array
play_1 = np.arange(n)
play_2 = np.arange(n)

def coinFlip(p):    
    #perform the binomial distribution (returns 0 or 1)    
    result = np.random.binomial(1,p) 
    #return flip to be added to numpy array    
    return result

for i in range(0, n):    
    play_1[i] = coinFlip(probability)    
    # i+=1
    play_2[i] = coinFlip(probability)    
    i+=1

#print results
print("probability is set to ", probability)
print("Tails = 0, Heads = 1: ", play_1)
#Total up heads and tails for easy user experience 
print("Head Count: ", np.count_nonzero(play_1 == 1))
print("Tail Count: ", np.count_nonzero(play_1== 0))

print("Head Count: ", np.count_nonzero(play_2 == 1))
print("Tail Count: ", np.count_nonzero(play_2== 0))

x = np.arange(1,n+1) 
from matplotlib import pyplot as plt 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.bar(x,play_1) 
plt.show()

head_1=np.cumsum(play_1==1)
tail_1=np.cumsum(play_1==0)

np.where(head_1 == tail_1)

play_1 = np.where(play_1==0, -1, play_1)

plt.figure(figsize=(17,5))
plt.plot(x[:100],play_1[:100],'--.',marker='^') 
plt.show()

from scipy.interpolate import make_interp_spline, BSpline
# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(1, 100, 70) 
spl = make_interp_spline(x, play_1)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth,'-.', color='g',marker='o')
plt.show()


## 2
import math

n = 20
k = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
k1 = (np.linspace(0,1,21))
distr = [0]*21
out_arc = [0]*21

def arc(k):    
    #perform the binomial distribution (returns 0 or 1)    
    result = (math.comb(2*n,k))/(2**2*n) 
    #return flip to be added to numpy array    
    return result

for i in k:    
    distr[i] = (math.comb(2*n,i))/(2**(2*n))     
    # out_arc[i] = np.sin(i)
    i+=1

out_arc[0] = np.arcsin(k1[0])
out_arc[1] = np.arcsin(k1[1])
out_arc[2] = np.arcsin(k1[2])
out_arc[3] = np.arcsin(k1[3])
out_arc[4] = np.arcsin(k1[4])
out_arc[5] = np.arcsin(k1[5])
out_arc[6] = np.arcsin(k1[6])
out_arc[7] = np.arcsin(k1[7])
out_arc[8] = np.arcsin(k1[8])
out_arc[9] = np.arcsin(k1[9])
out_arc[10] = np.arcsin(k1[10])
out_arc[11] = np.arcsin(k1[11])
out_arc[12] = np.arcsin(k1[12])
out_arc[13] = np.arcsin(k1[13])
out_arc[14] = np.arcsin(k1[14])
out_arc[15] = np.arcsin(k1[15])
out_arc[16] = np.arcsin(k1[16])
out_arc[17] = np.arcsin(k1[17])
out_arc[18] = np.arcsin(k1[18])
out_arc[19] = np.arcsin(k1[19])
out_arc[20] = np.arcsin(k1[20])

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(k1, out_arc, color = 'pink', alpha = 0.5)
ax1.plot(k1, distr,color = 'b', alpha = 0.8)
plt.show()

np.arcsin(0.05)

