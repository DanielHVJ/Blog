import numpy as np

#probability of heads vs. tails. This can be changed.
probability = 0.5
#num of flips required. This can be changed.
n = 100

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
plt.title("Coin tossing") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.bar(x,play_1) 
plt.show()

head_1=np.cumsum(play_1==1)
tail_1=np.cumsum(play_1==0)

k_1 = np.where(head_1 == tail_1)
(np.array(k_1).size)

head_2=np.cumsum(play_2==1)
tail_2=np.cumsum(play_2==0)

k_2 = np.where(head_2 == tail_2)
(np.array(k_2).size)

k_f = np.where(head_1==head_2)

play_1 = np.where(play_1==0, -1, play_1)


plt.figure(figsize=(17,5))
plt.plot(x[:100],play_1[:100],'--.',marker='^') 
plt.show()

play_2 = np.where(play_2==0, -1, play_2)
plt.figure(figsize=(17,5))
plt.plot(x[400:500],play_2[400:500],'g--.',marker='') 
plt.show()

plt.figure(figsize=(17,5))
plt.title('Equalization Player One') 
plt.plot(k_1,k_1,'-',marker='D',markersize=12) 
plt.show()

plt.figure(figsize=(17,5))
plt.title('Equalization Player Two')  
plt.plot(k_2,k_2,'o',marker='P',markersize=12) 
plt.show()

from scipy.interpolate import make_interp_spline, BSpline
# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(1, 100, 70) 
spl = make_interp_spline(x, play_1)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth,'-.', color='g',marker='o')
plt.show()


 
# Iterate each element in list
# and add them in variale total
total = [0]*9
for ele in range(1, len(play_1)):
    total[ele]= play_1[ele+1] - play_1[ele]
    ele+=1

total_2 = [0]*9
for ele in range(1, len(play_2)):
    total_2[ele]= play_1[ele+1] - play_1[ele]
    ele+=1

print('Positive side changes',total.count(1))
print('Negative side changes',total.count(-1))

import collections
ps1 = collections.Counter(play_1)[0]
pn1 = collections.Counter(play_1)[1]

ps2 = collections.Counter(play_2)[0]
pn2 = collections.Counter(play_2)[1]


        

## 2
import math

n = 20
k = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
k1 = (np.linspace(0,1,21))
distr = [0]*21
out_arc = [0]*21

out_array1 = np.sin(k1)
out_array2 = np.arcsin(k1)


def arc(k):    
    #perform the binomial distribution (returns 0 or 1)    
    result = (math.comb(2*n,k))/(2**2*n) 
    #return flip to be added to numpy array    
    return result

for i in (np.int64((k))):    
    distr[i] = (math.comb(2*n,i))/(2**(2*n))     
    out_arc[i] = np.sin(i)
    i+=1


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(k1, out_array1, color = 'pink', alpha = 0.5)
ax1.bar(k1, distr,color = 'b', alpha = 0.2, width=.1)
ax2.plot(k1, out_array2,color = 'g', alpha = 0.8)
plt.show()

# 


from astropy.table import QTable, Table, Column
from astropy import units as u

t = QTable()
t['k'] = k
t['a'] = distr 
# t['c'] = ['x', 'y']
type(t['a'])
t
