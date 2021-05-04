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
