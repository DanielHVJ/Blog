# Fluctuations in coin tossing

The ideal coin-tossing game will be described in the terminology of random walks which is better suited for generalizations. For the geometric description it is convenient to pretend that tossings are performed at a uniformed rate so that the _nth_ trial occurs at epoch _p_.

We denote individual step generically by _X1, X2, .... Xn_, and the positions by _S1 and S2_. Thus

_Sn = X1 + X2 + .... Xn, and S0 = 0_

For the probability we write
$$
p_{n,r}=P[S_n=r]=\binom{n}{\frac{n+r}{2}}2^{-n} \\
u_{2v}=\binom{2v}{v}2^{-2v}
$$
The last binomial coefficient could be expressed as Stirling's formula
$$
u_{2v}\approx\frac{1}{\sqrt{v\pi}}
$$

## Last visit and long leads

In a long coin-tossing the law of averages ensure that in the game each player will be on the winning side for about half of the time.

In this experiment we choose at random and observe the number  of the last trial at which the accumulated number os head and tails were equal, denoted by _2k_ (0 < k < n).

Symmetry implies that the inequalities _k > n/2_ and _k < n/2_ are equally likely.

```python
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
```

```php
probability is set to  0.5
Tails = 0, Heads = 1:  [0 1 1 ... 0 0 0]
Player One:
Head Count:  4961
Tail Count:  5039

Player Two
Head Count:  4920
Tail Count:  5080
```

Finding k for player One

```pascal
[   1,   47,   53,   55, 2725, 2727, 2729, 3381, 3441, 3443, 3463,
        3467, 3469, 3477, 3479, 3483, 3485, 3487, 3489, 3491, 3503, 3505,
        3507, 3509, 3517, 3519, 3521, 3523, 3525, 3527, 3529, 5325, 5329,
        5337, 5341, 5393, 5445, 5451, 5453, 5455, 5457, 5459, 5481, 5485,
        5489, 5491, 5493, 5723, 5725, 5729, 5731, 5733, 5763, 5769, 6199,
        6201, 6205, 6217, 6221, 6227, 6253, 6273, 6275, 6279, 6347, 6361,
        6363]
```

K for Player Two

```pascal
[   3,    5,   53,   69,   71,  101,  103,  105,  107,  109,  111,
         113,  121,  225,  229,  231,  251,  253,  257,  277,  555,  557,
         559,  561,  995,  997,  999, 1001, 1003, 1007, 1023, 1033, 1035,
        1037, 1039, 1041, 1043, 1049, 1349, 1351, 1499, 1503, 1507, 1511,
        1513, 1515, 1517]
```

As we can see the equalization appears for more times for player one (67 cases), while player two has a small number of coincidences (cumulative sum of tails = head) only 47.

Concluding, we can see that we cannot judge the luck of Player 1 and 2 equally,  even at random events one has the chances to succeed more.

### Arc sine law for last visits

The probability that up to and including epoch _2n_ the last visit to the origin occurs at epoch _2k_ is given by:
$$
\alpha_{2k,2n}=\frac{\binom{2n}{k}}{2^{2n}}
$$
We see that as k increases also increases the probability, also we can see that it is very similar to an _arc sine distribution_ of _k_ values.