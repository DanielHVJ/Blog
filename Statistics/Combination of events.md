# Combination of events

This chapter is concerned with events which are defined in terms of certain other events A<sub>1</sub>, A<sub>2</sub>, ... A<sub>N</sub>. For example, in the bridge example, A "at least  one player has a complete suit", that is the union of four events, that could occur simultaneously. 

## Union of events

If A<sub>1</sub> and A<sub>2</sub> are two events, then:
$$
A = A_1 \cup A_2 
\\
P(A) = P(A_1) + P(A_2) - P(A_1,A_2)
$$
In symbols this event is:
$$
A = A_1 \cup A_2 \cup ... A_N
$$
But, we know two subscripts are never equal, for the sum of all _p's_ with _r_ subscripts.
$$
S_1 = \sum p_i, 		and 		
\\
S_3 = \sum p_{ijk}
$$
Where the probability of P<sub>1</sub> of the realization of at least one among the events A<sub>1</sub>, A<sub>2</sub>, ... A<sub>N</sub> is given by.
$$
P_1 = S_1 - S_2 + S_3 - S_4 + ...\pm S_N
$$
Then P{E} appears as a contribution to those p<sub>i</sub>, p<sub>ij</sub>, ... p<sub>ijk</sub>... whose subscripts rage from 1 to n. Hence P{E} appears _n_ times as a contribution to S<sub>1</sub>, and (n  2) times as a contribution to S<sub>2</sub>.

```python
## BRIDGE GAME TES

# A has a complete suit
import math

pa = 4/math.comb(52,13)
print(pa)

# Both players A, B have a cmplete suits

pab = 4*3/(math.comb(52,13)*math.comb(39,13))
print(pab)

# For all players

pall = 4*3*2/(math.comb(52,13)*math.comb(39,13)*math.comb(26,13))
print(pall)
```

The probability of 3 players is equal to 4 players, since if 3 players have a complete suit, so does the last player. The probability of some player has a complete suit is:
$$
P_1 = 4p_1 - 6p_{1,2} + 4p_{1,2,3} - p_{1,2,3,4}
$$
using the Stirling's formula we have approximately:
$$
P_1 = \frac14*10^{-10}
$$
In this case,  _P<sub>1</sub>_ is nearly the sum of all probabilities of _A<sub>i</sub>_.

### Application to the classical occupancy problem

Returning to the problem of a random distribution of _r_ ball in _n_ cells, where we seek the probability _p(r,n)_ different ways.

Consider now, _m_ cells that are empty and can be distributed in _Combination of (n, m)_.

The relation among r and n is arbitrary, but if the average number _r/n_ of balls per cell is excessively large, then we cannot expect any empty cells; in this case:
$$
p_0(r,n) \approx 1
\\
p_m(r,n) \approx 0, m = 1
$$
Putting the abbreviation.
$$
ne^{-r/n}= \lambda
$$
Where _lambda_ represents the limit of empty and occupied cells.

If  we consider that _n and r_ tends to be higher than 1 and bigger, so we have that _lambda_ will remains bounded, then we have:
$$
p(m;\lambda) = e^{-\lambda}\frac{\lambda^m} {m!}
$$


  

 

