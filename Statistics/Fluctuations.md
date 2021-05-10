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

We set k value where the position where the accumulated sum is equal in head and tails.

Symmetry implies that the inequalities _k > n/2_ and _k < n/2_ are equally likely.



### Arc sine law for last visits

The probability that up to and including epoch _2n_ the last visit to the origin occurs at epoch _2k_ is given by:
$$
\alpha_{2k,2n}=\frac{\binom{2n}{k}}{2^{2n}}
$$
We see that as k increases also increases the probability, also we can see that it is very similar to an _arc sine distribution_ of _k_ values.