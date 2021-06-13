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