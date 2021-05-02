# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:33:47 2021

@author: Sabine
"""

# A Python program to print all
# permutations using library function
from itertools import permutations

# Get all permutations of [1, 2, 3]
perm = permutations([1, 2, 3])

# Print the obtained permutations
for i in list(perm):
	print (i)
    
import math
math.comb(5,3)
math.perm(7,5)
