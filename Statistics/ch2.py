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
math.perm(4,2)

def event_probability(event_outcomes, sample_space):
    probability = (event_outcomes / sample_space) * 100
    return round(probability, 1)

# Sample Space
cards = 52

# Determine the probability of drawing a heart
hearts = 13
heart_probability = event_probability(hearts, cards)

queen_of_hearts = 1
queen_of_hearts_probability = event_probability(queen_of_hearts, cards)

print(str(heart_probability) + '%')
print(str(queen_of_hearts_probability) + '%')

n = 52
k = 2

# Determine permutations and print result
Permutations = math.factorial(n) / math.factorial(n-k)
print(Permutations)

Combinations = Permutations / math.factorial(k)
print(Combinations)

#  Independent versus Dependent Events

cards = 52
cards_drawn = 1 
cards = cards - cards_drawn 

aces = 4
ace_probability1 = event_probability(aces, cards)

# Determine the probability of drawing an Ace after drawing an Ace on the first draw
aces_drawn = 1
aces = aces - aces_drawn
ace_probability2 = event_probability(aces, cards)

# Print each probability
print(ace_probability1)
print(ace_probability2)

#  Multiple Events

cards = 52

# Calculate the probability of drawing a heart or a club
hearts = 13
clubs = 13
heart_or_club = event_probability(hearts, cards) + event_probability(clubs, cards)

# Calculate the probability of drawing an ace, king, or a queen
aces = 4
kings = 4
queens = 4
ace_king_or_queen = event_probability(aces, cards) + event_probability(kings, cards) + event_probability(queens, cards)

print(heart_or_club)
print(ace_king_or_queen)


cards = 52

# Calculate the probability of drawing a heart or an ace
hearts = 13
aces = 4
ace_of_hearts = 1

heart_or_ace = event_probability(hearts, cards) + event_probability(aces, cards) - event_probability(ace_of_hearts, cards)

# Calculate the probability of drawing a red card or a face card
red_cards = 26
face_cards = 12
red_face_cards = 6

red_or_face_cards = event_probability(red_cards, cards) + event_probability(face_cards, cards) - event_probability(red_face_cards, cards)

print(round(heart_or_ace, 1))
print(round(red_or_face_cards, 1))


# Sample Space
cards = 52

# Outcomes
aces = 4

# Probability of one ace
ace_probability = aces / cards

# Probability of two consecutive independant aces 
two_aces_probability = ace_probability * ace_probability

# Two Ace Probability Percent Code
two_ace_probability_percent = two_aces_probability * 100
print(round(two_ace_probability_percent, 1))

import math

n=10
k=5
c = math.comb(10,5)
# Combinations = Permutations / math.factorial(k)
p = c*(n**-k)
print(p)

floors=10
pas=7
c = math.comb(10,7)

p = c*(floors**-pas)
print(p)

p=2
q=4

math.factorial(p+q)/(math.factorial(p)*math.factorial(q))
math.comb(6,4)

math.factorial(52)/math.factorial(13)**4

(math.factorial(4)*math.factorial(48)*13**4)/math.factorial(52)


math.factorial(10)/(3**6*6**10)

## Hyperdistribution

n = 25000
n1 = 5000
r = 200
k = 18

qk = (math.comb(r,k)*math.comb(n-r,n1-k))/math.comb(n,n1)
qk


n1 = 1000
r = 1000
k = 100

n = int(n1*r/k)
n

qk = (math.comb(r,k)*math.comb(n-r,n1-k))/math.comb(n,n1)
qk

n=52
n1=13


pk = (math.comb(n1,5)*math.comb(n1,4)*math.comb(n1,3)*math.comb(n1,1))/math.comb(n,n1)
pk
