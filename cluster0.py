#!/usr/bin/env python2

import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

movie = 'thor'

matrix = np.loadtxt('/home/wizkraft/openface-master/demos/matrix/matrix_'+movie+'.txt', usecols=range(128))


print matrix.shape

n = len(matrix[:,127])


'''
Frequency[i] stores the frequency of appearance of Person i
ClusterCenter[i] stores the centre/first image of Person i
IndexOfThePersonAssigned[i] stores the number of the person assigned to ith image
''' 


ClusterCenter = np.zeros(n+2)
IndexOfThePersonAssigned = np.zeros(n+1)
Frequency = np.zeros(n+1)



#initial = int(n/2)
initial = 0
IndexOfThePersonAssigned[initial]=1
ClusterCenter[1]=1


k=1
# k stores the number of persons discovered at a particular moment/iteration


for i in range(initial,n-1):

    DistanceVector = matrix[i,:] - matrix[i+1,:]

    if np.dot(DistanceVector,DistanceVector) < 1:
        IndexOfThePersonAssigned[i+1]=IndexOfThePersonAssigned[i]
    else:
        min = 5
        index = 1
        for j in range(1,k+1):
            DistanceVector = matrix[int(ClusterCenter[j]),:] - matrix[i+1,:]
            if np.dot(DistanceVector,DistanceVector) < min:
                min = np.dot(DistanceVector,DistanceVector)
                index = int(ClusterCenter[j])
        if min < 0.95:
            IndexOfThePersonAssigned[i+1] = IndexOfThePersonAssigned[index]
        else:
            IndexOfThePersonAssigned[i+1]=k+1
            ClusterCenter[k+1]=i+1
            k=k+1


s = k   # number of different characters in the movie

for i in range(1,n):
    Frequency[int(IndexOfThePersonAssigned[i])] = Frequency[int(IndexOfThePersonAssigned[i])]+1


max1 = 0
max2 = 0
max3 = 0
max4 = 0

index1 = 0
index2 = 0
index3 = 0
index4 = 0

for i in range(1,s+1):
    print Frequency[i],ClusterCenter[i]
    if max1 < Frequency[i]:
        max1=Frequency[i]
        index1 = i

for i in range(1,s+1):
    if i==index1:
        continue
    if max2 < Frequency[i]:
        max2=Frequency[i]
        index2 = i

for i in range(1,s+1):
    if i == index1 or i == index2:
        continue
    if max3 < Frequency[i]:
        max3=Frequency[i]
        index3 = i

for i in range(1,s+1):
    if i==index1 or i==index2 or i==index3:
        continue
    if max4 < Frequency[i]:
        max4=Frequency[i]
        index4 = i




print ClusterCenter[int(index1)]
print ClusterCenter[int(index2)]
print ClusterCenter[int(index3)]
print ClusterCenter[int(index4)]


frequencies = [0] * s
i=0

for i in range(0,s-1):
    frequencies[i] = int(Frequency[i+1])

alphab = []
calc = 1
while int(calc) <= s:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title(movie)
pyplot.xlabel('Person Number')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, frequencies, width, color='r')

plt.show()

t = np.loadtxt('/home/wizkraft/Desktop/Faces/'+movie+'/images.txt', usecols=range(1))

noi = 0 #number of intervals
while(t[noi]>0):
    noi = noi+1

t[noi] = n+1

a1 = [0]*noi
a2 = [0]*noi
a3 = [0]*noi
a4 = [0]*noi


for i in range(0,noi) :
    for j in range(int(t[i]),int(t[i+1])):
        if IndexOfThePersonAssigned[j]==index1:
            a1[i]=a1[i]+1
        if IndexOfThePersonAssigned[j]==index2:
            a2[i]=a2[i]+1
        if IndexOfThePersonAssigned[j]==index3:
            a3[i]=a3[i]+1
        if IndexOfThePersonAssigned[j]==index4:
            a4[i]=a4[i]+1

print a1
print a2
print a3
print a4

print("--- %s seconds ---" % (time.time() - start))

alphab = []
calc = 1
while int(calc) <= noi:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title('Actor 1')
pyplot.xlabel('Time Interval')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, a1, width, color='r')

plt.show()

alphab = []
calc = 1
while int(calc) <= noi:
    alphab.append(str(calc))
    calc = int(calc) + 1


pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

pyplot.title('Actor 2')
pyplot.xlabel('Time Interval')
pyplot.ylabel('Appearence Frequency')
plt.bar(pos, a2, width, color='r')

plt.show()

# edit in line 21 to upload desired matrix
# edit in line 153 to upload the images matrix specifying time interval