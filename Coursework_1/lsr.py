#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:50:31 2019

@author: joshhamwee
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from utilities import load_points_from_file

#Function given by utilities file with error catchin
def readFile(filename):
    if(filename != None):
        return load_points_from_file(filename)
    else:
        print("Please enter a file name")


def showGraphs(xs,ys,xfinal,yfinal,n):
    #Find out how many line segments and that there are equal amounts
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    
    #Change the colour for each segment
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    
    #Plot line graphs & scatter graphs for each segment
    for n in range(n):
        plt.plot(xfinal[0][n],yfinal[n][0],color = 'red')
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()
    
def least_squares_linear(xi,yi):
    #Initial variables
    n = 1
    xs = []
    gen = (xi**i for i in range(n+1))
    for val in gen:
        xs.append(val)
    
    #Creating the two matrices
    Y = np.matrix(yi)
    X = np.matrix(xs)
    
    #Matrix multiplication to work out a & b values
    X_transpose = X.transpose()
    A = (np.matmul(X,X_transpose))
    A = np.linalg.inv(A)
    A = np.matmul(X_transpose,A)
    A = np.matmul(Y,A)
    
    #Return final ybar value
    ybar = np.matmul(A,X)
    return np.array(ybar)


def least_squares_poly(xi,yi):
    #Initial variables
    n = 3
    xs = []
    gen = (xi**i for i in range(n+1))
    for val in gen:
        xs.append(val)
    
    #Creating the two matrices
    Y = np.matrix(yi)
    X = np.matrix(xs)
    
    #Matrix multiplication to work out a & b values
    X_transpose = X.transpose()
    A = (np.matmul(X,X_transpose))
    A = np.linalg.inv(A)
    A = np.matmul(X_transpose,A)
    A = np.matmul(Y,A)
    
    #Return final ybar value
    ybar = np.matmul(A,X)
    return np.array(ybar)


def least_squares_unknown_sin(xi,yi):
    #Coloumn for ones
    ones = np.ones(len(xi))
    
    #Creating the two matrices
    Y = np.matrix([yi])
    X = np.matrix([ones,np.sin(xi)])
    
    #Matrix multiplication to work out a & b values
    XT = X.transpose()
    inversed = np.linalg.inv(np.matmul(X,XT))
    A = np.matmul(Y,np.matmul(XT,inversed))
    
    #Return final ybar value
    ybar = np.matmul(A,X)
    return np.array(ybar)


def sum_squared_error(ybar,yoriginal):
    #Calculate the sum squared error on the ybar compared to the original y
    total_error = np.sum((ybar-yoriginal)**2)
    return total_error


if __name__== "__main__":
  xs, ys = readFile(sys.argv[1])
  
  #Find out how many line segments, then split into subarrary
  numberLineSegs = int(len(xs)/20)
  xsplit = np.split(xs,numberLineSegs)
  ysplit = np.split(ys,numberLineSegs)
  
  #Initial variables
  total_error = 0
  yfinal = []
  
  #Iterate through all line segments
  for i in range(len(xsplit)):
      #Calculate ybars for all function types
      ylinear = least_squares_linear(xsplit[i],ysplit[i])
      ypoly = least_squares_poly(xsplit[i],ysplit[i])
      yunknown = least_squares_unknown_sin(xsplit[i],ysplit[i])
      
      #Calculate error for all ybars
      ylinearerror = sum_squared_error(ylinear,ysplit[i])
      ypolyerror = sum_squared_error(ypoly,ysplit[i])
      yunknownerror = sum_squared_error(yunknown,ysplit[i])
      
      #Return the smallest error and append the yvalues to the final y array
      if(ylinearerror<ypolyerror and ylinearerror<yunknownerror):
          yfinal.append(ylinear)
      elif(ypolyerror<ylinearerror and ypolyerror<yunknownerror):
          yfinal.append(ypoly)
      else:
          yfinal.append(yunknown)
      total_error = total_error + min(ylinearerror,ypolyerror,yunknownerror)


  #Print the final error to be read by the automarker
  print(total_error)
  
  #Show the graph if --plot is specified in the sys.argv arguments
  xfinal = []
  xfinal.append(xsplit)
  if(len(sys.argv)==3):
      if(sys.argv[2] == '--plot'):
          showGraphs(xs,ys,xfinal,yfinal,numberLineSegs)
  
          
  
  
  
  
      
      
  