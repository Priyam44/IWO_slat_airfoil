import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

#Class for Building Bezier Profiles from the Control Points
class BezierBuilder:

    def __init__(self,control_points):
        self.control_points=control_points
        self.curve=self.Bezier()

    def Bernstein(self,n,k,t):
        """Bernstein polynomial generator"""
        coeff = binom(n,k)
        return coeff * (t ** k) * ((1-t)**(n-k))

    def Bezier(self,num=50):
        """building bezier curve from control points"""
        N=len(self.control_points) #number of control points
        t = np.linspace(0,1,num=num) #num is no of curve points
        curve = np.zeros((num,2))
        for i in range(num):
            for j in range(N):
                curve[i][0]+=(self.Bernstein(N-1,j,t[i])*self.control_points[j][0])
                curve[i][1]+=(self.Bernstein(N-1,j,t[i])*self.control_points[j][1])
        return curve

    def get_coordinates(self):
        return self.curve
        
    def plot(self):
        """Plotting the Bezier Curves"""
        #print (np.around(curve,2))
        curve=self.curve.transpose()
        cpoints=self.control_points.transpose()
        plt.plot(curve[0],curve[1],'b-')
        plt.plot(cpoints[0],cpoints[1],'r-')
        #plt.axis([0,10,-6,6])
