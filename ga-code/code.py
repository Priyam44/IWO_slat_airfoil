import random,os,math,copy,STLgen,aircordinates,subprocess
import numpy as np
import bezierbuilder as bb
import slopecheck as slopec
import matplotlib.pyplot as plt
import time
import pandas as pd
CHROMOSOME_SIZE=10
genes=np.array([[ 6.9961086,  -5.54978974],
[13.75059254,  5.94414712],
[25.11931948, 14.62378626],
[37.65078892, 18.92709087],
[27.62107402, 20.64926013],
[14.53714153, 14.75374382],
[14.03779692, 13.74067422],
[-13.94023124,  -1.9938111 ],
[17.11390514, -0.39592132],
[5.07960789, 1.798983  ]]
)
def transform_slat(target, gap, overhang, deflection, lead, trail,cpoints):
	"""# making slat horizontal
	slope = math.degrees(
		math.atan(abs((lead[1] - trail[1]) / (lead[0] - trail[0]))))
	print(target)
	target = STLgen.rotate(target, slope, [0, 0, 1])"""

	"""# getting tail to origin
	ysub = target[np.argmax(target, axis=0)[0]][1]
	target = target.transpose()
	target[0] = target[0] - target.max(1)[0]
	target[1] = target[1] - ysub"""
	#getting head to origindf = pd.read_csv('aircord.csv')
		
	target= target.transpose()
	print (cpoints[0][0])
	target[0]-=  6.9961086 - cpoints[0][0]
	target[1]-=  -5.54978974 + cpoints[0][1]
	# providing deflection
	target = STLgen.rotate(target.transpose(), deflection, [0, 0, 1])
	# providing gap and overhang
	target = target.transpose()
	height=gap
	#height = (abs(gap**2 - overhang**2))**0.5
	target[0] = target[0] - overhang
	target[1] = target[1] + height
	"""# providing angle of attack
	target = STLgen.rotate(target.transpose(), 13)"""
	return target.transpose()
def profile_gen(_genes):
	  cpoints = divide_cpoints(_genes)
	  curve1 = bb.BezierBuilder(cpoints[0])
	  curve2 = bb.BezierBuilder(cpoints[1])
	  curve = np.array(curve1.get_coordinates())
	  curve = np.append(curve, curve2.get_coordinates(), 0)
	  curve = np.append(curve.transpose(), np.zeros((1,curve.shape[0])), 0).transpose() #adding z coordinates =0

	  curve = transform_slat(curve,_genes[-2][1], _genes[-2][0],_genes[-1][0], _genes[0], _genes[3],_genes)  # transforming slat
	  # creating stl file of slat
	  curve = curve/1000

	  return curve

def divide_cpoints(_genes):
	cpoints1 = np.empty((0, 2))
	cpoints2 = np.empty((0, 2))
	for i in range(CHROMOSOME_SIZE):
		if(i <= 3):
			cpoints1 = np.append(cpoints1, [_genes[i]], axis=0)
		if(i >= 3 and i <= 7):
			cpoints2 = np.append(cpoints2, [_genes[i]], axis=0)
	cpoints2 = np.append(cpoints2, [_genes[0]], axis=0)
	cpoints = np.array([cpoints1, cpoints2])
	return cpoints
curve=profile_gen(genes)
plt.plot(curve.T[0],curve.T[1])
print(curve.T[0])
df = pd.read_csv('aircord.csv')
X = df['X']
Y = df['Y']
plt.plot(X/1000,Y/1000,'.')
plt.axes().set_aspect('equal')
plt.xlim(-0.04,0.05)
plt.ylim(-0.05,0.07)
plt.show()
h=open("coordinate.dat","w")
for i in range(len(curve)):
	h.write(str(curve[i][0])+","+str(curve[i][1])+"\n")
h.close()
