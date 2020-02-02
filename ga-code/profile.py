import aircordinates as ac
import numpy as np
cpoints=np.array([[ 6.9961086,  -5.54978974],
[13.75059254,  5.94414712],
[25.11931948, 14.62378626],
[37.65078892, 18.92709087],
[27.62107402, 17.09798259],
[14.53714153, 14.92558065],
[14.03779692 ,14.61821902],
[-13.94023124,  -1.9938111 ],
[2.8679262 , 0.58469625],
[0.        , 1.53286671]])

import bezierbuilder as bb

import matplotlib.pyplot as plt
import pandas as pd
import STLgen
CHROMOSOME_SIZE=10

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
    #getting head to origin
    target= target.transpose()
    print (cpoints[0][0])
    target[0]-=  6.9961086 - cpoints[0][0]
    target[1]-=  -5.54978974 + cpoints[0][1]
    # providing deflection
    # target = STLgen.rotate(target.transpose(), deflection, [0, 0, 1])
    # # providing gap and overhang
    # target = target.transpose()
    height=gap
    #height = (abs(gap**2 - overhang**2))**0.5
    target[0] = target[0] - overhang
    target[1] = target[1] + height
    """# providing angle of attack
    target = STLgen.rotate(target.transpose(), 13)"""
    return target.transpose()
def profile_gen(cp):
    cpoints = divide_cpoints(cp)
    curve1 = bb.BezierBuilder(cpoints[0])
    curve2 = bb.BezierBuilder(cpoints[1])
    curve = np.array(curve1.get_coordinates())
    curve = np.append(curve, curve2.get_coordinates(), 0)
    curve = np.append(curve.transpose(), np.zeros((1,curve.shape[0])), 0).transpose() #adding z coordinates =0

    curve = transform_slat(curve,cp[-2][1], cp[-2][0],
                           cp[-1][0], cp[0], cp[3],cp)  # transforming slat
    # creating stl file of slat
    curve = curve/1000

    return curve
def divide_cpoints(curve):
    cpoints1 = np.empty((0, 2))
    cpoints2 = np.empty((0, 2))
    for i in range(10):
      if(i <= 3):
          cpoints1 = np.append(cpoints1, [curve[i]], axis=0)
      if(i >= 3 and i <= 7):
          cpoints2 = np.append(cpoints2, [curve[i]], axis=0)
    cpoints2 = np.append(cpoints2, [curve[0]], axis=0)
    cpoints = np.array([cpoints1, cpoints2])
    return cpoints
curve=profile_gen(cpoints)
plt.plot(curve.transpose()[0],curve.transpose()[1])
df = pd.read_csv('aircord.csv')
x = df['X']
y = df['Y']
plt.plot(x/1000,y/1000,'.')
plt.axes().set_aspect('equal')
plt.xlim(-0.04,0.05)
plt.ylim(-0.05,0.07)
plt.show()
STLgen.STL_Gen(curve.T[0],curve.T[1],"curve101.stl")
np.save("curve101.npy",curve)
# np.save("aircord.npy",ac.aircord)
