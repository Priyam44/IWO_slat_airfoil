"""import os
filepath=os.path.join('/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue','priyam.dat')
if not os.path.exists('/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue'):
    os.makedirs('/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue')
file1=open("/home/fmg/OpenFOAM/fmg-6/run/Airfoil/simvalue/priyam.dat","w")
file1.write("hello")          
file1.close()"""
import STLgen,aircordinates,math
import matplotlib.pyplot as plt
import numpy as np
import bezierbuilder as bb
"""def transform_slat(target, gap, overhang, deflection, lead, trail):
    # making slat horizontal
    slope = math.degrees(
        math.atan(abs((lead[1] - trail[1]) / (lead[0] - trail[0]))))
    print(target)
    target = STLgen.rotate(target, slope, [0, 0, 1])

    # getting tail to origin
    ysub = target[np.argmax(target, axis=0)[0]][1]
    target = target.transpose()
    target[0] = target[0] - target.max(1)[0]
    target[1] = target[1] - ysub
    # providing deflection
    target = STLgen.rotate(target.transpose(), deflection, [0, 0, 1])
    # providing gap and overhang
    target = target.transpose()
    height = (abs(gap**2 - overhang**2))**0.5
    target[0] = target[0] - overhang
    target[1] = target[1] + height
    # providing angle of attack
    target = STLgen.rotate(target.transpose(), 13)
    return target

cord=transform_slat(aircordinates.cord,2,1.5,-35,[-11.236,	1.9943,0	],[3.0924	,      13.68	,0])
cord=STLgen.rotate(cord,13,[0,0,1])
STLgen.STL_Gen(cord.transpose()[0],cord.transpose()[1],0)"""
#STLgen.plot("air.stl")
#plt.plot(cord.transpose()[0],cord.transpose()[1])
#plt.plot(aircordinates.aircord_rot.transpose()[0],aircordinates.aircord_rot.transpose()[1])
#plt.axes().set_aspect('equal')
#plt.show()

k=np.array([[[ 7.60603239 ,-6.20833351],
[13.69517696 , 5.85851774],
[27.60684517 ,16.01853713],
[34.93704626 ,21.14138628]],
[[34.93704626 ,21.14138628],
[27.55482433 ,17.05456459],
[14.4443948  ,12.18180783],
[15.28433687 ,16.98635302],
[ 7.60603239 ,-6.20833351]]])
k = np.array([[  6.9961086  , -5.54978974],
[ 13.75059254 ,  5.94414712],
[ 25.11931948 , 14.62378626],
[ 37.65078892 , 18.92709087],
[ 27.62107402 , 17.09798259],
[ 14.53714153 , 11.92558065],
[ 14.03779692 , 14.61821902],
[-13.94023124 , -1.9938111 ]])
n1=bb.BezierBuilder(k[0])
n2=bb.BezierBuilder(k[1])
n11=n1.get_coordinates()
n22=n2.get_coordinates()
n11=np.append(n11,n22,0)
n11=n11/1000
plt.plot(n1.transpose()[0],n1.transpose()[1])
plt.plot(n2.transpose()[0],n2.transpose()[1])
#STLgen.STL_Gen(n11.transpose()[0],n11.transpose()[1],'new.stl')
#STLgen.combine('new_slat.STL','/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/wingMotion_snappyHexMesh/constant/triSurface/ClarkY_airfoil_surf.STL','/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/wingMotion_snappyHexMesh/constant/triSurface/air.stl')
#STLgen.plot('/home/fmg/OpenFOAM/fmg-6/run/Airfoil/cfd_openfoam/wingMotion_snappyHexMesh/constant/triSurface/air.stl')