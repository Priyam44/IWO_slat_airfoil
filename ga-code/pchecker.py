import numpy as np
import bezierbuilder as bb
import matplotlib.pyplot as plt
def pointcheck(coordinates,point,ref_point):
    slope=(coordinates[0][1]-coordinates[1][1])/(coordinates[0][0]-coordinates[1][0])
    pvalue=(coordinates[0][1]-point[1])-(slope*(coordinates[0][0]-point[0]))
    rpvalue=(coordinates[0][1]-ref_point[1])-(slope*(coordinates[0][0]-ref_point[0]))
    if(rpvalue!=0):
        if(pvalue/rpvalue > 0 ):
            print('true')
            return True
        else:
            print("false")
            return False
    else:
        print("error:rpvalue=0")
        return
"""def divide_cpoints(chromosome):
    cpoints1=np.empty((0,2))
    cpoints2=np.empty((0,2))

    for i in range(9):
        if(i<=4):
            cpoints1=np.append(cpoints1,[chromosome[i]],axis=0)
        if(i>=4 and i<=8):
            cpoints2=np.append(cpoints2,[chromosome[i]],axis=0)
    cpoints2=np.append(cpoints2,[chromosome[0]],axis=0)
    cpoints=np.array([cpoints1,cpoints2])
    return cpoints
initial_cpoints=np.array([[-0.11236,0.019943],
 [-0.09054228 , 0.06216698],
 [-0.05712186 , 0.09847833],
 [-0.01482015 , 0.12391961],
 [ 0.030924  ,  0.1368    ],
 [-0.13130294,  0.07826767],
 [ 0.03289166,  0.15488873],
 [-0.22181531,  0.04901755],
 [-0.11236   ,  0.019943  ]]
)
cpoints=divide_cpoints(initial_cpoints)
curve1=bb.BezierBuilder(cpoints[0])
curve2=bb.BezierBuilder(cpoints[1])
curve=np.array(curve1.get_coordinates())
curve=np.append(curve,curve2.get_coordinates(),axis=0)
curve=curve.transpose()
curve1.plot()
curve2.plot()
plt.show()
check_coordinates=np.array([initial_cpoints[5],initial_cpoints[4]])
pointcheck(check_coordinates,initial_cpoints[2],initial_cpoints[8])"""
