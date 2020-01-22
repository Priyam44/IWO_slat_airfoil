import numpy as np

def slope_cal(line):#calculates line slope
    slope=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
    return slope
def slope_checker(line1,line2):
    slope1=slope_cal(line1)
    slope2=slope_cal(line2)
    print(slope1,'\n',slope2,'\n')
    print("-----------------------")
    if(slope1>=slope2):
        return True
    else:
        return False
