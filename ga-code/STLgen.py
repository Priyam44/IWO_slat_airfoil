from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import stl,math
from mpl_toolkits import mplot3d
from stl import mesh
def STL_Gen(x,y,filename):
    res = len(x)
    #x=[x[i]/1000 for i in range(len(x))]
    #y=[y[i]/1000 for i in range(len(x))]
    z=0.25
    i = 0; v=[]
    while (i <res): # +ve z axis points
        v.append([x[i],y[i],z])
        i += 1
    i=0
    while (i <res): # -ve z axis points
        v.append([x[i],y[i],-z])
        i += 1
    vertices=np.array(v)

    i=0; f=[]
    while (i<res-1): # generating faces
        f.append([i,i+1,res+i+1])
        f.append([i,res+i+1,res+i])
        i +=1
    faces= np.array(f)
    suf=mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            suf.vectors[i][j]= vertices[f[j],:]
    suf.save(filename, mode=stl.Mode.ASCII)
def plot(filename):
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(filename)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()
def combine(filename1,filename2,final_file):
    main_body = mesh.Mesh.from_file(filename1)
    addendum = mesh.Mesh.from_file(filename2)

    combined = mesh.Mesh(np.concatenate([main_body.data, addendum.data]))
                                        #[copy.data for copy in copies] +
                                        #[copy.data for copy in copies2]))

    combined.save(final_file, mode=stl.Mode.ASCII)  # save as ASCII
    #plot(final_file)
def rotate(target,angle,W=[0,0,1]):
    const=math.pow((W[0]**2)+(W[1]**2)+(W[2]**2),0.5)
    W=np.array(W)
    W=W/const
    angle=round((angle*(math.pi)/180),3)
    skewW=np.array([[0,W[2],-W[1]],[-W[2],0,W[0]],[W[1],-W[0],0]])
    I=[[1,0,0],[0,1,0],[0,0,1]]
    S=round(math.sin(angle),3)*skewW
    C=round((1-math.cos(angle)),3)*np.linalg.matrix_power(skewW,2)
    R=I+S+C #matrix exponential
    #print("MATRIX EXPONENTIAL",R)
    line=np.zeros((target.shape[0],3))
    for i in range(target.shape[0]):
        line[i]=np.matmul(R,target[i].transpose())
    return line
