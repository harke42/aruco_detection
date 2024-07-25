import math
import numpy as np
import ast
import sys

#Marker2    Marker3  Marker1
cam_center, cam_top, cam_right, dist = sys.argv[1:]

cam_center = ast.literal_eval(cam_center)   
cam_center = [float(i) for i in cam_center]

cam_top = ast.literal_eval(cam_top)
cam_top = [float(i) for i in cam_top]
cam_right = ast.literal_eval(cam_right)
cam_right = [float(i) for i in cam_right]
dist = ast.literal_eval(dist)
dist = float(dist)

rv1 = [cam_right[0]-cam_center[0], cam_right[1]-cam_center[1], cam_right[2]-cam_center[2]]
#norm2 = np.linalg.norm(rv2)
#rv2 = [rv2[0]/norm2, rv2[1]/norm2, rv2[2]/norm2]
rv2 = [cam_top[0]-cam_center[0], cam_top[1]-cam_center[1], cam_top[2]-cam_center[2]]
#norm1 = np.linalg.norm(rv1)
#rv1 = [rv1[0]/norm1, rv1[1]/norm1, rv1[2]/norm1]

n = [rv1[1]*rv2[2] - rv1[2]*rv2[1], rv1[2]*rv2[0] - rv1[0]*rv2[2], rv1[0]*rv2[1] - rv1[1]*rv2[0]]
norm = np.linalg.norm(n)
n = [n[0]/norm, n[1]/norm, n[2]/norm]

cam_depth = 0.044 - 0.5*0.0125

cam_pos = [cam_center[0] + cam_depth*n[0], cam_center[1] + cam_depth*n[1], cam_center[2] + cam_depth*n[2]]
        
dist = dist + 0.0215 - 0.5*0.0125
print(n)
print([cam_pos[0] + dist*n[0], cam_pos[1] + dist*n[1], cam_pos[2] + dist*n[2]])   