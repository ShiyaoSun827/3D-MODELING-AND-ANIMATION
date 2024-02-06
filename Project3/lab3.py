import numpy as np
#########################
#       Exercise 1      #
#########################

def generateTranslationMatrix(x, y, z):
    '''
    return the homogeneous transformation matrix for the given translation (x, y, z)
      parameter: 
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    #A homoheneous Trans mastrix:[1 0 0 tx
    #                             0 1 0 ty
    #                             0 0 1 tz
    #                             0 0 0  1]
    #create identity matrix
    trans = np.eye(4)
    #change the entries (0s) on the last cols to x,y,z
    trans[0:3,3] = x,y,z
    return trans

def generateScalingMatrix(sx, sy, sz):
    '''
    return the homogeneous transformation matrix for the given scaling parameters (sx, sy, sz)
      parameter:
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    #A homoheneous scaling mastrix:[sx 0 0 0
    #                               0 sy 0 0
    #                               0 0 sz 0
    #                               0 0 0  1]
    #create identity matrix
    scaling = np.eye(4)
    #change the entries (1s,except the last one) on the diagonals to x,y,z
    scaling[0,0] = sx
    scaling[1,1] = sy
    scaling[2,2] = sz
    return scaling

def generateRotationMatrix(rad, axis):
    '''
    return the homogeneous transformation matrix for the given rotation parameters (rad, axis)
      parameter:
        rad: radians for rotation
        axis: axis for rotation, can only be one of ('x', 'y', 'z', 'X', 'Y', 'Z')
      return: 
        ndarray of size (4, 4)
    '''
    #create identity matrix
    rotate = np.eye(4)
    #check axis
    if axis == 'x' or axis == 'X':
        rotate[1,1] = np.cos(rad)
        rotate[1,2] = -np.sin(rad)
        rotate[2,1] = np.sin(rad)
        rotate[2,2] = np.cos(rad)
    elif axis == 'y' or axis == 'Y':
        rotate[0,0] = np.cos(rad)
        rotate[2,0] = -np.sin(rad)
        rotate[0,2] = np.sin(rad)
        rotate[2,2] = np.cos(rad)
    elif axis == 'z' or axis == 'Z':
        rotate[0,0] = np.cos(rad)
        rotate[0,1] = -np.sin(rad)
        rotate[1,0] = np.sin(rad)
        rotate[1,1] = np.cos(rad)
    return rotate
        

        

# Case 1
def part1Case1():
    # translation matrix
    t = [2,3,-2]
    # scaling matrix
    s = [0.5,2,2]
    # rotation matrix
    r = (45/180)*np.pi
    # data in homogeneous coordinate
    data = np.array([2, 3, 4, 1]).T
    #get trans matrix
    trans = generateTranslationMatrix(t[0],t[1],t[2])
    #get scal matrix
    scal = generateScalingMatrix(s[0],s[1],s[2])
    #get rotate matrix
    rotate = generateRotationMatrix(r,'z')
    #print it out step by step
    trans_data = np.matmul(trans,data)
    print("Test1:")
    print("(1)trans * data:\n",trans_data)
    scal_trans_data = np.matmul(scal,trans_data)
    print("(2)scal * trans * data:\n",scal_trans_data)
    rotate_scal_trans_data = np.matmul(rotate,scal_trans_data)
    print("(3)rotate * scal * trans * data(FINAL ANS):\n",rotate_scal_trans_data)
    
# Case 2
def part1Case2():
    # translation matrix
    t = [4,-2,3]
    # scaling matrix
    s = [3,1,3]
    # rotation matrix
    r = (-30/180)*np.pi
    # data in homogeneous coordinate
    data = np.array([6, 5, 2, 1]).T
    #get trans matrix
    trans = generateTranslationMatrix(t[0],t[1],t[2])
    #get scal matrix
    scal = generateScalingMatrix(s[0],s[1],s[2])
    #get rotate matrix
    rotate = generateRotationMatrix(r,'y')
    #print it out step by step
    scal_data = np.matmul(scal,data)
    print("Test2:")
    print("(1)scal * data:\n",scal_data)
    trans_scal_data = np.matmul(trans,scal_data)
    print("(2)trans * scal * data:\n",trans_scal_data)
    rotate_trans_scal_data = np.matmul(rotate,trans_scal_data)
    print("(3)rotate * trans * scal * data(FINAL ANS):\n",rotate_trans_scal_data)

# Case 3
def part1Case3():
    # translation matrix
    t = [5,2,-3]
    # scaling matrix
    s = [2,2,-2]
    # rotation matrix
    r = (15/180)*np.pi
    # data in homogeneous coordinate
    data = np.array([3, 2, 5, 1]).T
    #get trans matrix
    trans = generateTranslationMatrix(t[0],t[1],t[2])
    #get scal matrix
    scal = generateScalingMatrix(s[0],s[1],s[2])
    #get rotate matrix
    rotate = generateRotationMatrix(r,'x')
    #print it out step by step
    rotate_data = np.matmul(rotate,data)
    print("Test3:")
    print("(1)rotate * data:\n",rotate_data)
    scal_rotate_data = np.matmul(scal,rotate_data)
    print("(2)scal * rotate * data:\n",scal_rotate_data)
    trans_scal_rotate_data = np.matmul(trans,scal_rotate_data)
    print("(3)rotate * scal * trans matrix * data(FINAL ANS):\n",trans_scal_rotate_data)

#########################
#       Exercise 2      #
#########################

# Part 1
def generateRandomSphere(r, n):
    '''
    generate a point cloud of n points in spherical coordinates (radial distance, polar angle, azimuthal angle)
      parameter:
        r: radius of the sphere
        n: total number of points
    return:
      spherical coordinates, ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    '''
    #generate spherical points from uniform distribution randomly,3 rows,n cols
    SphereMatrix = np.zeros((3,n))#3 rows,n cols

    #uniform random sample
    radial_dist = np.random.uniform(0,r,size = (1,n))
    #assign radial_dist to SphereMatrix
    SphereMatrix[0,:] = radial_dist

    #polar angle is from 0 to pi
    polar_angle = np.random.uniform(0,np.pi,size = (1,n))
    #assign polar angle to SphereMatrix
    SphereMatrix[1,:] = polar_angle

    #azimuthal angles is from -pi to pi
    azimuthal_angles = np.random.uniform(-np.pi,np.pi,size = (1,n))
    #assign azimuthal_angles to SphereMatrix
    SphereMatrix[2,:] = azimuthal_angles

    return SphereMatrix
    





def sphericalToCatesian(coors):
    '''
    convert n points in spherical coordinates to cartesian coordinates, then add a row of 1s to them to convert
    them to homogeneous coordinates
      parameter:
        coors: ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    return:
      catesian coordinates, ndarray of size (4, n), where the 4 rows are ordered as (x, y, z, 1)
    '''
    #x = RadialDist * sin(polar) * cos(azimuthal)
    #y = RadialDist * sin(polar) * sin(azimuthal)
    #z = RadialDist * cos(polar)
    #intialze a matrix with all entries are 0s except the last row(all 1s)
    Catesian = np.zeros((4,coors.shape[1]))
    Catesian[-1,:] = 1
    #change spherical to catesian on each col of coors 
    for i in range(coors.shape[1]):
        #x = RadialDist * sin(polar) * cos(azimuthal)
        Catesian[0][i] = coors[0][i] * np.sin(coors[1][i]) * np.cos(coors[2][i])
        #y = RadialDist * sin(polar) * sin(azimuthal)
        Catesian[1][i] = coors[0][i] * np.sin(coors[1][i]) * np.sin(coors[2][i])
        #z = RadialDist * cos(polar)
        Catesian[2][i] = coors[0][i] * np.cos(coors[1][i]) 
    return Catesian
            

    

# Part 2
def applyRandomTransformation(sphere1):
    '''
    generate two random transformations, one of each (scaling, rotation),
    apply them to the input sphere in random order, then apply a random translation,
    then return the transformed coordinates of the sphere, the composite transformation matrix,
    and the three random transformation matrices you generated
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
      return:
        a tuple (p, m, t, s, r)
        p: transformed homogeneous coordinates, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        m: composite transformation matrix, ndarray of size (4, 4)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    '''
    #generate two random transformations,scal,rotation

    #scal entries
    S_matrix = np.zeros((1,3))
    #use any to check whether exists 0 entries inside the S_matrix matrix
    while np.any(S_matrix == 0):    
      S_matrix = np.random.uniform(-50,50,size = (1,3))
    #assign S_matrix to diagnals of Scal
    s = generateScalingMatrix(S_matrix[0,0],S_matrix[0,1],S_matrix[0,2])

    #Rotate entries,angle is from -pi to pi
    angle = np.random.uniform(-np.pi,np.pi)
    #choose the axis that it rotates randomly.0 = x, 1 = y, 2 = z
    randomnum = np.random.randint(3)
    #create the rotate matrix
    if randomnum == 0:
        r = generateRotationMatrix(angle,'x')
    elif randomnum == 1:
        r = generateRotationMatrix(angle,'y')
    elif randomnum == 2:
        r = generateRotationMatrix(angle,'z')
    
    #Trans entries
    T_matrix = np.random.uniform(-50,50,size = (1,3))
    #assign T_matrix to diagnals of Trans
    t = generateTranslationMatrix(T_matrix[0,0],T_matrix[0,1],T_matrix[0,2])

    #randomly select the order of the matrix (scal, rotation)

    order_num = np.random.randint(2)
    #rotation fisrt
    if order_num == 0:
        #rotation * sphere1
        R_S = np.matmul(r,sphere1)
        #Scaling * (rotation * sphere1)
        S_R_S = np.matmul(s,R_S)
        #Trans * (Scaling * (rotation * sphere1)),transformed homogeneous coordinates
        p = np.matmul(t,S_R_S)
        #composite transformation matrix, ndarray of size (4, 4)
        m = np.matmul(np.matmul(t,s),r)
    #Scaling first
    elif order_num == 1:
        #Scaling * sphere1
        R_S = np.matmul(s,sphere1)
        #rotation * (Scaling * sphere1)
        S_R_S = np.matmul(r,R_S)
        #Trans * (rotation * (Scaling * sphere1)),transformed homogeneous coordinates
        p = np.matmul(t,S_R_S)
        #composite transformation matrix, ndarray of size (4, 4)
        m = np.matmul(np.matmul(t,r),s)
    
    return (p, m, t, s, r)
         


        
        




def calculateTransformation(sphere1, sphere2):
    '''
    calculate the composite transformation matrix from sphere1 to sphere2
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        sphere2: homogeneous coordinates of sphere2, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
    return:
      composite transformation matrix, ndarray of size (4, 4)
    '''
    #Since transformed_M = HM,we can assume transformed_M = sphere2,M = sphere1
    #and M = U*sigma*V_transposed,where U is 4x4 Orthogonal matrix,V is nxn Orthogonal matrix
    #To calculate H,transformed_M * inverse_M = H * M * inverse_M
    #And H = transformed_M * inverse_M
    #Since M = U*sigma*V_transposed, and U,V are Orthogonal matrixes,
    #then U_transposed = inverse_U,V_transposed = inverse_V
    #inverse_M = inverse(U*sigma*V_transposed) = inverse_V_transposed * inverse_sigma * inverse_U
    #inverse_M=V * inverse_sigma * U_transposed


    #we can assume transformed_M = sphere2,M = sphere1
    transformed_M = sphere2
    M = sphere1
    #and M = U*sigma*V_transposed
    U, sigma, V_transposed  = np.linalg.svd(M)

    #we need a diagonal matrix for sigma, but the sigma that returned by np.linalg.svd
    # is a 1 dimensional array,we need to convert it ,and U*sigma*V_transposed,
    #the # of row is U.shape[1],col is V_transposed.shape[0]
    temp = np.zeros((U.shape[1],V_transposed.shape[0]))
    for i in range(len(sigma)):
        temp[i,i] = sigma[i]
    #compute Pseudo-Inverse of sigma
    inverse_sigma = np.linalg.pinv(temp)

    #inverse_M=V * inverse_sigma * U_transposed
    V = np.transpose(V_transposed)
    U_transposed = np.transpose(U)

    inverse_M = np.matmul(np.matmul(V,inverse_sigma),U_transposed)
    #H = transformed_M * inverse_M
    H = np.matmul(transformed_M,inverse_M)

    return H
  

    

def decomposeTransformation(m):
    '''
    decomposite the transformation and return the translation, scaling, and rotation matrices
      parameter:
        m: homogeneous transformation matrix, ndarray of size (4, 4)

    return:
      tuple of three matrices, (t, s, r)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    '''
    
    #we need to create t,s,r, which are trans,scal,rotate
    #initialize identity matrixes
    t = np.eye(4)
    s = np.eye(4)
    r = np.eye(4)

    #get the trans matrix
    t[0:3,3] =  m[0:3,3]

    #get the scaling num
    scaling_num = np.linalg.norm(m[:3, :3], axis=0)
    s[:3,:3] = np.diag(scaling_num)

    #we need to divide the entries of m by the scaling num to get the rotate matrix
    #normalized matrix 
    rotate = m[:3,:3]/scaling_num
    #use svd to get the u and transposed_v
    u, _, vh = np.linalg.svd(rotate)
    #compute the normolized 
    rotate = np.matmul(u, vh)
    
    r[:3,:3] = rotate
    
    
    return t, s, r








#########################
#      Main function    #
#########################
def main():
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    part1Case1()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    part1Case2()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    part1Case3()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Exercise2:")
    # Parameters for the sphere
    radius = 2
    num_points = 5

    #random generate a sphere

    sphere = generateRandomSphere(radius, num_points)
    #change it to cartesian
    cartesian = sphericalToCatesian(sphere)
    print("Original Sphere:\n", cartesian)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    p, m, t, s, r = applyRandomTransformation(cartesian)
    print("Transformed Sphere:\n", p)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Original Transformation Matrix:\n", m)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Original Translation Matrix:\n", t)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Original Scaling Matrix:\n", s)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Original Rotation Matrix:\n", r)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Calculate the transformation matrix from the original to the transformed sphere
    calculated_matrix = calculateTransformation(cartesian, p)

    # Decompose the transformation matrix
    decomposed_t, decomposed_s, decomposed_r = decomposeTransformation(calculated_matrix)

    
    print("Calculated Transformation Matrix:\n", calculated_matrix)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    print("Decomposed Translation Matrix:\n", decomposed_t)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    print("Decomposed Scaling Matrix:\n", decomposed_s)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
 
    print("Decomposed Rotation Matrix:\n", decomposed_r)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Original Rotation Matrix:\n", r)






if __name__ == "__main__":
    main()