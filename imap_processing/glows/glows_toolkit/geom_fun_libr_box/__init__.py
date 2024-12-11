"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl, version: Apr 12, 2021
LGTM:
Geometrical functions providing functionality of Mathematica library geomFunLibrBox
Function names and functionality are intentionally possibly similar to geomFunLibrBox
Functions are listed in alphabetical order
"""
import numpy as np

def angle_coord(lon_1, lat_1, lon_2, lat_2):
    """
    Compute angle between two directions lon_1,lat_1 and lon_2,lat_2 (longitude l and latitude b)

    Parameters
    ----------
      lon_1: number or 1D numpy array
        longitude for the first direction in radians
      lat_1: number or 1D numpy array
        latitude for the first direction in radians
      lon_2: number or 1D numpy array
        longitude for the second direction in radians
      lat_2: number or 1D numpy array
        latitude for the second direction in radians

    Returns
    -------
      a: number or 1D numpy array
        angle between the first and second direction in radians

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(np.degrees(gflb.angle_coord(0,0,np.radians(45),np.radians(45))))
      60.0
      >>> n=10
      >>> l1=b1=l2=np.zeros((n),dtype=float)
      >>> b2=np.radians(np.linspace(0,90,10))
      >>> print(np.degrees(gflb.angle_coord(l1,b1,l2,b2)))
      [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90.]
    """
    v_1 = make_vec(lon_1, lat_1)
    v_2 = make_vec(lon_2, lat_2)
    angle = angle_vec(v_1, v_2)
    return angle

def angle_vec(vec1, vec2):
    """
    Compute angle between two vectors vec1 and vec2

    Parameters
    ----------
      vec1,vec2: 1D or 2D numpy arrays
        input vectors

    Returns
    -------
      a: number or 1D numpy array
        angle between vectors vec1 and vec2 in radians
        if vec1 is an 1D numpy array, return the angle between vec1 and vec2
        if vec1 is a 2D numpy array, assume that rows represent vectors and return angles between
            vec1 and vec2 computed row-wise
        this function assumes that the angle between vec1 and vec2 is 0 if magnitude(vec1)==0 or
            magnitude(vec2)==0

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(np.degrees(gflb.angle_vec(np.array([0,0,1]),np.array([1,0,0]))))
      90.0
      >>> n=10
      >>> vecs1=np.zeros((n,3),dtype=float)
      >>> vecs1[:,0]=1
      >>> phi=np.radians(np.linspace(0,360,n))
      >>> vecs2=np.column_stack([np.cos(phi),np.sin(phi),vecs1[:,2]])
      >>> print(np.degrees(gflb.angle_vec(vecs1,vecs2)))
      [   0.   40.   80.  120.  160.  160.  120.   80.   40.    0.]
    """
    if np.shape(vec1) != np.shape(vec2):
        raise Exception('angle_vec() error: np.shape(vec1)!=np.shape(vec2)')
    if vec1.ndim == 1:
        if np.shape(vec1)[0] != 3:
            raise Exception('angle_vec() error: np.shape(vec1)[0]!=3')
        r_1 = np.sqrt(np.dot(vec1, vec1))
        r_2 = np.sqrt(np.dot(vec2, vec2))
        cos = np.dot(vec1, vec2)/(r_1*r_2)
        # handle a case where cos is shifted below -1 due to numerical errors
        if cos < -1.0 and np.fabs(cos+1.0) < 1.0e-14:
            cos = -1.0
        # handle a case where cos is shifted above 1 due to numerical errors
        if cos > 1.0 and np.fabs(cos-1.0) < 1.0e-14:
            cos = 1.0
        if r_1 != 0.0 and r_2 != 0.0:
            return np.arccos(cos)
        return 0.0

    if np.shape(vec1)[1] != 3:
        raise Exception('angle_vec() error: np.shape(vec1)[1]!=3')
    r_1 = np.sqrt((vec1 * vec1).sum(axis=1))
    r_2 = np.sqrt((vec2 * vec2).sum(axis=1))
    idxs0 = np.nonzero(np.logical_and(r_1 != 0.0, r_2 != 0.0))
    cos = (vec1[idxs0[0], :] * vec2[idxs0[0], :]).sum(axis=1)/(r_1[idxs0[0]]*r_2[idxs0[0]])
    idxs = np.nonzero(np.logical_and(cos < -1.0, np.fabs(cos+1.0) < 1.0e-14))
    cos[idxs[0]] = -1.0 # just in case if cos is shifted below -1 due to numerical errors
    idxs = np.nonzero(np.logical_and(cos > 1.0, np.fabs(cos-1.0) < 1.0e-14))
    cos[idxs[0]] = 1.0 # just in case if cos is shifted above 1 due to numerical errors
    angle = np.zeros((np.shape(vec1)[0]), dtype=float)
    angle[idxs0[0]] = np.arccos(cos)
    return angle

def get_spher_coord_rad(vec):
    """
    Compute radius r, longitude l and latitude b for a vector vec

    Parameters
    ----------
      vec: 1D or 2D numpy array
        input vector(s)

    Returns
    -------
      if vec is an 1D numpy array, return 1D numpy array rlb (see below)
      if vec is a 2D numpy array, assume that its rows represent vectors and return 2D array
        rlb (see below), where every row of rlb corresponds to every row of vec

      rlb: 1D or 2D numpy array
        if 1D then rlb=[r,l,b] where: r - magnitude of vector vec, l - longitude of the direction
          pointed by vec in radians, b - latitude of the direction pointed by vec in radians
        if 2D then columns of the array contain r,l,b

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> rlb=gflb.get_spher_coord_rad(np.array([0,1,1]))
      >>> print(rlb[0],np.degrees(rlb[1]),np.degrees(rlb[2]))
      1.41421356237 90.0 45.0
      >>> n=10
      >>> phi=np.radians(np.linspace(0,360,n))
      >>> tht=np.radians(np.linspace(-90,90,n))
      >>> vecs=np.column_stack([np.cos(phi)*np.cos(tht),np.sin(phi)*np.cos(tht),np.sin(tht)])
      >>> print(gflb.get_spher_coord_rad(vecs))
      [[ 1.          0.         -1.57079633]
       [ 1.          0.6981317  -1.22173048]
       [ 1.          1.3962634  -0.87266463]
       [ 1.          2.0943951  -0.52359878]
       [ 1.          2.7925268  -0.17453293]
       [ 1.          3.4906585   0.17453293]
       [ 1.          4.1887902   0.52359878]
       [ 1.          4.88692191  0.87266463]
       [ 1.          5.58505361  1.22173048]
       [ 1.          6.28318531  1.57079633]]
    """
    if vec.ndim == 1:
        if np.shape(vec)[0] != 3:
            raise Exception("get_spher_coord_rad() error: np.shape(vec)[0]!=3")
        vec_m = np.sqrt(np.dot(vec, vec))
        if vec_m == 0.0:
            lat = 0.0
        else:
            lat = np.arcsin(vec[2]/vec_m) # compute latitude
        # compute longitude
        if vec[0] == 0.0 and vec[1] == 0.0:
            lon = 0.0
        else:
            lon = np.arctan2(vec[1], vec[0])%(2*np.pi)
        return np.array([vec_m, lon, lat]) # return one array similarly as in Mathematica
    if np.shape(vec)[1] != 3:
        raise Exception("get_spher_coord_rad() error: np.shape(vec)[1]!=3")
    vec_m = np.sqrt((vec * vec).sum(axis=1))
    idxs = np.nonzero(vec_m != 0.0)
    lat = np.zeros((len(vec_m)), float)
    lat[idxs[0]] = np.arcsin(vec[idxs[0], 2]/vec_m[idxs[0]]) # compute latitude
    lon = np.arctan2(vec[:, 1], vec[:, 0])%(2*np.pi) # compute longitude
    idxs = np.nonzero(np.logical_and(vec[:, 0] == 0.0, vec[:, 1] == 0.0))
    lon[idxs[0]] = 0.0 # set lon=0 if vec[:,0]==0.0 and vec[:,1]==0.0
    return np.column_stack([vec_m, lon, lat]) # return one array similarly as in Mathematica

def lb2spin(l_0, b_0, soff, l_q, b_q):
    """
    This is "inverse" function with respect to spin2lb(), i.e. for a given (l_q,b_q) of a query
      point on a circle it computes the angular radius theta and the spin angle s. See spin2lb()
      for more details.

    Parameters
    ----------
      l_0: number or 1D numpy array
        longitude of the center of the circle in radians
      b_0: number or 1D numpy array
        latitude of the center of the circle in radians
      soff: number or 1D numpy array
        spin offset in radians
      l_q: number or 1D numpy array
        longitude of a query point in radians
      b_q: number or 1D numpy array
        latitude of a query point in radians

    Returns
    -------
      if parameters are numbers, return 1D numpy array [tht,s]
      if parameters are 1D numpy arrays, return 2D array [tht,s] where
      columns rpresent tht,s
      tht: number or 1D numpy array
        angular distance (in radians) between l_0,b_0 and l_q,b_q
      s: number or 1D numpy array
        spin angle (in radians) measuring position of l_q,b_q on a circle centered on l_0,b_0

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(np.degrees(gflb.lb2spin(np.radians(100.0), 0.0, 0.0, np.radians(70.0),\
          np.radians(0.0))))
      [ 30.  90.]
      >>> soff=0.0
      >>> s0=np.radians(np.array(range(4))*360.0/4)
      >>> l_0=np.radians(100.0*np.ones((len(s0)),dtype=float))
      >>> b_0=np.radians(0.0*np.ones((len(s0)),dtype=float))
      >>> tht0=np.radians(40.0*np.ones((len(s0)),dtype=float))
      >>> lb=gflb.spin2lb(l_0,b_0,s0,soff,tht0)
      >>> print(lb)
      [[  1.74532925e+00   6.98131701e-01]
       [  1.04719755e+00   8.62660832e-17]
       [  1.74532925e+00  -6.98131701e-01]
       [  2.44346095e+00  -7.11714745e-17]]
      >>> print(np.degrees(gflb.lb2spin(l_0,b_0,soff,lb[:,0],lb[:,1])))
      [[  40.    0.]
       [  40.   90.]
       [  40.  180.]
       [  40.  270.]]
    """
    tht = angle_coord(l_0, b_0, l_q, b_q)
    spin = (-position_angle(l_q, b_q, l_0, b_0)-soff)%(2.*np.pi)
    if np.isscalar(l_0):
        return np.array([tht, spin])
    return np.column_stack([tht, spin])

def make_co_vec(l_q, b_q):
    """
    Compute vectors for given values of longitude l_q and co-latitude b_q

    Parameters
    ----------
      l_q: number or 1D numpy array
        longitude of a vector in radians
      b_q: number or 1D numpy array
        co-latitude of a vector in radians

    Returns
    -------
      if l_q is a number, then return 1D array containing a vector
      if l_q is an 1D array, then return 2D array containing vectors as rows

      vec: 1D or 2D numpy array
        vector(s) corresponding to directions l_q,b_q

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.make_co_vec(np.radians(90.0),0.0))
      [ 0.  0.  1.]
      >>> n=10
      >>> l_q=np.radians(np.linspace(0,360,n))
      >>> b_q=np.radians(np.linspace(-90,90,n))
      >>> print(gflb.make_co_vec(l_q,b_q))
      [[ -1.00000000e+00  -0.00000000e+00   6.12323400e-17]
      [ -7.19846310e-01  -6.04022774e-01   3.42020143e-01]
      [ -1.33022222e-01  -7.54406507e-01   6.42787610e-01]
      [  2.50000000e-01  -4.33012702e-01   8.66025404e-01]
      [  1.63175911e-01  -5.93911746e-02   9.84807753e-01]
      [ -1.63175911e-01  -5.93911746e-02   9.84807753e-01]
      [ -2.50000000e-01  -4.33012702e-01   8.66025404e-01]
      [  1.33022222e-01  -7.54406507e-01   6.42787610e-01]
      [  7.19846310e-01  -6.04022774e-01   3.42020143e-01]
      [  1.00000000e+00  -2.44929360e-16   6.12323400e-17]]
    """
    if np.shape(l_q) != np.shape(b_q):
        raise Exception("make_co_vec() error: np.shape(l_q)!=np.shape(b_q)")
    if np.isscalar(l_q):
        return np.array([np.cos(l_q)*np.sin(b_q), np.sin(l_q)*np.sin(b_q), np.cos(b_q)])
    return np.column_stack([np.cos(l_q)*np.sin(b_q), np.sin(l_q)*np.sin(b_q), np.cos(b_q)])

def make_vec(l_q, b_q):
    """
    Compute vectors for given values of longitude l_q and latitude b_q

    Parameters
    ----------
      l_q: number or 1D numpy array
        longitude of a vector in radians
      b_q: number or 1D numpy array
        latitude of a vector in radians

    Returns
    -------
      if l_q is a number, then return 1D array containing a vector
      if l_q is an 1D array, then return 2D array containing vectors as rows

      vec: 1D or 2D numpy array
        vector(s) corresponding to directions l_q,b_q

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.make_vec(np.radians(90.0),0.0))
      [  6.12323400e-17   1.00000000e+00   0.00000000e+00]
      >>> n=10
      >>> l_q=np.radians(np.linspace(0,360,n))
      >>> b_q=np.radians(np.linspace(-90,90,n))
      >>> print(gflb.make_vec(l_q,b_q))
      [[  6.12323400e-17   0.00000000e+00  -1.00000000e+00]
       [  2.62002630e-01   2.19846310e-01  -9.39692621e-01]
       [  1.11618897e-01   6.33022222e-01  -7.66044443e-01]
       [ -4.33012702e-01   7.50000000e-01  -5.00000000e-01]
       [ -9.25416578e-01   3.36824089e-01  -1.73648178e-01]
       [ -9.25416578e-01  -3.36824089e-01   1.73648178e-01]
       [ -4.33012702e-01  -7.50000000e-01   5.00000000e-01]
       [  1.11618897e-01  -6.33022222e-01   7.66044443e-01]
       [  2.62002630e-01  -2.19846310e-01   9.39692621e-01]
       [  6.12323400e-17  -1.49975978e-32   1.00000000e+00]]
    """
    if np.shape(l_q) != np.shape(b_q):
        raise Exception("make_vec() error: np.shape(l_q)!=np.shape(b_q)")
    if np.isscalar(l_q):
        return np.array([np.cos(l_q)*np.cos(b_q), np.sin(l_q)*np.cos(b_q), np.sin(b_q)])
    return np.column_stack([np.cos(l_q)*np.cos(b_q), np.sin(l_q)*np.cos(b_q), np.sin(b_q)])

def position_angle(a_1, d_1, a_2, d_2):
    """
    Compute position angle assuming that the vector pointing towards a_1,d_1 is
    located on a circle in the sky centered at a_2,d_2. The position angle describes
    a position on the circle relative to the north pole direction. The function uses a
    formula for the relative position angle from Jean Meeus "Astronomical Algorithms", page 116.

    Parameters
    ----------
      a_1: number or 1D numpy array
        longitude of the first vector in radians
      d_1: number or 1D numpy array
        latitude of the first vector in radians
      a_2: number or 1D numpy array
        longitude of the second vector in radians
      d_2: number or 1D numpy array
        latitude of the second vector in radians

    Returns
    -------
      if input parameters are numbers, then return a number pa
      if input parameters are 1D numpy arrays, then return 1D numpy array pa

      pa: number or 1D numpy array
        position angle(s)

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(np.degrees(gflb.position_angle(np.radians(20.0),0.0,0.0,0.0)))
      90.0
      >>> n=10
      >>> l2=b2=np.zeros((n),dtype=float)
      >>> l1=np.radians(np.linspace(0,30,n))
      >>> b1=np.radians(30.0)-l1
      >>> print(np.degrees(gflb.position_angle(l1,b1,l2,b2)))
      [  0.           6.60406655  15.06329832  25.50555026  37.6074435
        50.43084061  62.72683044  73.55865702  82.60570856  90.        ]
    """
    delta_a = a_1-a_2
    pos_a = np.arctan2(np.sin(delta_a), np.cos(d_2)*np.tan(d_1)-np.sin(d_2)*np.cos(delta_a))
    return pos_a

def rot_two_vec_frame(l_c, b_c, l_x, b_x, l_q, b_q):
    """
    rotate (l_q,b_q) to a frame centered at (l_c,b_c) and x-axis coaligned with (l_c,b_c),(l_x,b_x)

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    """
    if np.isscalar(l_q):
        pos_ang = lb2spin(l_c, b_c, 0.0, l_x, b_x)[1]
    else:
        pos_ang = lb2spin(l_c, b_c, 0.0, l_x, b_x)[:, 1]
    # compute rotation matrix
    rot_mtrx = rzc_ryb_rza(-(2.0*np.pi-l_c), -(b_c-0.5*np.pi), pos_ang+np.pi)
    vec0 = make_vec(l_q, b_q) # temporary vectors to rotate
    if np.isscalar(l_q):
        vec = np.matmul(rot_mtrx, vec0) # rotate vec0
    else:
        vec = np.einsum('lij,lj->li', rot_mtrx, vec0)
    rlb = get_spher_coord_rad(vec)
    if np.isscalar(l_q):
        return np.array([-rlb[1], rlb[2]])
    return np.column_stack([-rlb[:, 1], rlb[:, 2]])

def rot_x(ang):
    """
    Construct a rotation matrix that rotates vectors by the angle ang about
    x axis, a convention is used here that for ang=pi/2 we should get e.g. [0,0,1]->[0,1,0]

    Parameters
    ----------
      ang: number or 1D numpy array
        angle(s) of rotation about x axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang
        rotation matrix or matrices

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.rot_x(np.radians(90.0)))
      [[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
       [  0.00000000e+00   6.12323400e-17   1.00000000e+00]
       [  0.00000000e+00  -1.00000000e+00   6.12323400e-17]]
      >>> print(gflb.rot_x(np.radians([0.0,45.0,90.0])))
      [[[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  0.00000000e+00  -0.00000000e+00   1.00000000e+00]]

       [[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   7.07106781e-01   7.07106781e-01]
        [  0.00000000e+00  -7.07106781e-01   7.07106781e-01]]

       [[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   6.12323400e-17   1.00000000e+00]
        [  0.00000000e+00  -1.00000000e+00   6.12323400e-17]]]
    """
    if np.isscalar(ang):
        rot_mtrx = [[1.0, 0.0, 0.0],
                    [0.0, np.cos(ang), np.sin(ang)],
                    [0.0, -np.sin(ang), np.cos(ang)]]
        rot_mtrx = np.array(rot_mtrx)
    else:
        n_ang = len(ang)
        zers = np.zeros((n_ang), float)
        ones = np.ones((n_ang), float)
        rot_mtrx = [[ones, zers, zers],
                    [zers, np.cos(ang), np.sin(ang)],
                    [zers, -np.sin(ang), np.cos(ang)]]
        rot_mtrx = np.array(rot_mtrx)
        rot_mtrx = rot_mtrx.transpose(2, 0, 1)
    return rot_mtrx

def rot_y(ang):
    """
    Construct a rotation matrix that rotates vectors by the angle ang about
    y axis, a convention is used here that for ang=pi/2 we should get e.g. [0,0,1]->[-1,0,0]

    Parameters
    ----------
      ang: number or 1D numpy array
        angle(s) of rotation about y axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang
        rotation matrix or matrices

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.rot_y(np.radians(90.0)))
      [[  6.12323400e-17   0.00000000e+00  -1.00000000e+00]
       [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
       [  1.00000000e+00   0.00000000e+00   6.12323400e-17]]
      >>> print(gflb.rot_y(np.radians([0.0,45.0,90.0])))
      [[[  1.00000000e+00   0.00000000e+00  -0.00000000e+00]
        [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

       [[  7.07106781e-01   0.00000000e+00  -7.07106781e-01]
        [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  7.07106781e-01   0.00000000e+00   7.07106781e-01]]

       [[  6.12323400e-17   0.00000000e+00  -1.00000000e+00]
        [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  1.00000000e+00   0.00000000e+00   6.12323400e-17]]]
    """
    if np.isscalar(ang):
        rot_mtrx = [[np.cos(ang), 0.0, -np.sin(ang)],
                    [0.0, 1.0, 0.0],
                    [np.sin(ang), 0.0, np.cos(ang)]]
        rot_mtrx = np.array(rot_mtrx)
    else:
        n_ang = len(ang)
        zers = np.zeros((n_ang), float)
        ones = np.ones((n_ang), float)
        rot_mtrx = [[np.cos(ang), zers, -np.sin(ang)],
                    [zers, ones, zers],
                    [np.sin(ang), zers, np.cos(ang)]]
        rot_mtrx = np.array(rot_mtrx)
        rot_mtrx = rot_mtrx.transpose(2, 0, 1)
    return rot_mtrx

def rot_z(ang):
    """
    Construct a rotation matrix that rotates vectors by the angle ang about
    z axis, a convention is used here that for ang=pi/2 we should get e.g. [1,0,0]->[0,-1,0]

    Parameters
    ----------
      ang: number or 1D numpy array
        angle(s) of rotation about z axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang
        rotation matrix or matrices

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.rot_z(np.radians(90.0)))
      [[  6.12323400e-17   1.00000000e+00   0.00000000e+00]
       [ -1.00000000e+00   6.12323400e-17   0.00000000e+00]
       [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
      >>> print(gflb.rot_z(np.radians([0.0,45.0,90.0])))
      [[[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
        [ -0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

       [[  7.07106781e-01   7.07106781e-01   0.00000000e+00]
        [ -7.07106781e-01   7.07106781e-01   0.00000000e+00]
        [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

       [[  6.12323400e-17   1.00000000e+00   0.00000000e+00]
        [ -1.00000000e+00   6.12323400e-17   0.00000000e+00]
        [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]]
    """
    if np.isscalar(ang):
        rot_mtrx = [[np.cos(ang), np.sin(ang), 0.0],
                    [-np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]]
        rot_mtrx = np.array(rot_mtrx)
    else:
        n_ang = len(ang)
        zers = np.zeros((n_ang), float)
        ones = np.ones((n_ang), float)
        rot_mtrx = [[np.cos(ang), np.sin(ang), zers],
                    [-np.sin(ang), np.cos(ang), zers],
                    [zers, zers, ones]]
        rot_mtrx = np.array(rot_mtrx)
        rot_mtrx = rot_mtrx.transpose(2, 0, 1)
    return rot_mtrx

def rxc_ryb_rza(ang_a, ang_b, ang_c):
    """
    Construct a rotation matrix that rotates vectors by the
    angle ang_a about z axis, then by the angle ang_b about y axis, then by the
    angle ang_c about x axis.

    Parameters
    ----------
      ang_a: number or 1D numpy array
        angle(s) of rotation about z axis in radians
      ang_b: number or 1D numpy array
        angle(s) of rotation about y axis in radians
      ang_c: number or 1D numpy array
        angle(s) of rotation about x axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang_a
        rotation matrix or matrices

    Examples
    --------
    """
    rot_mtrx = np.matmul(rot_x(ang_c), np.matmul(rot_y(ang_b), rot_z(ang_a)))
    return rot_mtrx

def rxc_rzb_rxa(ang_a, ang_b, ang_c):
    """
    Construct a rotation matrix that rotates vectors by the
    angle ang_a about x axis, then by the angle ang_b about z axis, then by the
    angle ang_c about x axis.

    Parameters
    ----------
      ang_a: number or 1D numpy array
        angle(s) of rotation about x axis in radians
      ang_b: number or 1D numpy array
        angle(s) of rotation about z axis in radians
      ang_c: number or 1D numpy array
        angle(s) of rotation about x axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang_a
        rotation matrix or matrices

    Examples
    --------
    """
    rot_mtrx = np.matmul(rot_x(ang_c), np.matmul(rot_z(ang_b), rot_x(ang_a)))
    return rot_mtrx

def rzb_rya(ang_a, ang_b):
    """
    Construct a rotation matrix that rotates vectors by the
    angle ang_a about y axis, then by the angle ang_b about z axis.

    Parameters
    ----------
      ang_a: number or 1D numpy array
        angle(s) of rotation about y axis in radians
      ang_b: number or 1D numpy array
        angle(s) of rotation about z axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang_a
        rotation matrix or matrices

    Examples
    --------
    """
    rot_mtrx = np.matmul(rot_z(ang_b), rot_y(ang_a))
    return rot_mtrx

def rzc_ryb_rza(ang_a, ang_b, ang_c):
    """
    Construct a rotation matrix that rotates vectors by the
    angle ang_a about z axis, then by the angle ang_b about y axis, then by the
    angle ang_c about z axis.

    Parameters
    ----------
      ang_a: number or 1D numpy array
        angle(s) of rotation about z axis in radians
      ang_b: number or 1D numpy array
        angle(s) of rotation about y axis in radians
      ang_c: number or 1D numpy array
        angle(s) of rotation about z axis in radians

    Returns
    -------
      if input parameter is a number, then return 3x3 rotation matrix
      if input parameter is a 1D numpy array, then return an array of 3x3 matrices

      rot_mtrx: 3x3 numpy array or Nx3x3 numpy array, where N is the length of the input array ang_a
        rotation matrix or matrices

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.rzc_ryb_rza(np.radians(90.0),np.radians(45.0),np.radians(0.0)))
      [[  4.32978028e-17   7.07106781e-01  -7.07106781e-01]
       [ -1.00000000e+00   6.12323400e-17   0.00000000e+00]
       [  4.32978028e-17   7.07106781e-01   7.07106781e-01]]
      >>> print(gflb.rzc_ryb_rza(np.radians([0.0,45.0,90.0]),np.radians([0.0,45.0,90.0]),\
                np.radians([0.0,45.0,90.0])))
      [[[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   1.00000000e+00   0.00000000e+00]
        [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

       [[ -1.46446609e-01   8.53553391e-01  -5.00000000e-01]
        [ -8.53553391e-01   1.46446609e-01   5.00000000e-01]
        [  5.00000000e-01   5.00000000e-01   7.07106781e-01]]

       [[ -1.00000000e+00   6.12323400e-17  -6.12323400e-17]
        [ -6.12323400e-17  -6.12323400e-17   1.00000000e+00]
        [  6.12323400e-17   1.00000000e+00   6.12323400e-17]]]
    """
    rot_mtrx = np.matmul(rot_z(ang_c), np.matmul(rot_y(ang_b), rot_z(ang_a)))
    return rot_mtrx

def spin2lb(l_0, b_0, spin_ang, spin_off, tht):
    """
    Computes longitude,latitude (l,b) for a given spin angle spin_ang.
    Assumptions: we have a circle of angular radius (half-angle) tht
    centered at l_0,b_0 in the sky map. Spin angle spin_ang measures the position
    angle on the circle relative to the north pole direction. Spin offset
    spin_off sets required offset of spin angle measurements with respect to
    the north-pole-direction convention described in Jean Meeus
    "Astronomical Algorithms", page 116.
    See IMAP resources at dt-300:/data/imap/shared/librBox/doc/spin2lb.pdf for details.

    Parameters
    ----------
      l_0: number or 1D numpy array
        longitude (in radians) of the center of the circle
      b_0: number or 1D numpy array
        latitude (in radians) of the center of the circle
      spin_ang: number or 1D numpy array
        spin angle (in radians) measuring position of l,b on a circle centered on l_0,b_0
      spin_off: number or 1D numpy array
        offset for the spin angle measurements in radians
      tht: number or 1D numpy array
        angular radius of the circle in radians

    Returns
    -------
      if input parameters are numbers, then return 1D array [l,b]
      if input parameters l_0,b_0,spin_ang,tht are 1D numpy arrays, then return 2D array [l,b]
        where columns represent l and b
      l: number or 1D numpy array
        longitude of a point on the circle
      b: number or 1D numpy array
        latitude of a point on the circle

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> spin_off=0.0
      >>> s0=np.radians(np.array(range(4))*360.0/4)
      >>> l_0=np.radians(100.0*np.ones((len(s0)),dtype=float))
      >>> b_0=np.radians(0.0*np.ones((len(s0)),dtype=float))
      >>> tht0=np.radians(40.0*np.ones((len(s0)),dtype=float))
      >>> print(np.degrees(gflb.spin2lb(l_0[0],b_0[0],s0[0],spin_off,tht0[0])))
      [ 100.   40.]
      >>> print(np.degrees(gflb.spin2lb(l_0,b_0,s0,spin_off,tht0)))
      [[  1.00000000e+02   4.00000000e+01]
       [  6.00000000e+01   4.94268248e-15]
       [  1.00000000e+02  -4.00000000e+01]
       [  1.40000000e+02  -4.07782511e-15]]
    """
    rot_mtrx = rzc_ryb_rza(-spin_ang-spin_off, b_0-0.5*np.pi, -l_0) # compute rotation matrix
    if np.isscalar(tht):
        vec0 = np.array([-np.sin(tht), 0.0, np.cos(tht)]) # temporary vector to rotate
        vec = np.matmul(rot_mtrx, vec0) # rotate vec0
    else:
        vec0 = np.column_stack([-np.sin(tht), np.zeros((len(tht)), dtype=float), np.cos(tht)])
        vec = np.einsum('lij,lj->li', rot_mtrx, vec0)
    rlb = get_spher_coord_rad(vec)
    if np.isscalar(l_0):
        return np.array([rlb[1], rlb[2]])
    return np.column_stack([rlb[:, 1], rlb[:, 2]])

def spin2lb_test_on_grid(spin_off, tht):
    """
    Test functions spin2lb() and lb2spin() for l_0,b_0,spin values on a grid of 11x11x11 cases
    for 0<=l_0<=2*pi, -pi/2<=b_0<=pi/2, 0<=spin<=2*pi
    See spin2lb() for more details.

    Parameters
    ----------
      spin_off: number or 1D numpy array
        offset for the spin angle measurements in radians, from 0 to 2*pi
      tht: number or 1D numpy array
        angular radius of a circle in radians, from 0 to pi

    Returns
    -------
      prints information on the screen if the test was passed or failed
      and returns 1 on passed and 0 on failed

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> print(gflb.spin2lb_test_on_grid(np.radians(270.0),np.radians(130.0)))
      spin2lb_test_on_grid(tht=2.26893,spin_off=4.71239): passed
      1
    """
    # create a grid of cases
    lin_spc_1 = np.linspace(0.0, 360.0, 11)
    lin_spc_2 = np.linspace(-90.0, 90.0, 11)
#    l_0, b_0, s_0 = np.radians(np.mgrid[0.0:360.0:11*1j, -90.0:90.0:11*1j, 0.0:360.0:11*1j])
    l_0, b_0, s_0 = np.radians(np.meshgrid(lin_spc_1, lin_spc_2, lin_spc_1))
    l_0 = l_0.flatten()
    b_0 = b_0.flatten()
    s_0 = s_0.flatten()
    tht_0 = tht*np.ones_like(s_0)

    lon_lat = spin2lb(l_0, b_0, s_0, spin_off, tht_0)
    tht1s = lb2spin(l_0, b_0, spin_off, lon_lat[:, 0], lon_lat[:, 1])
    tht_1 = tht1s[:, 0]
    spin_ang = tht1s[:, 1]

    ist = 0

    # check theta
    idxs = np.nonzero(np.fabs(tht_1-tht_0) > 1.0e-14)
    if np.shape(idxs[0])[0] > 0:
        ist = 1

    # check spin
    err_s = np.fabs(spin_ang-s_0)
    idxs = np.nonzero(np.fabs(err_s-2.0*np.pi) < 1.0e-14)
    err_s[idxs] = err_s[idxs]-2.0*np.pi
    if err_s.max() > 1.0e-14:
        ist = 1

    if ist == 0:
        print('spin2lb_test_on_grid(tht=%g,spin_off=%g): passed' % (tht, spin_off))
        return 1
    print('spin2lb_test_on_grid(tht=%g,spin_off=%g): failed' % (tht, spin_off))
    return 0

def spin2lb_test_on_text_file(filename):
    """
    Test spin2lb() and lb2spin() using content of a text file
    e.g. a file located in IMAP resources at
    dt-300:/data/imap/shared/develTests/spin2lb/spin2lb_test.txt
    See spin2lb() for more details.

    Parameters
    ----------
      filename: string
        name of the text file

    Returns
    -------
      prints information on the screen if the test was passed or failed
      and returns 1 on passed and 0 on failed

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> gflb.spin2lb_test_on_text_file('spin2lb_test.txt')
      spin2lb_test_on_text_file('spin2lb_test.txt'): passed
      1
    """
    # load a text file with testing cases
    l_0, b_0, s_0, spin_off, tht, l_q, b_q = np.radians(np.loadtxt(filename, unpack=True))

    # back-and-forth transforms
    ltbt = spin2lb(l_0, b_0, s_0, spin_off, tht)
    thttst = lb2spin(l_0, b_0, spin_off, ltbt[:, 0], ltbt[:, 1])
    thtt = thttst[:, 0]
    s_t = thttst[:, 1]

    ist = 0

    # check longitude
    idxs = np.nonzero(np.fabs(l_q-ltbt[:, 0]) > 1.0e-14)
    if np.shape(idxs[0])[0] > 0:
        ist = 1

    # check latitude
    idxs = np.nonzero(np.fabs(b_q-ltbt[:, 1]) > 1.0e-14)
    if np.shape(idxs[0])[0] > 0:
        ist = 1

    # check theta
    idxs = np.nonzero(np.fabs(tht-thtt) > 1.0e-14)
    if np.shape(idxs[0])[0] > 0:
        ist = 1

    # check spin
    err_s = np.fabs(s_t-s_0)
    idxs = np.nonzero(np.fabs(err_s-2.0*np.pi) < 1.0e-14)
    err_s[idxs] = err_s[idxs]-2.0*np.pi
    idxs = np.nonzero(err_s > 1.0e-14)
    if np.shape(idxs[0])[0] > 0:
        ist = 1

    if ist == 0:
        print('spin2lb_test_on_text_file(\'%s\'): passed' % filename)
        return 1
    print('spin2lb_test_on_text_file(\'%s\'): failed' % filename)
    return 0
