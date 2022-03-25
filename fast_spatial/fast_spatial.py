from numba import njit
import numpy as np
from time import time

# R : Rotational matrix (3x3 np.ndarray)
# qtn : quaternion [(x,y,z),w]
# rpy : ZYX euler angle (roll, pitch, yaw)
# axisangle : exponential coordinate of R
# T : Transformation matrix (4x4 np.ndarray)
# screwangle : exponential coordinate of T
# posqtn : position, quaternion

@njit
def allclose(a: np.ndarray, b: np.ndarray, tol=1e-8):
    return np.all(np.abs(a - b) < tol)

def R_inv(R: np.ndarray) -> np.ndarray:
    return R.T

@njit
def qtn_inv(qtn):
    x, y, z, w = qtn
    return np.array((-x, -y, -z, w))

@njit
def T_inv(T):
    R, t = T_to_Rt(T)
    return Rt_to_T(R_inv(R), -R.T@t)

@njit
def get_random_rpy():
    rnd = np.random.random(4)
    roll = 2*np.pi*rnd[0] - np.pi
    pitch = np.arccos(1 - 2*rnd[1]) + np.pi/2
    if rnd[2] < 1/2:
        if pitch < np.pi:
            pitch += np.pi
        else:
            pitch -= np.pi
    if pitch > np.pi:
        pitch -= np.pi*2
    yaw = 2*np.pi*rnd[3] - np.pi
    return np.array((roll, pitch, yaw))

@njit
def skew(vec: np.ndarray) -> np.ndarray:
    mat = np.zeros((3,3))
    mat[1,0] = vec[2]
    mat[0,1] = -vec[2]
    mat[2,0] = -vec[1]
    mat[0,2] = vec[1]
    mat[2,1] = vec[0]
    mat[1,2] = -vec[0]
    return mat

@njit
def adjoint(T):
    R, t = T_to_Rt(T)
    adj = np.eye(6)
    adj[:3,:3] = R
    adj[3:,:3] = skew(t)@R
    adj[:3,:3] = R
    return adj

@njit
def qtn_normalize(qtn: np.ndarray) -> np.ndarray:
    return qtn/np.linalg.norm(qtn)

@njit
def qtn_to_R(qtn: np.ndarray) -> np.ndarray:
    """[summary]
    Convert Quaternion to Rotation matrix 
    Args:
        qtn (size3 np.ndarray): quaternion
    Returns:
        [3x3 np.ndarray]: Rotation Matrix
    """
    qtn_norm = np.linalg.norm(qtn)
    if qtn_norm != 1:
        x, y, z, w = qtn/qtn_norm
    else:
        x, y, z, w = qtn
    R = np.empty((3,3))
    R[0,0] = 1-2*y**2-2*z**2
    R[1,0] = 2*x*y+2*w*z
    R[2,0] = 2*x*z-2*w*y
    R[0,1] = 2*x*y-2*w*z
    R[1,1] = 1-2*x**2-2*z**2
    R[2,1] = 2*y*z+2*w*x
    R[0,2] = 2*x*z+2*w*y
    R[1,2] = 2*y*z-2*w*x
    R[2,2] = 1-2*x**2-2*y**2
    return R

@njit
def R_to_qtn(R: np.ndarray) -> np.ndarray:
    """[summary]
    Convert rotation matrix to unit quaternion
    Args:
        R (3x3 np.ndarray): Rotation matrix.
    Returns:
        qtn [size 3 np.ndarray]: Unit quaternion.
    """
    tr = np.trace(R)
    if tr > 0.:
        s = np.sqrt(tr+1.)*2
        w = 0.25*s
        x = (R[2,1] - R[1,2])/s
        y = (R[0,2] - R[2,0])/s
        z = (R[1,0] - R[0,1])/s
    elif (R[0,0]>R[1,1]) & (R[0,0] > R[2,2]):
        s= np.sqrt(1. + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s; 
        z = (R[0,2] + R[2,0]) / s; 
    elif R[1,1] > R[2,2]:
        s= np.sqrt(1. + R[1,1] - R[0,0] - R[2,2]) * 2 
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1. + R[2,2] - R[1,1] - R[0,0]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    qtn = np.empty(4)
    qtn[:] = (x, y, z, w)
    return qtn

@njit
def axisangle_to_R(axis, angle):
    """[summary]
    Convert AxisAngle to Rotation Matrix.
    (Exponential of SO3)
    Args:
        axis (size3 np.ndarray or list): unit vector of rotation axis
        angle (float): rotation angle
    Returns:
        R [3x3 np.ndarray]: rotation matrix
    """
    if angle == 0.:
        return np.eye(3)
    else:
        axis_norm = np.linalg.norm(axis)
        if axis_norm != 1.:
            axis = axis/axis_norm
        theta = angle
        omega_hat = axis
        return np.eye(3) \
            + np.sin(theta)*skew(omega_hat) \
            + (1-np.cos(theta))*skew(omega_hat)@skew(omega_hat)

@njit
def R_to_axisangle(R):
    """[summary]
    Convert rotation matrix to unit quaternion
    (logarithm of SO3)
    Args:
        R (3x3 np.ndarray): Rotation matrix.
    Returns:
        axis [size 3 np.ndarray]: Unit axis.
        angle (float): rotation angle
    """
    if allclose(R, np.eye(3)):
        # no rotation
        axis = np.array((1., 0., 0.))
        angle = 0.
    elif np.trace(R) == -1:
        # angle is 180 degrees
        angle = np.pi
        if R[0,0] != -1:
            axis = 1/np.sqrt(2*(1+R[0,0])) * np.array([1+R[0,0], R[1,0], R[2,0]])
        elif R[1,1] != -1:
            axis = 1/np.sqrt(2*(1+R[1,1])) * np.array([R[0,1], 1+R[1,1], R[2,1]])
        else:
            axis = 1/np.sqrt(2*(1+R[2,2])) * np.array([R[0,2], R[1,2], 1+R[2,2]])
    else:
        angle = np.arccos(1/2*(np.trace(R)-1))
        axis = 1/(2*np.sin(angle))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return axis, angle

@njit
def qtn_to_axisangle(qtn):
    """[summary]
    Convert quaternion to axisangle
    Args:
        qtn (np.ndarray(4)): unit quaternion
    Returns:
        axis [size 3 np.ndarray]: Unit axis.
        angle (float): rotation angle
    """
    qtn = qtn_normalize(qtn)
    x, y, z, w = qtn
    angle = 2 * np.arccos(w)
    s = np.sqrt(1-w**2)
    if s < 1e-8:
        axis = np.array((x, y, z))
        return axis, angle
    else:
        axis = np.array((x, y, z))
        axis = axis/np.linalg.norm(axis)
        return axis, angle

@njit
def axisangle_to_qtn(axis, angle):
    """[summary]
    Convert axisangle to quaternion
    Args:
        axis [size 3 np.ndarray]: Unit axis.
        angle (float): rotation angle
    Returns:
        qtn (np.ndarray(4)): unit quaternion
        
    """
    s = np.sin(angle/2)
    w = np.cos(angle/2)
    x, y, z = axis * s
    return np.array((x,y,z,w))

@njit
def rpy_to_qtn(rpy):
    """[summary]
    Convert rpy to quaternion
    Args:
        rpy [np.ndarray(3)]: euler-angle representation(roll, pitch, yaw).
    Returns:
        qtn (np.ndarray(4)): unit quaternion
        
    """
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll/2), np.cos(roll/2)
    sp, cp = np.sin(pitch/2), np.cos(pitch/2)
    sy, cy = np.sin(yaw/2), np.cos(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    qtn = np.array((x, y, z, w))
    return qtn

@njit
def qtn_to_rpy(qtn):
    """[summary]
    Convert quaternion to rpy
    Args:
        qtn (np.ndarray(4)): unit quaternion
    Returns:
        rpy [np.ndarray(3)]: euler-angle representation(roll, pitch, yaw).
        
    """
    qtn = qtn_normalize(qtn)
    x, y, z, w = qtn
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if (abs(sinp) >= 1):
        pitch = np.sign(sinp)*np.pi/2
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    rpy = np.array((roll, pitch, yaw))
    return rpy
 
@njit
def Rt_to_T(R, t):
    """[summary]
    Args:
        R ([type]): [description]
        t ([type]): [description]
    Returns:
        [type]: [description]
    """
    T = np.eye(4)
    for i in range(3):
        T[i,:3] = R[i,:]
    T[:3,-1] = t
    return T

@njit
def T_to_Rt(T):
    """[summary]
    Args:
        T ([type]): [description]
    Returns:
        [type]: [description]
    """
    R = np.ascontiguousarray(T[:3,:3])
    t = np.ascontiguousarray(T[:3,3])
    return R, t

@njit
def posqtn_to_T(pos: np.ndarray, qtn):
    """[summary]
    Args:
        qtn ([type]): [description]
        t ([type]): [description]
    Returns:
        [type]: [description]
    """
    R = qtn_to_R(qtn)
    return Rt_to_T(R, pos)

@njit
def T_to_screwangle(T):
    """[summary]
    Args:
        T ([type]): [description]
    Returns:
        [type]: [description]
    """
    R, t = T_to_Rt(T)
    if allclose(T, np.eye(4)):
        tw = np.array((1.,0.,0.,0.,0.,0))
        theta = 0
    elif allclose(R, np.eye(3)):
        # pure translation
        omega = np.zeros(3)
        theta = np.linalg.norm(t)
        v = t/theta
        tw = np.hstack((omega, v))
    else:
        omega, theta = R_to_axisangle(R)
        Ginv = 1/theta*np.eye(3) \
            - 1/2*skew(omega) \
            + (1/theta-1/2/np.tan(theta/2))*skew(omega).dot(skew(omega))
        v = Ginv @ t
        tw = np.hstack((omega, v))
    return tw, theta

@njit
def screwangle_to_T(screw, angle):
    """[summary]
    Convert Twist-angle to R, p
    Args:
        tw (size:6 np.ndarray): twist
        angle (float) : angle
    Returns:
        [type]: [description]
    """
    omega, v = screw[:3], screw[3:]
    theta = angle
    G = np.eye(3)*theta \
        + (1-np.cos(theta))*skew(omega) \
        + (theta-np.sin(theta))*skew(omega)@skew(omega)

    if np.linalg.norm(omega) == 0:
        R = np.eye(3)
        t = v*theta
    else:
        R = axisangle_to_R(omega, theta)
        t = G@v
    
    return Rt_to_T(R, t)

@njit
def posqtn_to_screwangle(pos, qtn):
    """[summary]
    Args:
        T ([type]): [description]
    Returns:
        [type]: [description]
    """

    if allclose(qtn, np.array((0,0,0,1))):
        if np.linalg.norm(pos) < 1e-8:
            #identity
            screw = np.array((1.,0.,0.,0.,0.,0.))
            theta = 0
        else:
            # pure translation
            omega = np.zeros(3)
            theta = np.linalg.norm(pos)
            v = pos/theta
            screw = np.hstack((omega, v))
    else:
        omega, theta = qtn_to_axisangle(qtn)
        Ginv = 1/theta*np.eye(3) \
            - 1/2*skew(omega) \
            + (1/theta-1/2/np.tan(theta/2))*skew(omega)@skew(omega)
        v = Ginv@pos
        screw = np.hstack((omega, v))
    return screw, theta

@njit
def screwangle_to_posqtn(screw, angle):
    """[summary]
    Convert Twist-angle to R, p
    Args:
        tw (size:6 np.ndarray): twist
        angle (float) : angle
    Returns:
        [type]: [description]
    """
    T = screwangle_to_T(screw, angle)
    R, t = T_to_Rt(T)
    qtn = R_to_qtn(R)
    return t, qtn


def slerp(qtn1, qtn2, ratio):
    pass #TBD

@njit
def lerp(a, b, ratio):
    return a + (b - a) * ratio


def perf_timer(func, *args):
    func(*args)
    t1 = time()
    for i in range(100):
        func(*args)
    t2 = time()
    print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')

if __name__ == "__main__":
    perf_timer(get_random_rpy)