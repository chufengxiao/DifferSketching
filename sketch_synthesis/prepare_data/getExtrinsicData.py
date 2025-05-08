import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os,cv2,json,copy
from natsort import natsorted
from numpy.linalg import det
import math

def readStrokes(path):
    file = open(path,"r")
    json_str = file.read()
    strokes = json.loads(json_str)
    return strokes

def renderSketch(strokes):
    canvas = np.ones((800,800),dtype="uint8")*255

    for stroke in strokes['strokes']:
            
        if stroke['draw_type'] == 1:
            continue

        path = stroke['path']
        path = np.array(path).astype("int")
        for i in range(len(path)-1):
            cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),0,3)

    return canvas

def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return np.array(final)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def drawPath(path,color,canvas=None):
    if canvas is None:
        canvas = np.ones((800,800,3),dtype="uint8")*255
    path = path.astype('int')

    for i in range(len(path)-1):
        cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),color,2)
    return canvas


def showBezier(points,control_num):
    data = get_bezier_parameters(points[:,0], points[:,1], degree=control_num-1).tolist()

    xvals, yvals = bezier_curve(data, nTimes=500)

    # # Plot the control points
    # data = np.array(data)
    # x_val = data[:,0]
    # y_val = data[:,1]
    # plt.plot(points[:,0], points[:,1], "ro",label='Original Points')
    # plt.plot(x_val,y_val,'k--o', label='Control Points')
    # # Plot the resulting Bezier curve
    # plt.plot(xvals, yvals, 'b-', label='B Curve')
    # plt.legend()
    # plt.show()
    return xvals,yvals

def visualization(control_num,json_path):
    strokes = readStrokes(json_path)
    curves = copy.deepcopy(strokes)

    for i,stroke in enumerate(strokes['strokes']):
        if stroke['draw_type'] == 1:
            continue
        points = np.array(stroke['path'])
        if len(points) < control_num:
            continue
        canvas = drawPath(points,0)
        x,y = showBezier(points,control_num)
        curve = np.array([x,y]).T
        curves['strokes'][i]['path'] = curve.astype('int')
    
    origin = renderSketch(strokes)
    bezier = renderSketch(curves)

    plt.subplot(121)
    plt.imshow(origin,'gray')
    plt.subplot(122)
    plt.imshow(bezier,'gray')
    plt.show()

# get normalized R,T,S
def getRST(M):
    R_s = M[:2,:2]
    T = M[:,-1].T
    if (det(R_s)**0.5) < 0.0000001:
        return None
    R = R_s / (det(R_s)**0.5)

    if R[0,0] < 0.0000001:
        return None

    if abs(R[0,0]-1) < 0.00001:
        R[0,0] = 0.999
        R[1,1] = 0.999

    theta = math.asin(R[0,1]) * (180/np.pi)

    scale = R_s[0,0] / R[0,0]

    theta = (np.clip(theta,-45,45)+45)/90
    T = (np.clip(T,-800,800)+800)/1600
    scale = (np.clip(scale,0.1,2.1)-0.1)/2

    return theta,scale,T

# get transform matrix from R,T,S
def RST2M(R,S,T):
    print(R,S,T)
    S = S * 2+0.1
    R = (R * 90 -45) * (np.pi/180)
    T = T * 1600 - 800
    alpha,beta = math.cos(R)*S, math.sin(R)*S
    M = [[alpha,beta,T[0]],\
         [-beta,alpha,T[1]]]
    return np.array(M,dtype='float')
    

def getStrokeData(control_num,srcStrokes,dstStrokes,strokePrec):
    in_curves = []
    out_RST = []
    srcStrokes = srcStrokes['strokes']
    dstStrokes = dstStrokes['strokes']
    canvas = np.ones((800,800,3),dtype='uint8')*255
    for i,stroke in enumerate(srcStrokes):

        if stroke['draw_type'] == 1:
            continue
        if strokePrec[i] < 0.7:
            continue
        src_p = np.array(stroke['path']).astype('int')
        dst_p = np.array(dstStrokes[i]['path']).astype('int')
        if len(src_p) < control_num:
            continue
        assert src_p.shape[0] == dst_p.shape[0]
        
        # M = cv2.estimateRigidTransform(src_p, dst_p,fullAffine=False)
        M = cv2.estimateAffinePartial2D(src_p,dst_p)[0]

        if M is None:
            continue

        RST = getRST(M)

        if RST is None:
            continue
        r,s,t = RST

        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)
        
        # new_M = RST2M(r,s,t)
        # reg_path = cv2.transform(src_p[:,np.newaxis,:], new_M).squeeze()
        # src_points = bezier_curve(inputs.tolist(), nTimes=500)
        # src_points = np.array(src_points).T
        # canvas = drawPath(src_p,0,canvas)
        # canvas = drawPath(dst_p,(255,0,0),canvas)
        # canvas = drawPath(reg_path,(0,255,0),canvas)
        # plt.imshow(canvas)
        # plt.show()

        inputs = np.round(inputs).astype('int').flatten()
        inputs = inputs.tolist()

        dist = abs(r-0.5)+abs(s-0.5)+np.abs(np.array(t)-0.5).mean()
        inputs.append(dist)

        outputs = [r,s,t[0],t[1]]

        in_curves.append(inputs)
        out_RST.append(outputs)
        
    return in_curves,out_RST

if __name__ == "__main__":
    NP = 'P'  # switch between novice and professional sketches using 'N' or 'P'

    cg_list = ['Primitive','Chair','Lamp','Industrial_Component','Shoe','Animal','Animal_Head','Vehicle','Human_Face']

    control_num = 6 # number of control points for bezier curves
    root_dir = "../DifferSketching_Dataset" # the sketch of DifferSketching Dataset
    fail_list = open(os.path.join(root_dir,'eval_lower_1.2.txt')).read().split('\n') # filter out the failure cases of registration

    save_folder = "./data/extrinsic/"
    os.makedirs(save_folder,exist_ok=True)
    
    total_in_curves, total_out_M = [], []
    for cg in cg_list:
        src_dir = os.path.join(root_dir,"%s/stroke_json/"%cg)
        dst_dir = os.path.join(root_dir,"%s/global_json/"%cg)

        strokePrec_list = readStrokes(os.path.join(root_dir,'%s/regPrec_dict.json'%cg))

        json_list = natsorted(os.listdir(dst_dir))
        for i,json_path in enumerate(json_list):
            if NP != '' and not json_path.startswith(NP):
                continue
            if json_path.split('.')[0] in fail_list:
                continue

            print(cg,i,json_path,end='\r')
            strokePrec = strokePrec_list[json_path]
            srcStrokes = readStrokes(os.path.join(src_dir,json_path))
            dstStrokes = readStrokes(os.path.join(dst_dir,json_path))

            in_cur, out_rst = getStrokeData(control_num,srcStrokes,dstStrokes,strokePrec)
            
            total_in_curves += in_cur
            total_out_M += out_rst
            # total_out_M += out_curves
            # visualization(control_num,os.path.join(src_dir,json_path))

        print('\n len:',len(total_in_curves))

    np.savetxt(os.path.join(save_folder,'in_curves_%d_%s.txt'%(control_num,NP)),total_in_curves)
    np.savetxt(os.path.join(save_folder,'out_%d_%s.txt'%(control_num,NP)),total_out_M)

