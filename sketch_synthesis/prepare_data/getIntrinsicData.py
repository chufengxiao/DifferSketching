import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os,cv2,json,copy
from natsort import natsorted

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

def getStrokeBezier(control_num,srcStrokes,dstStrokes,strokePrec):
    in_curves = []
    out_curves = []
    srcStrokes = srcStrokes['strokes']
    dstStrokes = dstStrokes['strokes']
    
    for i,stroke in enumerate(srcStrokes):
        if stroke['draw_type'] == 1:
            continue
        if strokePrec[i] < 0.7:
            continue
        src_p = np.array(stroke['path'])
        if len(src_p) < control_num:
            continue
        dst_p = np.array(dstStrokes[i]['path'])

        outputs = get_bezier_parameters(dst_p[:,0], dst_p[:,1], degree=control_num-1)
        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)

        avg_dist = np.round(np.mean(np.sum((outputs-inputs)**2,axis=1)**0.5)).astype('int')
        
        # src_points = bezier_curve(inputs.tolist(), nTimes=500)
        # src_points = np.array(src_points).T
        # canvas = drawPath(src_points,0)
        # dst_points = bezier_curve(outputs.tolist(), nTimes=500)
        # dst_points = np.array(dst_points).T
        # canvas = drawPath(dst_points,(255,0,0),canvas)
        # plt.imshow(canvas)
        # plt.show()

        inputs = np.round(inputs).astype('int').flatten()
        outputs = np.round(outputs).astype('int').flatten()
        
        inputs = inputs.tolist()
        inputs.append(avg_dist)
        in_curves.append(inputs)
        out_curves.append(outputs.tolist())
        
    return in_curves,out_curves
    
if __name__ == "__main__":
    NP = 'P'
    # NP = '_N' # switch between novice and professional sketches

    cg_list = ['Primitive','Chair','Lamp','Industrial_Component','Shoe','Animal','Animal_Head','Vehicle','Human_Face']

    control_num = 6 # number of control points for bezier curves

    root_dir = "../DifferSketching_Dataset" # the sketch of DifferSketching Dataset
    fail_list = open(os.path.join(root_dir,'eval_lower_1.2.txt')).read().split('\n') # filter out the failure cases of registration

    save_folder = "./data/intrinsic/"
    os.makedirs(save_folder,exist_ok=True)
    
    total_in_curves, total_out_curves = [], []
    for cg in cg_list:
        in_curves, out_curves = [], []
        src_dir = os.path.join(root_dir,"%s/reg_json/"%cg)
        dst_dir = os.path.join(root_dir,"%s/stroke_json/"%cg)
        strokePrec_list = readStrokes(os.path.join(root_dir,'%s/regPrec_dict.json'%cg)) # filter out the strokes with low registration precision

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

            in_cur, out_cur  = getStrokeBezier(control_num,srcStrokes,dstStrokes,strokePrec)
            in_curves += in_cur
            out_curves += out_cur

            # visualization(control_num,os.path.join(src_dir,json_path))
        print('\n len:',len(in_curves))

        total_in_curves += in_curves
        total_out_curves += out_curves

    np.savetxt(os.path.join(save_folder,'in_curves_%d_%s.txt'%(control_num,NP)),total_in_curves,'%d')
    np.savetxt(os.path.join(save_folder,'out_curves_%d_%s.txt'%(control_num,NP)),total_out_curves,'%d')

