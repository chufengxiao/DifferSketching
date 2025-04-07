import numpy as np
import json,cv2

from scipy.special import comb
import matplotlib.pyplot as plt


def readStrokes(path):
    file = open(path,"r")
    json_str = file.read()
    strokes = json.loads(json_str)
    return strokes

def drawPath(gt=None,pred=None,input=None):
    def draw(canvas,path,color):
        path = path.astype('int')
        for i in range(len(path)-1):
            cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),color,2)
        return canvas
    canvas = np.ones((800,800,3),dtype="uint8")*255
    if input is not None:
        canvas = draw(canvas,input,0)
    if gt is not None:
        canvas = draw(canvas,gt,(0,255,0))
    if pred is not None:
        canvas = draw(canvas,pred,(255,0,0))
    
    plt.imshow(canvas)
    plt.show()
    return canvas

def bezier_curve(points, nTimes=500):

    def bernstein_poly(i, n, t):
        """
        The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    points = points.reshape(-1,2)
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    path = np.array([xvals,yvals]).T.astype('int')

    return path

def writeJSON(json_path,data):
    with open(json_path,'w+') as f:
        f.write(json.dumps(data))

def get_bezier_parameters(X, Y, degree=3):
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

def getStrokeBezier(control_num,srcStrokes):
    
    in_curves = []

    for i,stroke in enumerate(srcStrokes['strokes']):
        if 'draw_type' in stroke and stroke['draw_type'] == 1:
            continue
        src_p = np.array(stroke['path'])
        if len(src_p) < control_num:
            continue

        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)
        inputs = np.round(inputs).astype('int').flatten()/799
        
        in_curves.append(inputs)
    return np.array(in_curves)

def renderSketch(strokes):
    canvas = np.ones((800,800),dtype="uint8")*255

    for stroke in strokes['strokes']:
            
        if 'draw_type' in stroke and stroke['draw_type'] == 1:
            continue

        path = stroke['path']
        path = np.array(path).astype("int")
        for i in range(len(path)-1):
            cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),0,3)

    return canvas

def saveJson(py_data,save_path):
    json_str = json.dumps(py_data)
    file = open(save_path,"w+")
    file.write(json_str)
    file.close()
