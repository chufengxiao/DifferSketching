import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os,cv2,json,copy
from natsort import natsorted
from scipy.ndimage.morphology import distance_transform_edt
from scipy import spatial

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
        cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),color,3)
    return canvas


def showBezier(points,control_num):
    data = get_bezier_parameters(points[:,0], points[:,1], degree=control_num-1).tolist()

    xvals, yvals = bezier_curve(data, nTimes=1000)

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

def getStrokeBezier1(control_num,srcStrokes):
    in_curves = []
    out = []
    srcStrokes = srcStrokes['strokes']
    
    for i,stroke in enumerate(srcStrokes):
        if stroke['draw_type'] == 1:
            continue

        src_p = np.array(stroke['path'])
        if len(src_p) < control_num:
            continue

        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)
        
        b_points = bezier_curve(inputs.tolist(), nTimes=len(src_p))
        b_points = np.array(b_points).T

        tree = spatial.cKDTree(src_p)
        mindist, minid = tree.query(b_points)
        src_p = src_p[minid]
        offset = src_p - b_points
        outputs = (np.clip(offset,-20,20)+20)/40

        idx = np.arange(0,len(offset),1)/len(offset)
        idx = idx[:,np.newaxis]
        inputs_loc = np.array(inputs).flatten()[np.newaxis,:]/799

        inputs_loc = np.repeat(inputs_loc,repeats=len(offset),axis=0)
        inputs_all = np.concatenate((inputs_loc,idx),axis=1)
        

        # canvas = drawPath(b_points,0)
        # canvas = drawPath(src_p,(255,0,0),canvas)
        # plt.imshow(canvas)
        # plt.show()
        
        in_curves += inputs_all.tolist()
        out += outputs.tolist()
    print(len(in_curves),len(out))
    return in_curves,out

def getOffset(mu1,std1,mu2,std2,b_points):
    mu1,std1 = mu1*20-10, std1*5
    mu2,std2 = mu2*20-10, std2*5

    rdn_x = np.random.normal(mu1,std1,len(b_points))[:,np.newaxis]
    rdn_y = np.random.normal(mu2,std2,len(b_points))[:,np.newaxis]
    rdn_offset = np.concatenate((rdn_x,rdn_y),axis=1)
    canvas = drawPath(b_points+rdn_offset,0)
    plt.imshow(canvas)
    plt.show()

def getStrokeBezier(control_num,srcStrokes):
    in_curves = []
    out = []
    srcStrokes = srcStrokes['strokes']
    
    for i,stroke in enumerate(srcStrokes):
        if stroke['draw_type'] == 1:
            continue

        src_p = np.array(stroke['path'])
        if len(src_p) < control_num:
            continue

        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)
        
        b_points = bezier_curve(inputs.tolist(), nTimes=len(src_p))
        b_points = np.array(b_points).T

        tree = spatial.cKDTree(src_p)
        mindist, minid = tree.query(b_points)
        src_p = src_p[minid]
        offset = src_p - b_points
        mu1,std1 = np.mean(offset[:,0]),np.std(offset[:,0])
        mu2,std2 = np.mean(offset[:,1]),np.std(offset[:,1])

        mu1,std1 = (np.clip(mu1,-10,10)+10)/20, np.clip(std1,0,5)/5
        mu2,std2 = (np.clip(mu2,-10,10)+10)/20, np.clip(std2,0,5)/5
        outputs = np.array([mu1,std1,mu2,std2])
        inputs_loc = np.array(inputs).flatten()/799

        # print(inputs_loc)
        # print(outputs)
        # getOffset(mu1,std1,mu2,std2,b_points)

        in_curves.append(inputs_loc.tolist())
        out.append(outputs.tolist())

    return in_curves,out
    
if __name__ == "__main__":
    
    cg_list = ['Primitive','Chair','Lamp','Industrial_Component','Shoe','Animal','Animal_Head','Vehicle','Human_Face']

    control_num = 6

    root_dir = "../DifferSketching_Dataset" # the sketch of DifferSketching Dataset
    
    save_folder = "./data/curve_noise/"
    os.makedirs(save_folder,exist_ok=True)
    
    total_in_curves, total_out = [], []
    for cg in cg_list:

        src_dir = os.path.join(root_dir,"%s/global_json/"%cg)
        
        json_list = natsorted(os.listdir(src_dir))
        for i,json_path in enumerate(json_list):
            print(cg,i,json_path,end='\r')

            srcStrokes = readStrokes(os.path.join(src_dir,json_path))

            in_cur,output = getStrokeBezier(control_num,srcStrokes)

            total_in_curves += in_cur
            total_out += output

        print('\n len:',len(total_in_curves))

        
    np.savetxt(os.path.join(save_folder,'in_all.txt'),total_in_curves)
    np.savetxt(os.path.join(save_folder,'out_all.txt'),total_out)

