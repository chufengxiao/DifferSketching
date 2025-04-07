#%%
import numpy as np
import os,json,cv2,math
import matplotlib.pyplot as plt
import pickle,copy

from components import readStrokes, get_bezier_parameters,bezier_curve

from scipy import spatial
from scipy.special import softmax
from scipy.optimize import minimize,Bounds

def drawCanvas(path,canvas,color=0):
    canvas = np.array(canvas)
    path = np.array(path).astype('int')
    for i in range(len(path)-1):
        cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),color,3)

    return canvas

def optimize(path,target_list,idx_list):
    def L2(src,dst):
        item = np.sum((src-dst)**2,axis=1)**0.5

        return np.sum(item)

    def shapeF(path):
        valid_p = path[1:-1:5]
        pre_p = path[:-2:5]
        post_p = path[2::5]

        shape_feat = valid_p - 0.5*(pre_p+post_p)
        return shape_feat

    def smoothTerm(path,fix_idx):
        error = 0
        idx_list = np.where(fix_idx==1)[0]
        for idx in idx_list:
            front = max(0,idx-10)
            back = min(len(path),idx+10)
            
            pre_p = path[front:back-1]
            post_p = path[front+1:back]
  
            error += L2(pre_p,post_p)
        return error

    def objf(pred_path,origin_path,fix_points,fix_idx):
        pred_path = pred_path.reshape((2,-1)).T

        ## position term
        fix_error = L2(pred_path*fix_idx,fix_points*fix_idx)
        # rest_error = L2(pred_path*(1-fix_idx),origin_path*(1-fix_idx))
        P_term = fix_error*2000 #+ rest_error

        ## shape term
        gt_shape = shapeF(origin_path)
        pred_shape = shapeF(pred_path)
        S_term = L2(gt_shape,pred_shape)

        M_term = 500*smoothTerm(pred_path,fix_idx)
        # print(P_term,S_term,M_term)


        return P_term + M_term + S_term

    # canvas = np.ones((800,800,3),dtype="uint8")*255
    # canvas = drawCanvas(path,canvas,0)
    # path = np.round(path).astype('int')
    # for idx in range(len(idx_list)):
    #     canvas = cv2.circle(canvas,tuple(target_list[idx]),4,(255,0,0),3)
    #     canvas = cv2.circle(canvas,tuple(path[idx_list[idx]]),3,(0,255,0),3)
    #     cv2.line(canvas,tuple(target_list[idx]),tuple(path[idx_list[idx]]),(127,127,127),2)
    # plt.subplot(121)
    # plt.imshow(canvas)

    target_list = np.array(target_list)

    f_path = np.array(path).T.flatten()
    fix_points = np.array(path)
    fix_points[idx_list] = target_list
    fix_idx = np.zeros((len(fix_points),1),dtype='int')
    fix_idx[idx_list] = 1

    result = minimize(objf,f_path,args=(path,fix_points,fix_idx),method='L-BFGS-B',bounds=Bounds(0,799))
    # 'SLSQP'
    output = np.round(result.x).astype('int')
    output = output.reshape((2,-1)).T

    # canvas = drawCanvas(output,canvas,(255,0,0))
    # plt.subplot(122)
    # plt.imshow(canvas)
    # plt.show()
    return output

def getValidStrokes(strokes):
    validStrokes = []
    for stroke in strokes['strokes']:
        if 'draw_type' in stroke and (stroke['draw_type'] != 0 or len(stroke['path']) < 6):
            continue
        path = np.array(stroke['path'])

        _, idx = np.unique(path,axis=0,return_index=True)

        if len(idx) > 6:
            path = path[sorted(idx)]

        validStrokes.append(path)
    return validStrokes

def getStrokeBezier(control_num,srcStrokes):
    
    in_curves = []

    for i,src_p in enumerate(srcStrokes):
        if len(src_p) < control_num:
            continue
        src_p = np.array(src_p)
        inputs = get_bezier_parameters(src_p[:,0], src_p[:,1], degree=control_num-1)
        inputs = np.round(inputs).astype('int').flatten()/799
        
        in_curves.append(inputs)
    return np.array(in_curves)

def renderSketch(strokes):
    canvas = np.ones((800,800,3),dtype="uint8")*255
    cmap = plt.cm.jet

    colors = cmap(np.linspace(0, 1, len(strokes)))
    colors = np.array(colors[:,:3]*255,dtype="uint8").tolist()

    s_num = 0
    for j,stroke in enumerate(strokes):
        
        path = np.array(stroke).astype("int")
        
        for i in range(len(path)-1):
            cv2.line(canvas,(path[i][0],path[i][1]),(path[i+1][0],path[i+1][1]),colors[s_num],3)
        s_num += 1

    return canvas

def writeJSON(strokes,json_path):
    saveStrokes = []
    for stroke in strokes:
        saveStrokes.append({'path':stroke.tolist()})
    saveStrokes = {'strokes':saveStrokes}
    with open(json_path,'w+') as f:
        f.write(json.dumps(saveStrokes))
        print('write',json_path)

class Synthesis:
    def __init__(self,NP='N'):
        
        self.mlp1 = pickle.load(open('./model/%s_mlp_intrinsic.sav'%NP, 'rb'))
        self.mlp2 = pickle.load(open('./model/%s_mlp_extrinsic.sav'%NP, 'rb'))
        self.mlp3 = pickle.load(open('./model/mlp_curveNoise.sav', 'rb'))
        self.cp = 6
        self.srcStrokes = None
        

    def load(self,srcStrokes):
        self.srcStrokes = srcStrokes
        b_curves = getStrokeBezier(self.cp,self.srcStrokes)
        srcCurve = np.round(b_curves*799).astype('int')
        self.srcCurve = self.renderCurves(srcCurve)

        # src_img = renderSketch(self.srcCurve)
        # plt.imshow(src_img)
        # plt.show()


    def getInCurve(self,strokes,n1,n2):
        in_cur = getStrokeBezier(self.cp,strokes)
        dist1 = np.ones((len(in_cur),1))*n1
        dist2 = np.ones((len(in_cur),1))*n2
        self.input1 = np.concatenate((in_cur,dist1),axis=1)
        self.input2 = np.concatenate((in_cur,dist2),axis=1)
        
        return in_cur

    def predict1(self,strokes,noise):
        in_cur = getStrokeBezier(self.cp,strokes)
        dist = np.ones((len(in_cur),1))*noise
        input = np.concatenate((in_cur,dist),axis=1)
        output = self.mlp1.predict(input)
        output = np.round(output*799).astype('int')
        output = self.renderCurves(output)
        return output
    
    def predict2(self,strokes,noise):
        in_cur = getStrokeBezier(self.cp,strokes)
        dist = np.ones((len(in_cur),1))*noise
        input = np.concatenate((in_cur,dist),axis=1)
        pred = self.mlp2.predict(input)
        output = self.transPath(pred,strokes)
        visual = self.transPath(pred,self.srcCurve)
        return output,visual
    
    def predict3(self,strokes):
        input = getStrokeBezier(self.cp,strokes)
        output = self.mlp3.predict(input)
        output = self.localNoising(output,strokes)
        return output
    
    def localNoising(self,preds,strokes):
        noiseStrokes = []
        for i,pred in enumerate(preds):
            mu1,std1,mu2,std2 = pred
            mu1,std1 = mu1*20-10, std1*5
            mu2,std2 = mu2*20-10, std2*5

            rdn_x = np.random.normal(mu1,std1,len(strokes[i]))[:,np.newaxis]
            rdn_y = np.random.normal(mu2,std2,len(strokes[i]))[:,np.newaxis]
            rdn_offset = np.concatenate((rdn_x,rdn_y),axis=1)
            rdn_offset = np.round(rdn_offset)
            noiseStrokes.append(strokes[i]+rdn_offset)
        return noiseStrokes

    def transOptimize(self,src,dst,i):

        offset_list = []
        dist_list = []

        target_list = []
        idx_list = []

        for j in range(i):
            tree = spatial.cKDTree(src[j])
            mindist, minid = tree.query(src[i])
            b_idx = np.argmin(mindist)
            a_idx = minid[b_idx]

            offset_src = src[i][b_idx] - src[j][a_idx]
            target = offset_src + dst[j][a_idx]
            offset_dst = target - dst[i][b_idx]
            target = np.round(target).astype('int')
            if mindist[b_idx] < 10:
                target_list.append(target)
                idx_list.append(b_idx)
            
            offset_list.append(offset_dst)
            dist_list.append(mindist[b_idx])

        dist_list = np.array(dist_list)
        weights = softmax(1/(dist_list+1))
        trans = np.average(offset_list,axis=0,weights=weights)
        trans = np.round(trans).astype('int')
        initTrans_path = dst[i] + trans
        needOptim = False

        for num,idx in enumerate(idx_list):
            i_dist = np.sum((initTrans_path[idx] - target_list[num])**2)**0.5

            if i_dist > 5:
                needOptim = True
                break
        if needOptim:
            final_path = optimize(initTrans_path,target_list,idx_list)
        else:
            final_path = initTrans_path
        return final_path

    def transPath(self,preds,strokes):
        transStrokes = []

        for i,stroke in enumerate(strokes):
            
            R,S,t0,t1 = preds[i]
            T = np.array([t0,t1])

            S = S * 2+0.1
            R = (R * 90 -45) * (np.pi/180)
            T = T * 1600 - 800
            alpha,beta = math.cos(R)*S, math.sin(R)*S
            
            M = [[alpha,beta,T[0]],\
                [-beta,alpha,T[1]]]
            M = np.array(M,dtype='float')
            trans_path = cv2.transform(stroke[:,np.newaxis,:], M).squeeze()
            transStrokes.append(trans_path)
        return transStrokes

    def renderCurves(self,bezierPoints):
        curves = []
        for i,curve in enumerate(bezierPoints):
            path = bezier_curve(curve,len(self.srcStrokes[i])).astype("int")
            curves.append(path)
        return curves

    def layout_refine(self,strokes,transMethod):

        # strokes = np.array(strokes)
        strokes = copy.deepcopy(strokes)
        
        x_min,y_min,x_max,y_max = 9999,9999,-9999,-9999
        for i in range(len(strokes)):
            if i > 0:
                strokes[i] = transMethod(self.srcCurve,strokes,i)
                # strokes[i] = self.getTrans(self.srcCurve,strokes,i)
                # strokes[i] = self.transOptimize(self.srcCurve,strokes,i)
                sx_min,sy_min = np.min(strokes[i],axis=0)
                sx_max,sy_max = np.max(strokes[i],axis=0)
                x_min,y_min = min(sx_min,x_min),min(sy_min,y_min)
                x_max,y_max = max(sx_max,x_max),max(sy_max,y_max),
        
                
        m_x, m_y = int(400-(x_min+x_max)/2), int(400-(y_min+y_max)/2)

        for i in range(len(strokes)):
            strokes[i] += [m_x,m_y]
        return strokes

    def getTrans(self,src,dst,i):
        offset_list = []
        dist_list = []

        for j in range(i):
            tree = spatial.cKDTree(src[j])
            mindist, minid = tree.query(src[i])
            b_idx = np.argmin(mindist)
            a_idx = minid[b_idx]

            offset_src = src[i][b_idx] - src[j][a_idx]
            target = offset_src + dst[j][a_idx]
            offset_dst = target - dst[i][b_idx]
            
            offset_list.append(offset_dst)
            dist_list.append(mindist[b_idx])
        
        weights = softmax(1/(np.array(dist_list)+1))

        trans = np.average(offset_list,axis=0,weights=weights)
        trans = np.round(trans).astype('int')
        trans_stroke = dst[i] + trans
        return trans_stroke

    def pipeline(self,in_noise,ex_noise,getVisual=False):

        print("Sketch Synthesis =======")
        print("Step 1: extrinsic disturbing ...")
        step1, step1_visual = model.predict2(self.srcStrokes,ex_noise)
        
        print("Step 2: intrinsic disturbing ...")
        step2 = model.predict1(step1,in_noise) # intrinsic disturbing

        print("Step 3: point disturbing ...")
        step3 = model.predict3(step2) # point disturbing

        print("Step 4: layout initialization and optimization ...")
        step42 = model.layout_refine(step3,self.transOptimize) # layout init + layout optimization

        pipeline_visual = None

        if getVisual:
            step41 = model.layout_refine(step3,self.getTrans) # layout init
            src_img = renderSketch(self.srcCurve)
            img1 = renderSketch(step1_visual)
            img2 = renderSketch(step2)
            img3 = renderSketch(step3)
            img41 = renderSketch(step41)
            img42 = renderSketch(step42)
            pipeline_visual = np.concatenate((src_img,img1,img2,img3,img41,img42),axis=1)
            
            return pipeline_visual # step-by-step result


        return step42 # final result


if __name__ == "__main__":
    N_or_P = 'N' # for the model trained on novices data or professionals data
    
    model = Synthesis(NP=N_or_P)
    
    ex_noise = 0.3 # extrinsic noise level
    in_noise = 0.3 # intrinsic noise level
    
    src_dir = "./sketch_json/"
    
    json_path = "P021_8_1_SAH_210.json" # horse expample
    
    # json_path = "P004_0_0_GCN_Armor_cat.json" # cat example
    
    save_dir = os.path.join('./results',N_or_P)
    
    srcStrokes = readStrokes(os.path.join(src_dir,json_path))
    srcStrokes = getValidStrokes(srcStrokes)
    model.load(srcStrokes)
    
    pipeline_visual = model.pipeline(in_noise=in_noise,ex_noise=ex_noise,getVisual=True)

    plt.imshow(pipeline_visual)
    plt.show()
    
    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(os.path.join(save_dir,"%s_in=%.2f_ex=%.2f.png"%(json_path.split('.')[0],in_noise,ex_noise)),cv2.cvtColor(pipeline_visual,cv2.COLOR_RGB2BGR))
    

