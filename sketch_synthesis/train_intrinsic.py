#%%
import numpy as np
import os,json,cv2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import comb
import matplotlib.pyplot as plt
import pickle

class MLP:
    def train(self,save_model_path,NP='N'):
        
        x_MinMax = preprocessing.MinMaxScaler()

        control_num = 6

        ## ---------------- load data
        x = np.loadtxt('./data/intrinsic/in_curves_%d_%s.txt'%(control_num,NP))
        y = np.loadtxt('./data/intrinsic/out_curves_%d_%s.txt'%(control_num,NP))

        ## ---------------- preprocessing
        x_loc = x[:,:(control_num*2)]
        x_dist = x[:,control_num*2]
        x_dist = x_dist.reshape(len(x_dist),-1)
        x_loc = x_loc/799
        x_dist_norm = x_MinMax.fit_transform(x_dist)

        x_input = np.concatenate((x_loc,x_dist_norm),1)
        y_output = y/799

        ## ---------------- Split dataset
        np.random.seed(233)
        x_train, x_test, y_train, y_test = train_test_split(x_input,y_output,test_size = 0.01)

        mlp = MLPRegressor(
            hidden_layer_sizes=(100,50),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        mlp.fit(x_train, y_train)

        pred_train = mlp.predict(x_train)
        mse_1 = mean_squared_error(pred_train,y_train)

        print ("Train ERROR = ", mse_1)

        pred_test = mlp.predict(x_test)
        mse_2 = mean_squared_error(pred_test,y_test)
        print(pred_test[0],y_test[0])
        print ("Test ERROR = ", mse_2)

        pickle.dump(mlp, open(save_model_path, 'wb'))
    
    def predict(self,input):
        results = self.model.predict(input)
        return results


if __name__ == '__main__':
    NP = 'N' # switch between novice and professional sketches using 'N' or 'P'
    save_folder = "./train_models"
    os.makedirs(save_folder,exist_ok=True)
    save_model_path = os.path.join(save_folder,f"{NP}_mlp_intrinsic.sav")
    mlp = MLP()
    mlp.train(save_model_path=save_model_path,NP=NP)