#%%
import numpy as np
import os,json,cv2,math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import comb
import matplotlib.pyplot as plt
import pickle



class MLP:
    def train(self,save_model_path='./model/mlp_curveNoise.sav'):
  
        ## ---------------- load data
        x = np.loadtxt('./data/curve_noise/in_all.txt')
        y = np.loadtxt('./data/curve_noise/out_all.txt')

        ## ---------------- Split dataset
        np.random.seed(233)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
        # print(x_train[:2],y_train[:2])

        mlp = MLPRegressor(
            hidden_layer_sizes=(100,50),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        mlp.fit(x_train, y_train)

        pred_train = mlp.predict(x_train)
        mse_1 = mean_squared_error(pred_train,y_train)

        print ("Train ERROR = ", mse_1)

        pred_test = mlp.predict(x_test)
        mse_2 = mean_squared_error(pred_test,y_test)
        # print(pred_test[:5],y_test[:5])
        print ("Test ERROR = ", mse_2)

        self.model = mlp

        pickle.dump(mlp, open(save_model_path, 'wb'))
    

    def predict(self,input):
        results = self.model.predict(input)
        return results



if __name__ == '__main__':

    save_folder = "./train_models"
    os.makedirs(save_folder,exist_ok=True)
    save_model_path = os.path.join(save_folder,"mlp_curveNoise.sav")

    mlp = MLP()
    mlp.train(save_model_path=save_model_path)

