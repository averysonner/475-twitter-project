from os import path
from math import sqrt
import random

import numpy as np

from pandas import DataFrame, crosstab

import matplotlib.pyplot as plt
import seaborn

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data import *

dump_results = path.abspath('./output/results.csv')
dump_dir = path.abspath('./output/dumped_outputs.txt')

sample_w = { #hardcoded weights - dont use anymore
    1:124,
    2:217,
    3:309,
    4:882,
    5:1671,
    6:2342,
    7:1594,
    8:932,
    9:167,
    10:12
}

##################################################################################
# Brain Bagger
# -Takes in prediction data from multiple classifiers, in the form of (n x m) array
# n = number of predictors
# m = length of prediction vector 
# -Returns new prediction vector
# Prediction vector is determined via selecting the brain_method
# brain_method(s) 'average', 'min', 'max', 'random'
# default: brain_method='average'

def brain_bagger(y_predictions, brain_method='average'):
    #Initial length calculations
    num_predictions = len(y_predictions[0])

    new_predictions = [0 for i in range(num_predictions)]
    #Average the results of the outputs for the 
    if(brain_method == 'average'):
        
        t_pred = np.array(y_predictions).T.tolist()
        
        for i in range(num_predictions):
            new_predictions[i] = round(np.average(t_pred[i]))
        
        return new_predictions
        
    elif(brain_method == 'min'):
        #Transpose predictions
        t_pred = np.array(y_predictions).T.tolist()
        for i in range(num_predictions):
            new_predictions[i] = min(t_pred[i])

        return new_predictions

    elif(brain_method == 'max'):
        #Transpose predictions
        t_pred = np.array(y_predictions).T.tolist()
        for i in range(num_predictions):
            new_predictions[i] = max(t_pred[i])

        return new_predictions
    
    elif(brain_method == 'random'):
        #Transpose predictions
        t_pred = np.array(y_predictions).T.tolist()
        for i in range(num_predictions):
            new_predictions[i] = random.choice(t_pred[i])
    
        return new_predictions

def heat_map(dframe, iteration):
    seaborn.heatmap(dframe.to_numpy(), cmap = "inferno", xticklabels = [], yticklabels = [])
    plt.savefig(f'./img/heatmap{iteration+1}.png', dpi = 300)

if __name__ == '__main__':
    X, y = get_data()

    for iteration in range(6):
        print(f'#{iteration+1}')
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)

        NeuralNets = []
        NeuralNets.append(MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100), activation="tanh", learning_rate_init=0.0005, max_iter = 200))
        NeuralNets.append(MLPClassifier(hidden_layer_sizes=(250,250,250,250,250,250), activation="tanh", learning_rate_init=0.0005, max_iter = 220))
        NeuralNets.append(MLPClassifier(hidden_layer_sizes=(163,112,229,102,130,115), activation="tanh", learning_rate_init=0.001, max_iter = 160))
        NeuralNets.append(MLPClassifier(hidden_layer_sizes=(242,134,122,107,200,158), activation="tanh", learning_rate_init=0.001, max_iter = 180))
        NeuralNets.append(MLPClassifier(hidden_layer_sizes=(184,236,136,225,247,170), activation="tanh", learning_rate_init=0.001, max_iter = 160))

        prediction_set = []
        for net in NeuralNets:
            '''
            # ===== Sample =====
            weighted_X_train = np.array([X_train[0]])
            weighted_Y_train = np.array([])

            for i in range(len(X_train)):
                num = random.randint(1,2000)
                weight = sample_w[Y_train[i]]
                
                if num > (weight/2):
                    for ii in range(math.ceil(num/weight)):
                        weighted_X_train = np.append(weighted_X_train, np.array([X_train[i]]), axis = 0)
                        weighted_Y_train = np.append(weighted_Y_train, np.array([Y_train[i]]))
                        if ii >= 9:
                            ii=100
            
            weighted_X_train = np.delete(weighted_X_train, (0), axis=0) #remove the extra feature set
            indexer = np.arange(weighted_X_train.shape[0])
            np.random.shuffle(indexer)

            weighted_X_train = weighted_X_train[indexer]
            weighted_Y_train = weighted_Y_train[indexer]
            '''

            # ===== Fit =====
            net.fit(X_train, Y_train)

            pred = net.predict(X_test)
            # ===== Bag =====
            prediction_set.append(pred)

            print(f'Network error: {sqrt(mean_squared_error(Y_test, pred))}, {mean_absolute_error(Y_test, pred)}')
        # ===== Predict =====

        predictors = {'average'} #only average this one
        for p in predictors:
            y_pred = brain_bagger(prediction_set, p)

            MSE  = mean_squared_error(Y_test, y_pred)
            RMSE = sqrt(MSE)
            MAE  = mean_absolute_error(Y_test, y_pred)
            
            df = DataFrame({'output':y_pred, 'answer':Y_test})
            
            if dump_results:
                df.to_csv(dump_results, header=False, index=False)
            
            if dump_dir:
                cross = crosstab(df['answer'], df['output'], rownames=['True Popularity'], colnames=['Predicted'], normalize = 'index')
                
                dump = open(f'{dump_dir}.{iteration+1}',"w")

                #dump.write(cross.to_string())
                dump.write("\n\n")
                dump.write("MSE: %.4f\n" % MSE)
                dump.write("RMSE: %.4f\n" % RMSE)
                dump.write("MAE: %.4f\n" % MAE)


                dump.close()
                heat_map(cross, iteration)


    """
    print('StdDev: ', statistics.stdev(dif))
    print('StdError: ', statistics.stdev(dif)/sqrt(len(y_pred)))
    """
