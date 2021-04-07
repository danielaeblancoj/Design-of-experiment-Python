#!/usr/bin/python
# app.py
"""
IMPORTS
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
import pandas

import plotly.express as px
from collections import Counter
from sklearn.inspection import permutation_importance





#layers = 10
#max_neurons = 10
#n_batch = 2

layers = 10
max_neurons = 15
n_batch = 1
n_queries = 14
n_initial = 4 # Number of initial points.
random_init=np.array([1]) #If you put 1 it will do initial selection randomly. If you leave it empty then it will take DOE recommended experiments to start


lim_point_closeness=[0.15,0.15,0.15,0.15]
neurons_input_ann=0 #0 if you want it to run with optimal number, put the number if you want to fix it. Layers (5,2)
VarExp=0.95 #how much do you want the principal components to explain
doe_experiments=pandas.read_csv("Plackett_Burman_design.csv",header=None)
doe_combined=pandas.read_csv("Combined_PlacketBurnam_Sukharev_design.csv",header=None)
doe_experiments=doe_experiments.values
doe_combined=doe_combined.values
doe_experiments= np.delete(doe_experiments,0, axis=1) #Delete the first column that is just empty
i1_name=doe_experiments[0,0]
i2_name=doe_experiments[0,1]
i3_name=doe_experiments[0,2]
i4_name=doe_experiments[0,3]
doe_experiments=np.delete(doe_experiments,0,axis=0) #delete first row bc that's just name of variables

pred_todo=np.array([[]])
alphav=1#If you increase this, you increase overfitting, can reach higher prediction values if increasing. Default 1e-5
#number of neurons is not enough
num_inputs=4

"""
FUNCTIONS
"""

def eval_func(Xpool):
    x=Xpool[:,0]
    y=Xpool[:,1]
    z=Xpool[:,2]
    w=Xpool[:,3]
    a=1
    b=5.1/(4*np.pi**2)
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    #Ypoolreal = a * (y - b * x**2 + c*x-z**2/w - r)**2 + s * (1 - t+w) * np.cos(x) + s*z - np.sin(z)
    #Ypoolreal=a*(x-0.001*y**2+c/z-w**2)+np.cos(x)*y-np.cos(w)*y
    Ypoolreal = a * (x - 0.001 * y- w ** 2) + np.sin(x)*w - np.cos(w) * x

    return Ypoolreal


# FUNCTION PLOT TRAINING DATA
def plot_training_data3D(inputs, eff_train,i1_name,i2_name,i3_name):

    trace = go.Scatter3d(
        x=inputs[:,0], y=inputs[:,1], z=inputs[:,2],
        name="Training Data", mode="markers",
        marker=dict(size=12, color=eff_train, colorscale="Viridis", opacity=0.8, colorbar=dict(thickness=20))
    )

    layout = go.Layout(
        title=dict(text="Training Data", x=0.5, xanchor="center"),
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title=i1_name),
            yaxis=go.layout.scene.YAxis(title=i2_name),
            zaxis=go.layout.scene.ZAxis(title=i3_name)
        )
    )
    new_figure = go.Figure(data=[trace], layout=layout)
    new_figure.show()



def optimal_ann(max_neurons, x_training, y_training):
    performance_neurons = []
    for neurons in range(1, max_neurons + 1):  # I add the one so that it actually goes to the max neuron I set
        ann = MLPRegressor(solver="lbfgs", alpha=alphav, hidden_layer_sizes=neurons, random_state=1, max_iter=3000,
                           learning_rate_init=0.1)
        ann.fit(x_training, y_training)  # Refit model to new data
        y_pred = ann.predict(x_training)
        error = 1 / len(y_training) * sum((y_training - y_pred) ** 2)
        performance_neurons.append(error)  # Initial performance of prediction
    #print("Optimal neurons")
    #print(np.argmin(performance_neurons)+1)
    return np.argmin(performance_neurons) + 1


def quering(ann, x_pool, x_pool_av, x_training, y_training, n_batch):
    ann.fit(x_training, y_training)  # Refit model to new data

    prediction = ann.predict(
        x_pool_av)  # Use GP or given regressor and obtain predictions of pool + STANDARD DEVIATION (std)
    prediction = prediction / max(prediction)  # normalize the predictions

    # The error will be designed as the distance to the closest known point
    distances = np.zeros(len(x_training))
    min_distances = np.zeros(len(x_pool_av))  # Number of rows of x_training
    inputs = len(x_pool[0])  # Number of columns is the number of inputs that are being used for prediction

    pos = 0
    for k in x_pool_av:
        pos2 = 0
        for t in x_training:
            factors = []
            for inp in range(0, inputs):
                dist_calc = (k[inp] - t[inp]) ** 2  # Distance from each point of my available point to each
                # of the training points, which are our actual KNOWN values
                factors.append(dist_calc)
            distance = np.sqrt(sum(factors))
            distances[pos2] = distance
            closest_dist = min(distances)
            min_distances[pos] = closest_dist

            pos2 = pos2 + 1

        pos = pos + 1

    std = min_distances
    std = std / max(std)  # normalize the errors
    scores = 0.8 * prediction + 0.2 * std

    raw_query_idx = scores.argsort()[-n_batch:][::-1] # get index of n_batch highest scores
    longer_order=4*n_batch
    query_idx = scores.argsort()[-longer_order:][::-1]
    many_query_values=x_pool_av[query_idx]
    raw_query_values=x_pool_av[raw_query_idx]

    #Make sure that they are not all aglomerated in the same place

    max_x_pool=[]
    for index in range(0, inputs):
        max_x_pool.append(max(x_pool[:,index]))

    absValues = np.abs(max_x_pool)
    lim_dist = []
    for index in range(0, len(x_pool[0])):
        lim_dist.append(lim_point_closeness[index] * absValues[index]) #You want the points you query to be separated
    #by at least 5% of the total length of one side of the cube
    query_values_useful = many_query_values
    k = 0
    for point in np.arange(n_batch - 1):
        points_too_close = []
        for point2 in np.arange(point + 1, len(query_values_useful - 1)):
            values = abs(query_values_useful[point2, :] - query_values_useful[point, :])
            diff = values - lim_dist
            if np.all(diff <= 0):
                points_too_close.append(point2)
        query_values_useful = np.delete(query_values_useful, points_too_close, axis=0)
        if len(points_too_close) > 0:
            k = k + 1  # warning signal

    good_query_values = query_values_useful[0:n_batch, :]
    if k > 0:
        print('One or more of the suggested next points were too close and the algorithm has adjusted accordingly')



    return raw_query_values, good_query_values, k  # Return the indexes and the X input values corresponding to those

def check_duplicates(y_pred):

    countings = Counter(y_pred)
    mostrep = countings.most_common(1)[0][0] #Find value that repeats most
    y_pred = y_pred.tolist() #Change to list because count only works on lists
    counts = y_pred.count(mostrep) #Number of times most repeated value appears on y_pred array
    total_values=len(y_pred)
    percentage=counts/total_values*100
    return percentage

def run_ann_and_al_many_inputs(neurons, x_training, y_training, x_pool, x_pool_av, n_batch):
    ann = MLPRegressor(solver="lbfgs", alpha=alphav, hidden_layer_sizes=neurons, random_state=1, max_iter=3000,
                       learning_rate_init=0.1)
    # Initial prediction
    ann.fit(x_training, y_training)  # Refit model to new data
    y_pred = ann.predict(x_pool)

    # Feature importance
    model = ann.fit(x_training, y_training)
    featureimp = permutation_importance(model, x_training, y_training, n_repeats=10, random_state=0)
    mean_importance = featureimp.importances_mean
    std_importance = featureimp.importances_std

    # Check if feature importances are all zero
    is_all_zero = np.all((mean_importance == 0))
    if is_all_zero:  # If they are all zero, the model is likely defaulting to ONE value throughout the space
        # So the best is to increase number of neurons by 1 and refit to get out of that deadend
        neurons = neurons[0] + 1

        ann = MLPRegressor(solver="lbfgs", alpha=alphav, hidden_layer_sizes=neurons, random_state=1, max_iter=3000,
                           learning_rate_init=0.1)
        # Initial prediction
        ann.fit(x_training, y_training)  # Refit model to new data
        y_pred = ann.predict(x_pool)

        # Feature importance
        featureimp = permutation_importance(model, x_training, y_training, n_repeats=10, random_state=0)
        mean_importance = featureimp.importances_mean
        std_importance = featureimp.importances_std

        print('Model defaulted to unvarying prediction. Number of nodes has been automatically adjusted')
        is_all_zero = np.all((mean_importance == 0))
        if is_all_zero:
            print('Analysis suggests incorrect number of nodes, please change manually')

    # Check if the prediction is still defaulting to one value
    y_pred = ann.predict(x_pool)
    percentage_rep = check_duplicates(y_pred)
    if percentage_rep > 25:
        print(
            'Over 25% has been assigned the same prediction value. Caution advised. Higher number of nodes recommended')

    #print("Feature importance on the model")
    #print(mean_importance)
    #print(std_importance)

    if pred_todo.any():  # if it's not empty, so there is something to predict
        peq = ann.predict(pred_todo)
        print('Little prediction')
        print(peq)

    # Check for warnings

    # Plot prediction with fixed efficiency scale
    y_pred = ann.predict(x_pool)

    # Calculate predictions of just training data bc that's what you know for error calc
    y_t_pred = ann.predict(x_training)

    np_sum = np.sum((y_training - y_t_pred) ** 2)
    error = np.sqrt(1 / len(y_training) * np_sum)
    performance_history = [error]  # Initial performance of prediction

    # Automatically queries the new samples. Calls the query strategy function and selects new queries
    raw_query, x_new, k = quering(ann, x_pool, x_pool_av, x_training, y_training,
                                  n_batch)  # New queried points that will be added to training set

    #print('Original suggested points')
    #print(raw_query)

    # IDENTIFICATION OF OPTIMAL POINTS

    ind_max_pred = y_pred.argsort()[-5:][::-1]
    max_pred_coord = x_pool[ind_max_pred]
    max_pred_eff = y_pred[ind_max_pred]

    '''
    #IF YOU WANT TO ADD MAX PREDICTION TO SUGGESTED POINTS
    addedmax=max_pred_coord[0,:]
    if addedmax.all() == x_new.all():
        print('no need to add predicted max to suggested points')

    else:
        print(addedmax)
        print(x_new)
        exit()
        x_new=np.append(addedmax,x_new,axis=0)
    '''

    ''' 
    print("Suggested Next Coordinates")
    print(x_new)
    print("Prediction Error")
    print(error)
    print("Max predicted coordinates")
    print(max_pred_coord)
    print("Max predicted output")
    print(max_pred_eff)
    '''

    ''' 
    # PLOTLY FOR HTML - feature importance from PERMUTATION on model!!

    y = np.arange(1, len(mean_importance) + 1)

    figure6 = px.bar(x=y, y=mean_importance,
                     labels={'x': 'Input feature', 'y': 'Mean Importance'}, width=1000)
    figure6.update_layout(xaxis={'tickformat': ',d', 'title_font_size': 20, 'tickfont_size': 15},
                          yaxis={'title_font_size': 20, 'tickfont_size': 15})
    figure6.show()
    '''

    return x_new, max_pred_coord, max_pred_eff, error, neurons


def plot2d2trace(xvalues1,trace1,xvalues2,trace2,nametrace1,nametrace2,xlabel,ylabel,graphtitle):
    figTRAIN = go.Figure()
    figTRAIN.add_trace(go.Scatter(x=xvalues1, y=trace1, mode='lines+markers', line=dict(color='navy'),
                                  marker=dict(size=16, color='navy'), name=nametrace1,
                                  customdata=trace1,
                                  hovertemplate='x:%{x}<br>y:%{y}<br>z:%{z}<br>target: %{customdata} <extra></extra> '))

    figTRAIN.add_trace(go.Scatter(x=xvalues2, y=trace2, mode='lines+markers',
                                  line=dict(color='lightcoral'),
                                  marker=dict(size=16, color='lightcoral'), name=nametrace2,
                                  customdata=trace2,
                                  hovertemplate='x:%{x}<br>y:%{y}<br>z:%{z}<br>target: %{customdata} <extra></extra> '))

    figTRAIN.update_layout(xaxis={'title': xlabel, 'title_font_size': 15, 'tickfont_size': 15},
                           yaxis={'title': ylabel, 'title_font_size': 15, 'tickfont_size': 15},
                           title=graphtitle,
                           showlegend=True)
    figTRAIN.update_layout(
        xaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'))
    #figTRAIN.update_layout(width=600, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    figTRAIN.update_layout(width=600, height=400)

    figTRAIN.show()

def plot2d1trace(xvalues1,trace1,xlabel,ylabel,graphtitle):
    figTRAIN = go.Figure()
    figTRAIN.add_trace(go.Scatter(x=xvalues1, y=trace1, mode='lines+markers', line=dict(color='navy'),
                                  marker=dict(size=16, color='navy'),
                                  customdata=trace1,
                                  hovertemplate='x:%{x}<br>y:%{y}<br>z:%{z}<br>target: %{customdata} <extra></extra> '))

    figTRAIN.update_layout(xaxis={'title': xlabel, 'title_font_size': 15, 'tickfont_size': 15},
                           yaxis={'title': ylabel, 'title_font_size': 15, 'tickfont_size': 15},
                           title=graphtitle,
                           showlegend=False)
    figTRAIN.update_layout(
        xaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(mirror=True, ticks='inside', tickwidth=1.5, showline=True, linewidth=1, linecolor='black'))
    figTRAIN.update_layout(width=600, height=400)

    figTRAIN.show()





RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

#DOE RESULTS
doe_experiments = doe_experiments.astype(np.float)
resultsDOE=eval_func(doe_experiments)


#REAL RESULTS FOR OVERALL DATA
input1 = np.linspace(40,70, 10)
input2 = np.linspace(290,350,10)
input3 = np.linspace(0.2,0.4,10)
input4 = np.linspace(5,11, 10)
inputs=np.array([[input1],[input2],[input3],[input4]])

combinations = list(itertools.product(input1,input2,input3,input4)) # vector with all possible combinations of the input variables
inputs_full = np.array(combinations)
resultsREAL=eval_func(inputs_full)

#Find  overall real maximums from real data
print('overall data max '+ str(max(resultsREAL)))
max_data_idx=np.argmax(resultsREAL)
max_data=inputs_full[max_data_idx]
print('overall data max at ' + str(inputs_full[max_data_idx]))


#Find  overall real maximums from DOE
print('DOE max '+ str(max(resultsDOE)))
max_data_idx=np.argmax(resultsDOE)
print('DOE max at ' + str(doe_experiments[max_data_idx]))
print('DOE used ' + str(len(resultsDOE)) + ' experiments')



#PLOT MAXIMA FROM DOE
#Obtain results from combined DOE aapproach, so we can have some of corners and also centers
resultscombDOE=eval_func(doe_combined)
xvalues=np.arange(1, len(resultscombDOE))
plot2d1trace(xvalues,resultscombDOE,'Number of points','Value obtained','DOE')

print('DOE combined max '+ str(max(resultscombDOE)))
max_data_idx=np.argmax(resultscombDOE)
print('DOE combined max at ' + str(doe_combined[max_data_idx]))
print('DOE combined used ' + str(len(resultscombDOE)) + ' experiments')


## NOW WE TEST OUR ALGORITHM
# PICKING INITIAL POINTS

if random_init.any(): # if it's not empty, then you select the initial points randomly
    initial_idx = np.random.choice(range(len(inputs_full)), size=n_initial, replace=False)

    x_training, y_training = inputs_full[initial_idx], resultsREAL[initial_idx]  # INITIAL TRAINING VALUES!!
    x_pool_av = np.delete(inputs_full, (initial_idx),
                      axis=0)  # Delete values that were already tested from available pool of points
    y_pool_av = np.delete(resultsREAL, initial_idx, axis=0)

else:
    initial_idx = np.random.choice(range(len(doe_experiments)), size=n_initial, replace=False)
    x_training=doe_experiments[initial_idx]
    y_training=eval_func(x_training)
    x_pool_av=inputs_full




errorvec = np.zeros(n_queries)
neuronsvec = np.zeros(n_queries)
distvec = np.zeros(n_queries)
maxvec=np.zeros(n_queries)
npointsvec=np.zeros(n_queries)
for idx in range(n_queries):
    var1 = 'x_new_' + str(idx + 1)
    var2 = 'max_coord_' + str(idx + 1)
    var3 = 'max_eff_' + str(idx + 1)
    neurons_opt = optimal_ann(max_neurons, x_training, y_training)
    if num_inputs == 2:
        [var1, var2, var3, error, neurons_used] = run_ann_and_al2D(neurons_opt, x_training, y_training, inputs_full,
                                                                   x_pool_av, n_batch)
    else:
        [var1, var2, var3, error, neurons_used] = run_ann_and_al_many_inputs(neurons_opt, x_training, y_training, inputs_full, x_pool_av,n_batch)

    print('Sunthetics it' + str(idx + 1) + 'number of points ' + str(len(x_training)))
    npointsvec[idx]=len(x_training)
    errorvec[idx] = error
    neuronsvec[idx] = neurons_used
    x_training = np.append(x_training, var1, axis=0)  # Add new values to training set, augmenting training set
    y_new = eval_func(var1)
    y_training = np.append(y_training, y_new, axis=0)

    #Calculate actual max, not predicted
    #var2=np.array([[70,290,0.2,5],[70,290,0.2,5]])
    value=eval_func(var2)
    print('Sunthetics it'+str(idx+1) +' max ' + str(value[0]))
    print('Sunthetics it' + str(idx + 1) + ' max at ' + str(var2[0,:]))
    maxvec[idx]=value[0]


    # Calculate distance between maximums
    factors = []
    for k in range(0, num_inputs):
        dist_calc = (var2[0, k] - max_data[k]) ** 2
        factors.append(dist_calc)
        distance = np.sqrt(sum(factors))

    distvec[idx] = distance

'''
# PLOT evolution of error with iterations
xvalues=np.arange(1, len(errorvec))
plot2d1trace(xvalues,errorvec,'Iteration','MSE',None)


# PLOT number of neurons used on each iteration
xvalues=np.arange(1, len(neuronsvec))
plot2d1trace(xvalues,neuronsvec,'Iteration','Neurons',None)


# PLOT distance of max predicted point on iteration from actual maximum
xvalues=np.arange(1, len(distvec))
plot2d1trace(xvalues,distvec,'Iteration','Distance to real data maximum',None)


# PLOT max of each iteration
xvalues=np.arange(1, len(maxvec))
plot2d1trace(xvalues,maxvec,'Iteration','Max obtained',None)


# PLOT max of each iteration AGAINST NUMBER OF TRAINING POINTS
plot2d1trace(npointsvec,maxvec,'Number of training points','Max obtained',None)
'''



# PLOT DOE - Sunthetics together
xvalues2=np.arange(1, len(resultscombDOE))
plot2d2trace(npointsvec,maxvec,xvalues2,resultscombDOE,'Sunthetics ML','DOE','Number of training points','Value obtained',None) #Plots the max PREDICTED by Sunthetics
plot2d2trace(npointsvec,y_training,xvalues2,resultscombDOE,'Sunthetics ML','DOE','Number of training points','Values tested',None) #Plots actual values tested/verified by Sunthetics




print('final dataset ' + str(x_training))

print('final values tested FROM campaign' + str(y_training))










