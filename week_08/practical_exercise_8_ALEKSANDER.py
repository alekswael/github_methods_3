#%% Importing packages
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.pyplot as plt

#%%
# 1.1 Loading data
data = np.load("megmag_data.npy")

#%%
# 1.1.i
data.shape
# There are 682 repetitions, 102 sensors and 251 time samples.

#%% 1.1.ii
# Create times array
times = np.arange(-200, 804, 4)

#%% 1.1.iii
# Sensor covariance matrix
covar = np.zeros(shape = (102,102))

for i in range(682):
    X = data[i]
    covar = covar + X @ np.transpose(X)

covar = (1/682)*covar

plt.close("all")
plt.figure()
plt.imshow(covar, cmap = "hot")
plt.colorbar()
plt.show()

#%% 1.1.iv
datamean = data.mean(axis=0)
datamean.shape

#%% 1.1.v
plt.close("all")
plt.figure()
plt.plot(times, datamean.T)
plt.axvline()
plt.axhline()
plt.show()

#%% 1.1.vi
sensor_max = np.amax(datamean)

plt.close("all")
plt.figure()
plt.plot(times, datamean.T)
plt.axvline()
plt.axhline()
plt.axhline(np.amax(datamean))
plt.show()

np.argmax(datamean) # finding the index (think: coordinates) for the biggest value
print(np.argmax(datamean))

a = np.unravel_index(np.argmax(datamean), shape = (102, 251)) # shape = the space where it searches for the index
print(a) # The sensor with max magnetic value is no. 73

#%% 1.1.vii
plt.figure() 
plt.plot(times, data[:,73,:].T)
plt.axhline(np.amax(data[:,73,:].T))
plt.axvline(x = 0)
plt.axhline(y = 0)
plt.xlabel("Time in ms")
plt.ylabel("Signal for one sensor")
plt.title("Signal for the max-sensor for all repetitions")
plt.show()

#%% 1.1.viii

#EXLPAIN

#%% 1.2
y = np.load("pas_vector.npy")

#%% 1.2.i

# It has the same length as repetitions, because there is a PAS rating per repitition.

#%% 1.2.ii
# subsetting the data, so I'm only looking at sensor 73 
data_1_2 = data[:, 73, :]

# in y I find the indices for each different pas rating and assign them to new lists
pas_1 = np.where(y == 1)
pas_2 = np.where(y == 2)
pas_3 = np.where(y == 3)
pas_4 = np.where(y == 4)

# finding the average brain activation in sensor 73 seperated by pas-rating
sens_73_pas1_avg = np.mean(data_73[pas_1], axis = 0)
sens_73_pas2_avg = np.mean(data_73[pas_2], axis = 0)
sens_73_pas3_avg = np.mean(data_73[pas_3], axis = 0)
sens_73_pas4_avg = np.mean(data_73[pas_4], axis = 0)

# plotting this baby
plt.figure
plt.plot(times, sens_73_pas1_avg)
plt.plot(times, sens_73_pas2_avg)
plt.plot(times, sens_73_pas3_avg)
plt.plot(times, sens_73_pas4_avg)
plt.axvline()
plt.axhline()
plt.xlabel(" time")
plt.ylabel("magnetic field")
plt.title("Average magnetic field for each PAS-rating (sensor 73)")
plt.legend(["pas 1", "pas 2", "pas 3", "pas 4"])
plt.show

#%% 1.2.iii

# EXPLAIN

##################### EXERCISE 2 ######################

#%% 2.1

# 2.1.1: We’ll start with a binary problem - create a new array called data_1_2 that only contains PAS responses 1 and 2. Similarly, create a y_1_2 for the target vector

data_1_2 = np.concatenate((data[pas_1], data[pas_2]), axis=0)
data_1_2.shape

##########################
y_1_2 = []
for i in range(len(y)):
    if y[i] == 1:
        y_1_2.append(1)
    if y[i] == 2:
        y_1_2.append(2)


np.column_stack

#%%

# 2.1.2: Scikit-learn expects our observations (data_1_2) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our data_1_2 is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use np.reshape to create a variable X_1_2 that fulfils these criteria.
## Answer to Q: reshape(3,1) first number; the number of dimensions that I want.
## We cant really interpret the flattened data frame, but we need to flatten it in order for sklearn to be able to work with it.

# repetition as rows, and sensor and time as columns
X_1_2 = data_1_2.reshape(214, -1)
X_1_2.shape

#%%
# 2.1.3: Import the StandardScaler and scale X_1_2
from sklearn.preprocessing import StandardScaler # package to standardize values in df

sc = StandardScaler()
X_1_2_scaled = sc.fit_transform(X_1_2)

# 2.1.4: Do a standard LogisticRegression - can be imported from sklearn.linear_model - make sure there is no penalty applied
from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(penalty='none') # no regularisation

logR.fit(X_1_2_scaled, y_1_2)

#%%

# 2.1.5: Use the score method of LogisticRegression to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are over fitting?

print(logR.score(X_1_2_scaled, y_1_2))

"""
The fact that we are not penalizing the model
"""
#%%
# 2.1.6: Now apply the L1 penalty instead - how many of the coefficients (.coef_) are non-zero after this?
logR = LogisticRegression(penalty="l1", solver='liblinear', random_state=1) # With regularization
logR.fit(X_1_2_scaled, y_1_2)
print(logR.score(X_1_2_scaled, y_1_2))

fit1 = logR.fit(X_1_2_scaled, y_1_2)

print(np.sum(fit1.coef_ == 0))
print(np.sum(fit1.coef_ != 0)) # = 217 coefs were nonzero

#%%
# 2.1.7: Create a new reduced X that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; X_reducedXT or XT Xreduced (you choose the right one)) . Plot the covariance of the features using plt.imshow. Compared to the plot from 1.1.iii, do we see less covariance?
coefs = logR.coef_.flatten()
non_zero = coefs != 0
X_reduced = X_1_2_scaled[:, non_zero]

# Non-zero coefficients covariance matrix
coef_covar = np.zeros(shape = (217,217))

for i in range(217):
    X = X_reduced[i]
    covar = covar + X @ np.transpose(X)

coef_covar = (1/682)*coef_covar

plt.close("all")
plt.figure()
plt.imshow(covar, cmap = "hot")
plt.colorbar()
plt.show()

#%%
# 2.2.i
from sklearn.model_selection import cross_val_score, StratifiedKFold

#%% 2.2.ii
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
    # randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
    
    # create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y

#%% 2.2.ii CONTINUED
y_1_2 = np.array(y_1_2) # Has to be array instead of list, thats why
data_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2) # Assigning new data
X_1_2_equal = data_1_2_equal.reshape(198, -1)
X_1_2_equal = sc.fit_transform(X_1_2_equal)


#%% 2.2.iii
cv = StratifiedKFold()

logR = LogisticRegression()
logR.fit(X_1_2_equal, y_1_2_equal)

scores = cross_val_score(logR, X_1_2_equal, y_1_2_equal, cv=5)
print(np.mean(scores))

# 0.470


#%% 2.2.iv
cv = StratifiedKFold()

logR = LogisticRegression(C=1e5, penalty="l2")
logR.fit(X_1_2_equal, y_1_2_equal)

scores = cross_val_score(logR, X_1_2_equal, y_1_2_equal, cv=5)
print(np.mean(scores))

# Score is 0.475

cv = StratifiedKFold()

logR = LogisticRegression(C=1e1, penalty="l2")
logR.fit(X_1_2_equal, y_1_2_equal)

scores = cross_val_score(logR, X_1_2_equal, y_1_2_equal, cv=5)
print(np.mean(scores))

# Score is 0.475

cv = StratifiedKFold()

logR = LogisticRegression(C=1e-5, penalty="l2")
logR.fit(X_1_2_equal, y_1_2_equal)

logR.predict

scores = cross_val_score(logR, X_1_2_equal, y_1_2_equal, cv=5)
print(np.mean(scores))

# Score is 0.456

#%% 2.2.v
# I need 251 models.

cv = StratifiedKFold()
logR = LogisticRegression(C=1e1, penalty="l2", solver = "liblinear")
cv_scores = []

# Subsetting time
for i in range(251):
    t = sc.fit_transform(data_1_2_equal[:,:,i])
    logR.fit(t, y_1_2_equal)
    scores = cross_val_score(logR, t, y_1_2_equal, cv=5)
    cv_scores.append(np.mean(scores))

#%% 2.2.v CONTINUED
# Picking highest score
np.amax(cv_scores)
np.argmax(cv_scores) # Indeci with highest classification

#%% 2.2.v CONTINUED
plt.figure() 
plt.plot(times, cv_scores)
plt.axvline(x = 0)
plt.axvline(times[94])
plt.axhline(y = 0.5) # Chance level is 50% for binary classification
plt.xlabel("Time in ms")
plt.ylabel("Classification score")
plt.title("Classification scores at given times")
plt.show()

times[94]

# Classification is best at 176 ms.

#%% 2.2.vi

cv = StratifiedKFold()
logR = LogisticRegression(C=1e-1, penalty="l1", solver = "liblinear")
cv_scores = []

# Subsetting time
for i in range(251):
    t = sc.fit_transform(data_1_2_equal[:,:,i])
    logR.fit(t, y_1_2_equal)
    scores = cross_val_score(logR, t, y_1_2_equal, cv=5)
    cv_scores.append(np.mean(scores))

#%% 2.2.vi CONTINUED
# Picking highest score
np.amax(cv_scores)
np.argmax(cv_scores) # Indeci with highest classification

#%% 2.2.vi CONTINUED
plt.figure() 
plt.plot(times, cv_scores)
plt.axvline(x = 0)
plt.axvline(times[36])
plt.axhline(y = 0.5) # Chance level is 50% for binary classification
plt.xlabel("Time in ms")
plt.ylabel("Classification score")
plt.title("Classification scores at given times")
plt.show()

times[36]

# Is best at -56 ms.

#%% 2.2.vii


#%%
# EXERCISE 3 - Do a Support Vector Machine Classification on all four PAS-ratings  
def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)
    
    new_indices = np.concatenate((first_choice, second_choice,
                                 third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y

# %%
