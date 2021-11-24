#!/usr/bin/env python
# coding: utf-8

# # Practical Exercise 8 
# Sirid Wihlborg

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ## Exercise 1: Loading and inspecting the data

# ### 1.1: Loading data 

# #### 1.1 i: Explaining the shape of the data

# In[4]:


# First we load the MEG data
data = np.load("megmag_data.npy")
data.shape


# We see that the data is a 3-dimensional array. It has repetetion (can be considered "trial" ie. number of visual stimuli) on the 0-axis. It has number of sensors that record magnetic fields (in Tesla) on the 1-axis. The 2-axis is number of time samples.
# We have 682 trials, 102 sensors and 251 time samples. 

# ![data.jpeg](attachment:data.jpeg)

# #### 1.1 ii: Creating time array

# In[5]:


#Creating a 1d-array time vector:
times = np.arange(-200, 804, step = 4)


# #### 1.1 iii: Creating covariance matrix

# Covariance matrix: $cov_{x,y}=\frac{\sum_{i=1}^{N}XX^T}{N-1}$

# In[6]:


# Creating a function to make a covariance matrix
# this only works if you are looing at the variable in the 1. array
def cov_func(data):

    cov = np.zeros(shape = (len(data[0]), len(data[0]))) # making an empty matrix 

    for i in range(len(data[0])):

        X = data[i, :, :]
        X_T = np.transpose(X)

        cov += np.dot(X, X_T) # adding the X*X_T values to the pre-made empty matrix
    
    cov = cov / len(data[0])

    return cov


cov = cov_func(data)

plt.imshow(cov)


# #### 1.1 iv: Making an average of brain activity over the repetition variable

# In[7]:


# Average brain activation from -200ms to 800ms accross all repetitions.
rep_avg = np.mean(data, axis = 0)
rep_avg.shape


# #### 1.1 v: Plotting the averaged brain activity

# In[8]:


plt.figure() # intialise plot
plt.plot(times, rep_avg.T) # remember to transpose your averaged brain activity variable
plt.axvline(x = 0)
plt.axhline(y = 0)
plt.xlabel("Time in ms")
plt.ylabel("Averaged brain activity across repetitions")
plt.title("Averaged Brain Activity across Repetitions")
plt.show()


# #### 1.1 vi: Finding the largest magnetic field in the average and find the belonging sensor

# In[9]:


sensor_max = np.max(rep_avg) # finding maximum magnetic field value
print(sensor_max)


np.argmax(rep_avg) # finding the index (think: coordinates) for the biggest value
print(np.argmax(rep_avg))

a = np.unravel_index(np.argmax(rep_avg), shape = (102, 251)) # shape = the space where it searches for the index
print(a)
print(times[112]) # remember that times is defined by from -200 to 800 by 4ms (dependent on experiment design)
# answer: 248, see the graph x-axis, this correspond to the time with biggest magnetic field


# We can conclude that the sensor receiving the strongest signal is number 73.

# #### 1.1 vii: Plotting the magnetic field for each repetition for the max-sensor

# In[10]:

plt.figure() 
plt.plot(times, data[:,73,:].T)
plt.axvline(x = 0)
plt.axhline(np.amax(data[:,73,:]))
plt.axhline(y = 0)
plt.xlabel("Time in ms")
plt.ylabel("Signal for one sensor")
plt.title("Signal for the max-sensor for all repetitions")
plt.show()


# #### viii: Description of how the average is represented in the single repetitions
# Describe in your own words how the response found in the average is represented in the single
# repetitions. But do make sure to use the concepts signal and noise and comment on any differences
# on the range of values on the y-axis

# **Obssss missing**

# ### 1.2 Loading in the PAS ratings

# #### i: Describing the array

# In[11]:

y = np.load("pas_vector.npy")
y.shape # checking the dimensions of the PAS array


# y is a 1D array with the length 682, exactly the same as number of repitions. This makes sense, since for each visual stimuli they gave a PAS rating. 

# #### ii:  Now make four averages (As in Exercise 1.1.iii), one for each PAS rating, and plot the four time courses (one for each PAS rating) for the sensor found in Exercise 1.1.v 1.1.vi

# In[12]:


# subsetting the data, so I'm only looking at sensor 73 
data_73 = data[:, 73, :]

# in y I find the indices for each different pas rating and assign them to new lists
PAS1 = np.where(y == 1)
PAS2 = np.where(y == 2)
PAS3 = np.where(y == 3)
PAS4 = np.where(y == 4)

# finding the average brain activation in sensor 73 seperated by pas-rating
sens_73_pas1_avg = np.mean(data_73[PAS1], axis = 0)
sens_73_pas2_avg = np.mean(data_73[PAS2], axis = 0)
sens_73_pas3_avg = np.mean(data_73[PAS3], axis = 0)
sens_73_pas4_avg = np.mean(data_73[PAS4], axis = 0)

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


# Breaking down: What does "data_73[PAS1]" mean??
# 
# 
# Answer: So, this is a 2D array that has repetitions/trials on 0-axis, 
#     ONLY the repetitions on the same indices associted with PAS 1 (in total = 99 trials were rated as PAS 1).
#     On the 1-axis we have time. 

# In[13]:


data_73[PAS1].shape


# #### 1.2 iii: 

# ## Exercise 2: Logistic Regression to classify pairs of ratings

# ### 1) Now, we are going to do Logistic Regression with the aim of classifying the PAS-rating given by the subject

# #### 2.1 i: We’ll start with a binary problem - create a new array called data_1_2 that only contains PAS responses 1 and 2. Similarly, create a y_1_2 for the target vector

# In[14]:


PAS1 = np.where(y == 1)
PAS2 = np.where(y == 2)
PAS3 = np.where(y == 3)
PAS4 = np.where(y == 4)

data_1_2 = np.concatenate((data[PAS1], data[PAS2]), axis = 0)
y_1_2 = np.concatenate((y[PAS1], y[PAS2]), axis = 0)


# #### ii. Scikit-learn expects our observations (data_1_2) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our data_1_2 is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use np.reshape to create a variable X_1_2 that fulfils these criteria.

# In[15]:


data_1_2.shape # output: (214, 102, 251)

X_1_2 = np.reshape(data_1_2, (214, 102*251))
X_1_2.shape


# #### iii. Import the StandardScaler and scale X_1_2

# In[16]:


# from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_1_2_std = sc.fit_transform(X_1_2)


# #### iv. Do a standard LogisticRegression - can be imported from sklearn.linear_model - make sure there is no penalty applied

# In[17]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1, penalty = 'none')
lr.fit(X_1_2_std, y_1_2)


# #### v. Use the score method of LogisticRegression to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are overfitting?

# In[18]:


lr.score(X_1_2_std, y_1_2)


# *We have an accuracy of 100%. So, what a fucking great model, am I right?? Lol. No, obviously we got some overfitting going on here. We didn't split the data into test and training, so we are basically predicting the exact same data that we used for training, NOT GREAT man...*

# #### vi. Now apply the L1 penalty instead - how many of the coefficients (.coef_) are non-zero after this?

# In[42]:


lr_penalty = LogisticRegression(random_state=1, penalty = 'l1', solver = 'saga', tol = 0.01) # first it didn't converge, but then I set tol = 0.01. tol: Tolerance for stopping criteria.
lr_penalty.fit(X_1_2_std, y_1_2)
lr_penalty.score(X_1_2_std, y_1_2)

# checking coef
lr_penalty.coef_.shape # output (1, 25602)


zeros = lr_penalty.coef_[lr_penalty.coef_==0]
non_zeros = lr_penalty.coef_[lr_penalty.coef_!=0]

print(zeros.shape + non_zeros.shape)
10316+15286 # check that the amount of total values is equal to the number in your original list


# *We have 15286 coefficients that are non-zero. This means that almost half of all the coefficients got set to zero. Daaaamn. What a rough penalty Lasso...!*

# #### vii. Create a new reduced X that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; XreducedXT reduced or XT reducedXreduced (you choose the right one)) . Plot the covariance of the features using plt.imshow. Compared to the plot from 1.1.iii, do we see less covariance? 
# 
# 
# 

# In[53]:


X_reduced = non_zeros

X_reduced.shape

cov = np.cov(X_reduced)
plt.figure()
plt.imshow(cov)


# In[ ]:


def cov_func(data):

    cov = np.zeros(shape = (len(data[0]), len(data[0]))) # making an empty matrix 

    for i in range(len(data[0])):

        X = data[i, :, :]
        X_T = np.transpose(X)

        cov += np.dot(X, X_T) # adding the X*X_T values to the pre-made empty matrix
    
    cov = cov / len(data[0])

    return cov


cov = cov_func(data)

plt.imshow(cov)


# %%
