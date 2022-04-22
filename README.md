
# SONAR - Rock and Mine Prediction

SONAR - Sound Navigation and Ranging is a technique that uses sound
propagation (usually underwater, as in submarine navigation) to 
navigate, measure distances (ranging), communicate with or detect
objects on or under the surface of the water, such as other vessels.
Furthermore, SONAR helps to explore and map the ocean because sound
waves travel farther in the water than do radar and light waves. 
NOAA (National Oceanic and Atmospheric Administration) scientists
promarily use SONAR to develop nautical charts, locate underwater 
hazards to navigation, search for and map objects on the seafloor,
for instance, shipwrecks, and map the seafloor itself.

In order to know where a mineral (mine) is being underwater exactly
is immensely difficult without the expansion of the Sound Navigation 
and Ranging (SONAR) methodology, which uses specific parameters to 
determine if a barrier or a surface is a mine or rock. Specifically,
the NOAA scientist needs to predict whether the object on the seabed
is the mine or rock by using a sonar signal that sends sound and 
receives a switchback of the signal to detect whether the object is
the mine or rock in the ocean.

This work is my small project related to Machine learning. This 
project is the detection of rock and mine using SONAR data. All
programming is in Python using Jupyter Notebook. Since the project is a 
classification type problem, we have used logistic regression 
model which is the supervised learning algorithm.


## The workflow 

First of all, we need to collect the SONAR data. In the laboratory, 
an experiment may be done in which the SONAR is used to send and 
receive the signal bounced back from some metal cylinders and 
rocks because the mine will be made of metals. The scientists 
collect the SONAR data obtained from the metal cylinder and rock.
When we have the dataset, we shall process the data. Because the data
may not be used directly, we must preprocess the data. After that, 
the data will be split into the training and test data. Then, we 
will feed the dataset of SONAR data to our Machine Learning model.
As mentioned above, the logistic regression model is used because 
this model works very well for binary classification problem, and this
problem is a binary classification problem due to a fact that we are
going to predict whether the object on the seabed is the mine or rock.
The next step is that logistic regression model will be trained by 
learning from the dataset.

Thus, the workflow is shown by the following process:

- SONAR Data --> Data preprocessing --> Train test split --> Logistic Regression model
- New Data --> Trained Logistic Regression model --> Rock (R) or Mine (M)


