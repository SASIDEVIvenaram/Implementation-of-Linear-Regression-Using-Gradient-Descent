# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SASIDEVI V
RegisterNumber: 212222230136
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(X,y,theta))
  return theta,j_history  
  
theta,j_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000)s")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions= np.dot(theta.transpose(),x)
    return predictions[0]
    
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```
## Output:
### Profit Prediction
![ml31](https://user-images.githubusercontent.com/118707332/229972367-6f7d25ac-64c3-4374-9969-9d3d3355f291.png)
### Compute Cost
![ml32](https://user-images.githubusercontent.com/118707332/229972368-6b91a9ed-1384-48b8-befa-b4fa6052a4c7.png)
### h(x) Value
![ml33](https://user-images.githubusercontent.com/118707332/229972379-e4abbf23-e735-40cc-aaf1-629939cbd138.png)
### Cost function using Gradient Descent Graph
![ml34](https://user-images.githubusercontent.com/118707332/229972444-57d9a53c-1149-433b-9bdd-c6ee82075d9c.png)
### Profit Prediction
![ml35](https://user-images.githubusercontent.com/118707332/229972458-c2c1dc50-b8c8-46e9-9144-e27f9ee3eaca.png)

### Profit for the Population 35,000
![ml36](https://user-images.githubusercontent.com/118707332/229972476-2b036c6c-72a9-4d85-9b69-a3f60ea37480.png)
### Profit for the Population 70,000
![ml37](https://user-images.githubusercontent.com/118707332/229972496-7adf706c-7fdf-493f-b266-27ee2254391a.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
