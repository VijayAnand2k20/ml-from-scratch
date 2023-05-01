# **Linear Regression**

This repository is dedicated to learning machine learning from scratch with Python by creating own models, without the use of external machine learning frameworks or pre-built machine modles.. The goal of this project is to gain a deep understanding of the fundamentals of machine learning algorithms, including how they work, how they are implemented, and how they can be applied to real-world problems.

- [**Linear Regression**](#linear-regression)
  - [**Before running**](#before-running)
  - [**Algorithm**](#algorithm)
    - [**Approximation**:](#approximation)
    - [**Cost Function**](#cost-function)
    - [**Gradient Descent**](#gradient-descent)


## **Before running**
    
To avoid any errors, install the packages from the requirements.txt by running the following commmand in your terminal:
```
pip install -r requirements.txt
```

## **Algorithm**

In linear regression, we'll predict continuos values, such as the price of a house, the number of sales of a product, etc. The goal of linear regression is to find a line that best fits the data. The line is called the regression line.

![Linear Regression graph](https://imgs.search.brave.com/WjDTzPke1JXClvnyZQe1dUJ9AtyIDTZ6n8eeQZ_dlbM/rs:fit:900:600:1/g:ce/aHR0cHM6Ly9jZG4t/aW1hZ2VzLTEubWVk/aXVtLmNvbS9tYXgv/MTIwMC8xKndVUlFY/SHJOanE3TW1XSFVr/QWVhblEucG5n)

<p style="text-align: center;">
Source: <a href="https://towardsdatascience.com/linear-regression-from-scratch-977cd3a1db16">towards data science</a>
</p>

Since the result is a line, it can be represented by the following way.

### **Approximation**:

$
y = mx + c
$

where $m$ is the slope of the line and $c$ is the y-intercept.

### **Cost Function**

Now, we have to come up with the values for $m$ and $c$.

For this, we define something called a cost function. For linear regression,

Mean Squared Error(MSE) = 
$
J(m, c) = \frac 1 N  \sum_{i=1}^{n}(y_i - (mx_i + c))^2
$

where,
$$
y_i=Actual\ value for\ data_i
\\
mx_i + c = The \ predicted\ value\ with\ the\ current\ i
\\
N=Number\ of\ entries\ in\ the\ dataset
$$

The summation of the squares of the difference between all the actual values and their predicted values and dividing the result by the total number of entries gives us the $Mean\ Squared\ Error$

So, finally the cost function will return us how bad the values chosen for $m$ and $c$. And hence, we have to reduce the error as much as possible i.e,. We have to find the minimum value of this function.

### **Gradient Descent**

To find the minimum value of any function, we calculate the derivative(or gradient) of the function.

The derivative of the cost function J with respect to the parameters m and c can be expressed as:

$$\frac{\partial J}{\partial m} = -\frac{2}{N} \sum_{i=1}^{N} x_i(y_i - (mx_i + c))$$

$$\frac{\partial J}{\partial c} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - (mx_i + c))$$

where N is the number of data points, $x_i$ and $y_i$ are the feature and target values of the i-th data point, and $m$ and $c$ are the parameters of the linear regression model.


Gradient Descent is simply an iterative method that uses the derivative to get to the minimum value of the cost function. The algorithm is as follows:

1. Start with some initial values for $m$ and $c$.
2. Calculate the derivative of the cost function with respect to $m$ and $c$.
3. Update the values of $m$ and $c$ using the following equations:
4. Repeat steps 2 and 3 until the values of $m$ and $c$ converge to their minimum values.
5. The values of $m$ and $c$ at convergence are the optimal values for the linear regression model.

$$m = m - \alpha \frac{\partial J}{\partial m}$$

$$c = c - \alpha \frac{\partial J}{\partial c}$$

$$\frac{\partial J}{\partial m} = \frac{1}{N} \sum_{i=1}^{N} 2x_i(\hat{y} - y_i)$$

$$\frac{\partial J}{\partial c} = \frac{1}{N} \sum_{i=1}^{N} 2(\hat{y} - y_i)$$

where,
   -  $\alpha$ is the learning rate
   -  $\hat{y}$ is the predicted value

The learning rate is a hyperparameter that controls how much we are adjusting the parameters with respect to the loss gradient. If the learning rate is too small, we will need too many iterations to converge to the minimum. If the learning rate is too large, we may overshoot the minimum. It is usually necessary to try different learning rates for your model and see which one gives the best results.

<!-- showing gradient descent graph: https://imgs.search.brave.com/QaLCH9HjYSvWYQ8wB8or6dyGyxJ7WimqAkF5H2zxxko/rs:fit:1200:1061:1/g:ce/aHR0cHM6Ly9pbWFn/ZXMuZGVlcGFpLm9y/Zy9nbG9zc2FyeS10/ZXJtcy9kZDZjZGQ2/ZmNmZWE0YWYxYTEw/NzVhYWMwYjVhYTEx/MC9zZ2QucG5n with source: deepai.org -->

![Gradient Descent](https://imgs.search.brave.com/QaLCH9HjYSvWYQ8wB8or6dyGyxJ7WimqAkF5H2zxxko/rs:fit:1200:1061:1/g:ce/aHR0cHM6Ly9pbWFn/ZXMuZGVlcGFpLm9y/Zy9nbG9zc2FyeS10/ZXJtcy9kZDZjZGQ2/ZmNmZWE0YWYxYTEw/NzVhYWMwYjVhYTEx/MC9zZ2QucG5n)

<p style="text-align: center;">
Source: <a href="https://deepai.org/machine-learning-glossary-and-terms/gradient-descent">deepai.org</a>
</p>

That's it for the theory part. Now, for the implementation part, checkout the code in `LinearRegression/linear_regression.py`