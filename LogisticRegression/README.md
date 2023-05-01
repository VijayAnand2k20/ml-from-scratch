# **Logistic Regression**

This repository is dedicated to learning machine learning from scratch with Python by creating own models, without the use of external machine learning frameworks or pre-built machine modles.. The goal of this project is to gain a deep understanding of the fundamentals of machine learning algorithms, including how they work, how they are implemented, and how they can be applied to real-world problems.

- [**Logistic Regression**](#logistic-regression)
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
In logistic regression, we predict discrete values, such as whether an email is spam or not, whether a tumor is malignant or not, etc. 

![Sigmoid function graph](https://miro.medium.com/v2/resize:fit:828/format:webp/1*dm6ZaX5fuSmuVvM4Ds-vcg.jpeg)

<p style="text-align: center;">
Source: <a href="https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148">towards data science</a>


### **Approximation**:
$$
f(w,b) = wx + b
$$
where w is the weight, b is the bias, and x is the input.

$$
h(x) = \frac{1}{1 + e^{-x}}
$$
where h is the activation function. In logistic regression, it is sigmoid function.

$$
y = h(f(w,b)) => h(wx + b)
$$

$$
y = \frac{1}{1 + e^{-(wx + b)}}
$$
y is the approximated value.


### **Cost Function**

Now, we have to come up with the values for $w$ and $b$.

For this, we define something called a cost function. For logistic regression,

$$
J(\theta) = \frac{1}{N} \sum_{i=1}^{N} Cost(h_\theta(x^{(i)}), y^{(i)})
$$

where,
$$
N=Total\ number\ of\ data,
% insert new line here
$$
$$
h_\theta(x^{(i)})=Predicted\ value\ for\ data_i,
$$
$$
y^{(i)}=Actual\ value\ for\ data_i,
$$

<br>

If $y^{(i)} = 1$,

$$
 Cost(h_\theta(x^{(i)}), y^{(i)}) = -log(h_\theta(x^{(i)}))
$$

If $y^{(i)} = 0$,

$$
 Cost(h_\theta(x^{(i)}), y^{(i)}) = -log(1 - h_\theta(x^{(i)}))
$$

<br>

Overall,  $Cost(h_\theta(x^{(i)}), y^{(i)})$ can be written as,

$$
J(w, b) = - \frac 1 N  \sum_{i=1}^{N}(y_i \log(h(wx_i + b)) + (1 - y_i) \log(1 - h(wx_i + b)))
$$

where,
$$
y_i=Actual\ value\ for\ data_i
$$
$$
h(wx_i + b)=Predicted\ value\ for\ data_i
$$
$$
N=Total\ number\ of\ data
$$

<br>

### **Gradient Descent**

To fit the parameters $w$ and $b$ to the data, we have to minimize the cost function $J(w, b)$.

To minimize the cost function, we use gradient descent algorithm.


$w$ = $w$ - $\alpha *  dw$  

$b$ = $b$ - $\alpha * db$
$$
w = w - \alpha \frac{\partial J(w, b)}{\partial w}
$$
$$
b = b - \alpha \frac{\partial J(w, b)}{\partial b}
$$

where,
$$
\alpha=Learning\ rate
$$

$$
\frac{\partial J(w, b)}{\partial w}=\frac{1}{N} \sum_{i=1}^{N} (h(wx_i + b) - y_i) x_i
$$

$$
\frac{\partial J(w, b)}{\partial b}=\frac{1}{N} \sum_{i=1}^{N} (h(wx_i + b) - y_i)
$$

for the calculations of derivative, refer to [this](https://youtu.be/0VMK18nphpg) video.

![Gradient Descent](https://imgs.search.brave.com/QaLCH9HjYSvWYQ8wB8or6dyGyxJ7WimqAkF5H2zxxko/rs:fit:1200:1061:1/g:ce/aHR0cHM6Ly9pbWFn/ZXMuZGVlcGFpLm9y/Zy9nbG9zc2FyeS10/ZXJtcy9kZDZjZGQ2/ZmNmZWE0YWYxYTEw/NzVhYWMwYjVhYTEx/MC9zZ2QucG5n)

<p style="text-align: center;">
Source: <a href="https://deepai.org/machine-learning-glossary-and-terms/gradient-descent">deepai.org</a>
</p>

That's it for the theory part. Now, for the implementation part, checkout the code in `LogisticRegression/logistic_regression.py`