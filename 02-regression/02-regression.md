## 2.5 Linear regression
Let $x_i$ be the feature vector and $g$ is the linear regression model. Then 
$$
g(x_i) = w_0 + \sum_{j=1}^n w_jx_i[j],
$$
where $w_j$, $j = 1,2,...,n$ is the wight for each feature and $w_0$ is the bias term.

In vector form:
$$
g(x_i) = w_0 + x_i^Tw.
$$
We can also write it in a concise way by integrating $w_0$ into the weight vector $w$ and add one component $1$ as the first component of $x_i$. Then
$$
g(x_i) = \sum_{j=0}^n w_j\~{x}_i[j],
$$
where $\~{x}_0 = 1$ and $\~{x}_i[j] = x_i[j]$ for $j = 1,2,...,n$.

Assuming that we have $m$ samples. We have the feature matrix of size $m\times (n+1)$
$$
\mathbf{X} = \begin{bmatrix}
1&X_{11}&\dots&X_{1n}\\
1&X_{21}&\dots&X_{2n}\\
\vdots&\vdots&\vdots&\vdots\\
1&X_{m1}&\dots&X_{mn}
\end{bmatrix}
$$

Then we have 
$$
\begin{bmatrix}
1&X_{11}&\dots&X_{1n}\\
1&X_{21}&\dots&X_{2n}\\
\vdots&\vdots&\vdots&\vdots\\
1&X_{m1}&\dots&X_{mn}
\end{bmatrix}
\begin{bmatrix}
w_0\\
w_1\\
\vdots\\
w_n
\end{bmatrix}
=
\begin{bmatrix}
X_1^T\cdot \mathbf{w}\\
X_2^T\cdot \mathbf{w}\\
\vdots\\
X_m^T\cdot \mathbf{w}
\end{bmatrix}
,$$
where the right side is the prediction vector.

## 2.7 Traing a linear regression model

Our goal is to find the "right" $\mathbf{w}$ for $g(\mathbf{X})=\mathbf{X}\cdot \mathbf{w}$ such that $g(\mathbf{X})$ as close to $\mathbf{y}$ as possible. To solve $\mathbf{X}\cdot \mathbf{w} = y$, we could multiply $\mathbf{X}^{-1}$ on both side to the left, then we can get $\mathbf{w} = \mathbf{X}^{-1}y$, assuming $\mathbf{X}$ is invertible. The problem is it is usually not the case because $\mathbf{X}$ is usually not even square. To solve this problem, we can first multiply on both sides with $\mathbf{X}^T$ to get the Gram-Matrix $\mathbf{X}^T\mathbf{X}$, which is $(n+1)\times(n+1)$, assuming that $\mathbf{X}$ is $m\times(n+1)$. Then we get 
$$
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}.
$$
By multiplying $(\mathbf{X}^T\mathbf{X})^{-1}$ on both sides we get
$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^Ty
$$

## 2.9 RMSE
How can we evaluate how good our model is? For this we introduce Root Mean Squared Error (RMSE).
$$
RMSE = \sqrt{\frac{1}{m}\sum_{i=1}^m(g(x_i)-y_i)^2},
$$
where $g(x_i)$ is the prediction for $x_i$ and $y_i$ is the actual value.

## 2.10 Validating the model
We have solved the model $g$ from the training data set, i.e. we get the result for $w$. Then we apply the model to validation data set and calculate the RMSE to validate the model.

## 2.12 Categorical variables
Categorical variables are usually strings, which are not numericals. But sometimes numerical datas can also be categorical, for example, "number of doors" in our data set.

For one categorical variable we use several binary variables to replace it. The number of binary variables is the number of different categories.

## 2.13 Regularization
Recall that 
$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^Ty
.$$ 
There could be problem when $\mathbf{X}^T\mathbf{X}$ is not invertible. For example, when $\mathbf{X}^T\mathbf{X}$ is a matrix which has two identical columns, it is not invertible. Python would throw an error information that the matrix is singular.  

The real problem is that when one column is very slightly different then the other column, then the matrix is actully numerically not invertibal. But python would still try to find the inverse. Then the inverse we get could be extreme large. 

The solution for this problem is that we add some very small number to the diagonal elements of the Gram matrix. 

## 2.14 Tuning the model
We compare the RMSE with different regularization parameters to choose the optimal one.

## 2.15 Using the model
Now we train the model using the train and validation data set combining together. 
