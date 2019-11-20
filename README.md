# Machine-Learning-Multiple-Linear-Regressin-with-Backward-Elimination

Multiple Linear Regression (aka MLR) is statistical techique for predict to the outcome dependent variable with using multi independet variable. 

MLR assumes that there was a linear relationship between the dependent variable and the independent variable like as Simple Linear Regression (SLR). But the main differences between SLR and MLR, you have multiple independent variable in the MLR unlike SLR. 

**Tips** The *scatter* plot can show the linear relationship in your data.

For more details visit this: [MLR-Explaination](https://www.investopedia.com/terms/m/mlr.asp)

## Installation the Library
In this tutorial I used the statsmodels library so you have to install to

For Python pip installation
```
pip install statsmodels
```
For Anaconda installation
There's a trick in here, I have Anaconda distrubitions like Spyder or Jupyter Notebook. So, I have been using a virtual envoirement on Spyder. İf you have too, activate the virtual envoirement first.
```
activate <your virtual env. name>
conda search statsmodels
```
If the conda could find to the libraries than type this:
```
conda install -c conda-forge statsmodels
```
## Usage
After the download all files in the repo;
1. Open your favorite editor for Python (PyCharm, Spyder or others)
2. Under the Files, you should see like Open Project or File.
3. Navigate to the download section and choose .py file.
4. Check one by one the all librarries works.
5. Run the sections divided by "#%%".

## What's the Backward Elimination and Why I need to use this?
So to say working dozens of independent variables to predict a dependent variable could be unpleasent-unlovable for you. Actually all of us hate to the choose correct features to the ML algorithm. So, Backward Elimination comes up with a solution thanks to Statistics.  

In the Statistics there is a huge topics called as P-Value. [Detailed](https://www.youtube.com/watch?v=KS6KEWaoOOE)

To understanding the P - Values we have to mention that the null hypothesis: The null hypothesis is the assumption that the parameters associated to your independet variables are equal to zero. So, this means you doesn't have a certain pattern to the observations. The P-Value is the probability that the parameters associated to your independent variables have certain nonzero variables when the null hypothesis is True.

A more plenty explanition is the lower P-Values are the good guys, higher P-Values are the bad guys, and there is no ugly. [Movie](https://www.imdb.com/title/tt0060196/)

## Manually Backward Elimination
In your ML algorithm you should do this elimanation methods once manually for a good understanding. I mean, I did it manually for many times in the past. After the some practices you can use a for loop to do this automatically.

```
import statsmodels.api as sm
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
```

**The OLS** means that the Ordinary Least Squares. Nothing different from the Linear Regression there are same things.

```
X = sm.add_constant(X)
y = np.dot(X, beta) + e
model = sm.OLS(y, X)
results = model.fit()
model.summary()
```

This gives you a good summary of the regression. We will give attention to just P-Values column. 

```
                           OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 4.020e+06
Date:                Tue, 19 Nov 2019   Prob (F-statistic):          2.83e-239
Time:                        05:11:47   Log-Likelihood:                -146.51
No. Observations:                 100   AIC:                             299.0
Df Residuals:                      97   BIC:                             306.8
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.3423      0.313      4.292      0.000       0.722       1.963
x1            -0.0402      0.145     -0.278      0.781      -0.327       0.247
x2            10.0103      0.014    715.745      0.000       9.982      10.038
==============================================================================
Omnibus:                        2.042   Durbin-Watson:                   2.274
Prob(Omnibus):                  0.360   Jarque-Bera (JB):                1.875
Skew:                           0.234   Prob(JB):                        0.392
Kurtosis:                       2.519   Cond. No.                         144.
==============================================================================
```

In the above there was a artificial data and I did not kick out the highest P-Values. It doesn't make any sense to do that for an artificial data. But you should do this in your ML algorithm that have linear relationship between variables.

Still if I do that it's seems like;

```
X = X[:,[0,2]]
```
This means I kicked out the first 1 indexed independent variable. After the obtaining new X, you should fit the algorithm on your new data and labels. And run the .summary command again until when there is no higher values than the significant level.
