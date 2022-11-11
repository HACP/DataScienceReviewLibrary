# Customer Lifetime Value - BG/NBD model

The Beta Geometric (BG), Negative Binomial Distribution (NBD) Model was introduced in [“Counting Your Customers” the Easy Way:
An Alternative to the Pareto/NBD Model, Fader, Hardie and Lee (2005)](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) as way to analyze customer -base by estimating parameters from the input data. 

> We develop the BG/NBD model from first principles and present the expressions required for making individual-level statements about future
buying behavior. [Fader2005]

This is a great advantage of this model: it presents a way to build the model from reasonable assumptions and a few parameters with clear business interpretation, describes how to estimates the parameters using Maximum Likelihood Estimation Methods and finally showcases its predictve power. 

## Model inputs and parameters

The model inputs are:
* $T$: length of the time period during which repeated transaction have occured (T = 39 - time of first purchase)
* $x$: number of transactions in the period T (frequency)
* $t$: time of the last transaction (recency)

The model parameters are:
* $r$
* $\alpha$
* $a$
* $b$

## Model Assumptions:
* The number of transactions made by a customer follows a Poisson Distribution with transaction rate $\lambda$. In other words the time $t$ between transactions is distributed exponentially $f(t_i | t_{i-1}; \lambda)$ (the probability of $t_i$ depends only on $t_{i-1}$ and the parameter $\lambda$
* The parameter $\lambda$ is not unique and follows a gamma  distribution with parameters $r$ and $\alpha$, $f(\lambda | r, \alpha)$. In other words,  _customers have different shopping behaviors_. 
* Once a transaction occurs, the customer becomes inactive with probability $p$, the time of "drop-out" is distributed as a shifted geometric distribution $P(inactive)$ _customers might drop off and do not come back_
* $p$ follows a beta distribution, $f(p|a,b)$ _the probability of dropping off is not the same for all customers_
* The transaction rate $\lambda$ and drop-put probability $p$ vary independently across customers 

[Here](https://stats.stackexchange.com/questions/251506/is-it-possible-to-understand-pareto-nbd-model-conceptually) there is an excellent explanation of the assumptions. 

## Maximum Likelihood Estimation
Our model's parameter set ultimately reduces to $r, \alpha, a, b$ and we can estimate this set from the experimental data using the [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). The idea is to find the parameters that under the assumed statistical model the oberved data is most probable. Equations 6 and 7 in [Faber2005] provide the nessary equations to compute the log-likelihood. We can the use any standard optimization method to find the parameters that maximize the log-likelihood (e.g. Scipy optimization package) 

Out of the 14 available optimizers, 9 did not need a Jacobian (not provided). Out of the 9, one (BFGS) gave a False successful flag with values within range and two (TNC, SLSQP) gave a True successful Flag but the numbers are completely off from the range. 

can we use the different optimization methods as way to find the mean and std of the parameters? 

## Data and Preprocessing
In this project we will use the same dataset described in [Faber2005] the CDNOW small data set. The data contains information about sales at the customer level (anonimized), the date of the transaction, and the quantity/total price of the purchase. The data was collected from January 1997 through June 1998 and we will focus on the first 39 weeks. We have a total of 6919 corresponding to 2357 unique customers. 

Data Summary Table

| Column Name   | Type          |
| ------------- | ------------- |
| Customer_ID   | string        |
| Date          | string        |
| Quantity      | number        |
| Amount        | number        |

We will transform the Data field from string to pandas date object. The key varables in our model are:
* T: length of the time period during which repeated transaction have occured (T = 39 - time of first purchase)
* x: number of transactions in the period T (frequency)
* t: time of the last transaction (recency)

Note that if x = 0 then t = 0

## Discussion - Interpretability vs Accuracy

The generic form of the Interpretability vs Accuracy Trade-off says that in general more powerful models (accuracy wise) are less likely to provide clear explanations of their results. 

Two important questions to consider were discussed in [“Why Should I Trust You?” Explaining the Predictions of Any Classifier, Ribeiro (2016)](https://arxiv.org/pdf/1602.04938.pdf)
- Do we trust the model? In this case we are using sound assumptions to build our model from first principles. Does it guarantee trust? No.
- Do we trust the predictions? TBD

