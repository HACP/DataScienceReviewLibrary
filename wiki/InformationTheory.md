# A very brief introduction to Information Theory

Information Theory is a mathematical framework to quantify the information transmitted in a message through symbols. More precisely, information is a sequence of symbols transmitted through a medium or channel. 

Entropy is the amount of uncertainty in a sequence of symbols, but it does not happen in a vacuum, we often have knowledge of the distribution of symbols. If we think of the data to be modeled in our data science or machine learning problem, then we'd like to quantify the amount of bits needed to represent it. The most basic approach is to compute the entropy of each feature, since we can think of it as a sequence of symbols. At its core, the machine does not have any meta-knowledge about the symbol represented in the dat set. 

The Entropy $H_{X}$ of a discrete random variable, probability distribution pairs  $(X, p(x))$ is defined as 

```math
H(X) = - \sum_{x \in X}{p_i \log(p_i)} = \mathbb{E} \log \left( \frac{1}{p(X)}\right)
```

where we define $0 \log 0 = 0$ by continuity. An important remark: the second equality in the definition of entropy can be interpreted as the mean value of the logarithm of the inverse of the probability. 

Shannon's entropy quantifies precisely the minimum number of bits (when we use logarithm base 2) needed to transmit a message. Alternatively, Shannon's 
entropy can be interpreted as the average number of yes-or-not questions needed to describe the content of a message. 

The mutual information measures the amount of information that one random variable contains about another. Alternatively, it is the reduction in the uncertainty of one variable due to the knowledge of another.

``` math
I(X,Y) = \sum_{x,y} p(x,y) \log \left( \frac{p(x,y)}{p(x) p(y)} \right)
```

## Contribution of this project
In the file [InformationTheoryMetricsLib](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryMetricsLib.py), the function [get_entropy](https://github.com/HACP/DataScienceReviewLibrary/blob/ff6177f3950957da302e9e055a6e14ff7e60a3f3/code/src/InformationTheoryMetricsLib.py#L25) computes the entropy of a discrete distribution using the above equation. In the continuous case, we have issues with the binning. To overcome this challenges, we can define entropy as a function of the k-nearest distance. The function [get_mutual_information_mixed](https://github.com/HACP/DataScienceReviewLibrary/blob/ff6177f3950957da302e9e055a6e14ff7e60a3f3/code/src/InformationTheoryMetricsLib.py#L139) solves this issue. It is important to note that I(X;X) = H(X) holds only for discrete distributions.


## References 
- [1] https://web.mit.edu/6.933/www/Fall2001/Shannon2.pdf
- [2] https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
- [3] https://www.quantamagazine.org/how-claude-shannons-concept-of-entropy-quantifies-information-20220906/
- [4] https://proceedings.neurips.cc/paper/2017/file/ef72d53990bc4805684c9b61fa64a102-Paper.pdf

