# A very brief introduction to Information Theory

Information Theory is a mathematical framework to quantify the information transmitted in a message through symbols. More precisely, information is a sequence of symbols transmitted through a medium or channel. 

Entropy is the amount of uncertainty in a sequence of symbols, but it does not happen in a vacuum, we often have knowledge of the distribution of symbols. If we think of the data to be modeled in our data science or machine learning problem, then we'd like to quantify the amount of bits needed to represent it. The most basic approach is to compute the entropy of each feature, since we can think of it as a sequence of symbols. At its core, the machine does not have any meta-knowledge about the symbol represented in the dat set. 

The Entropy $H_{X}$ of a discrete random variable, probability distribution pairs  $(X, p(x))$ is defined as 

```math
H(X) = - \sum_{x \in X}{p_i \log(p_i)} = \mathbb{E} \log \left( \frac{1}{p(X)}\right)
```

where we define $0 \log 0 = 0$ by continuity. An important remark: the second equality in the definition of entropy can be interpreted as the mean value of the logarithm of the inverse of the probability. 

Shannon's entropy quantifies precicely the minimum number of bits (when we use logarithm base 2) needed to transmit a message. Alternatively, Shannon's 
entropy can be interpreted as the average number of yes-or-not questions needed to describe the content of a message. 


## References 
- [1] https://web.mit.edu/6.933/www/Fall2001/Shannon2.pdf
- [2] https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
- [3] https://www.quantamagazine.org/how-claude-shannons-concept-of-entropy-quantifies-information-20220906/
