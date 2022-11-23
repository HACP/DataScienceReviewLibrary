# Data Science and Information Theory Review: Down the rabbit hole 

> Knowledge, as justified true belief, is normally seen as resulting from lucky guesses, but optimality provides a sounder basis to epistemology and, therefore, this can be the standard against which the inherent dimensionality of data may be compared [Information theory and dimensionality of space, Kak (2020)](https://www.nature.com/articles/s41598-020-77855-9)

This repository is a collection of observations, codes and references from a personal approach to data science from the persective of Information Theory. The process to develop these notes was non-linear, often following a chain of somewhat connected topics related to Information Theory. Many of the techniques, codes and approaches that we employed can be used across different problems. That was the motivation to attempt to write a "library" of functions. From exploratory data analysis (EDA), feature engineering, model training, model selection, evaluation, pipeline and story telling, among others, we will explore connections with concepts and practical aspects of Information Theory. We will explore a few applications in network theory, flowck dynamics, reinforcement learning etc. Real world applications will be highlighted.

> By considering the event as primary, we can measure its information as related to its frequency: the less likely the event, the information is greater. The logarithm measure associated with information has become widely accepted because it is additive. In other words, additivity of information is the underlying unstated assumption in the information-theoretic approach to structure and since information is related to the experimenter, this constitutes a subject-centered approach to reality. The entropy, or the average information associated with a source, is maximized when the events have the same probability. [Information theory and dimensionality of space, Kak (2020)](https://www.nature.com/articles/s41598-020-77855-9)

We will use a set of well-known benchmark datasets (e.g [Titanic data set](https://www.kaggle.com/competitions/titanic) from [Kaggle](https://www.kaggle.com))  to build our use cases and will develop a library of functions in python to streamline the approach. We will write the librabry following a basic form of Test driven development (TDD) and [pytest](https://docs.pytest.org/en/7.1.x/). Additionally, a wiki with in-depth explanation of concepts, tools and algorithms will develop some concepts and details further.

> Thisalgorithmbelongsideologicallytothatphilosophicalschool that allows wisdom to emerge rather than trying to impose it, that emulates nature rather than trying to control it, and that seeks to make things simpler rather than more complex. Once again nature has provided us with a techniquefor processing informationthat is at once elegant and versatile. [Particle Swarm Optimization, Kennedy and Eberhart (1995)](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)


## Contents

### Information Theory - Concepts and applications
> Can we define a quantity which will measure, in some sense, how much information is “produced” [by such a process], or better, at what rate
information is produced? [A Mathematical Theory of Communication, Shannon (1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

Data Science leverages models to convert raw data into (actionable) information. How much information is encoded in the data at hand is a great starting question. In this section we introduce the concept of entropy in the context of information theory. Then we discuss the challenges with continuous and mixed distributions and implement a method to estimate entropy and mutual information for mixed distributions. We consider the relevance matrices, a matrix buitl from the pair-wise mutual information from a data set and explore the behavior for different MI thresholds [link](http://groups.csail.mit.edu/medg/ftp/butte/masters.pdf). The animation below shows the change of the relevance matrix for the CTR dataset as we vary the mutual information threshold [more](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/InformationTheory.md)


![Relevance Matrix for CTR](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/RelevanceNetworkCTR2.gif)


- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/InformationTheory.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryMetricsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryMetricsLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Mutual%20Information%20Test.ipynb)

### Dimensionality: Volume and surface area of n-balls; Curse of Dimensionality

#### What do we mean by _dimension_?

In physics, one definition of _degrees of freedom_ for a general mechanical system is the following: 
> The number of degrees of freedom of a system is the number of independent variables that must be specified to define completely the condition of the system [Reference0](https://www.researchgate.net/publication/239538229_On_the_Computation_of_Degrees-of-Freedom_A_Didactic_Perspective/link/0deec5349199f10f5b000000/download).

A similar concept exists in data science - given a data set and a problem (context), what's the minimum number of independent variables needed to specify completely the system so that we can explain/predict the outcome of a future (unseen) state. 

Often times, we are dealing with a data set with several features, but only a few convey relevant information about the problem in hand. Ideally, we would like to identify the number of degrees of freedom of our system, from the data set. The dimension of the configuration space makes the concept of number of degrees of freedom more precise with the following intuitive notions [Information Dimension and the Probabilistic Structure of Chaos, Farmer (1982)](https://zfn.mpdl.mpg.de/data/Reihe_A/37/ZNA-1982-37a-1304.pdf). 

- Direction, related to the topological dimension 
- Capacity, related to fractal dimension
- Measure, related to information dimension.


#### Structure of high dimensional spaces

> We might be pardoned for supposing that in a space of infinite dimension we should find the Absolute and Unconditioned if anywhere, but we have reached an opposite conclusion. This is the most curious thing I know of in the Wonderland of Higher Space. [Properties of the locus r=constant in space of n dimensions, Heyl (1897)](https://books.google.com/booksid=j5pQAAAAYAAJ&pg=PA38&lpg=PA38&dq=%22We+might+be+pardoned+for+supposing+that+in+a+space+of+infinite+dimension+we+should+find+the+Absolute+and+Unconditioned+if+anywhere,+but+we+have+reached+an+opposite+conclusion.+This+is+the+most+curious+thing+I+know+of+in+the+Wonderland+of+Higher+Space.%22&source=bl&ots=68yeF8EDZu&sig=ACfU3U3A6ISL_MrOXuKCudEHKUmEnR1W7Q&hl=en&sa=X&ved=2ahUKEwirn_HAv_76AhUKkmoFHZD3CA0Q6AF6BAgJEAM#v=onepage&q=%22We%20might%20be%20pardoned%20for%20supposing%20that%20in%20a%20space%20of%20infinite%20dimension%20we%20should%20find%20the%20Absolute%20and%20Unconditioned%20if%20anywhere%2C%20but%20we%20have%20reached%20an%20opposite%20conclusion.%20This%20is%20the%20most%20curious%20thing%20I%20know%20of%20in%20the%20Wonderland%20of%20Higher%20Space.%22&f=false)

Our intution of dimensions is heavely influenced by our 3-dimensional world. We tend to extrapolate what happens in 1,2 and 3 dimension to higher dimensions and often our intuition is wrong. In this section, we will consider the volume of the unit ball in n dimensions, the ratio of this volume vs the volume of the n dimensional cube that contains it. As we will see, this ratio tends to zero as the dimension increases - most of the volume of the n-cube is located _outside_ the n-ball, as the cube expands the ball shrinks, both at impressive rates. [Reference1](https://www.americanscientist.org/article/an-adventure-in-the-nth-dimension)


> There is little difficulty in applying the calculus of variations, dynamic programming, or any of a number of direct methods, to systems described by state vectors of low dimension to obtain efficient computational techniques. When, however, the dimension is “large” (a relative rather than absolute property), or infinite, the “curse of dimensionality” prevents the direct use of general methods. [A new type of approximation leading to reduction of dimensionality in control processes, Bellman (1969](https://pdf.sciencedirectassets.com/272578/1-s2.0-S0022247X00X04635/1-s2.0-0022247X69900614/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJHMEUCIFTKRvHs4pH0sG44Kb63Hv6jtzmxP%2FbOK4Zykr8t8l9LAiEA%2FondaDNP3mWp5IlLkJpedwb0s1kIOs17aLXsBcr5EiAq1QQI8%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDF2n6BI76D3wujmGzyqpBDpIR%2BX5u%2B5uZP8vR%2B%2FjA92EcenH4wO7ZRIcgUqHL1t0PbW6aQoutK2tJRQjGBhqnwraThNNDwSSyVDbE%2BCYM4I2A2jN6wXj3x%2BV%2FKNuLaSy0SwpPeBTC%2B2iVLjXvj%2BOF1gXC4SYrb65rdLqe2cyPJSYzl82BNO8DY4ocmLydQ%2BpCQzU1aJYA6rrHucROK9dpoVzBZVQK2URZShPGJp0oOPLAfe%2F%2F3GlJXfvHNmnbO2hH1Uu2l67j9AuEBDB7DTitZzaGWO6hfGHa7g9k0W99jW1K4UpGI%2BLld%2Bu5MJgOvi4Cl6eJ8%2FBzloALzewi0J92BPwNoyJVWO5PXOb8wqYsuS6zLxvXhn52%2FNHTm6l9d0J%2FhpSnOj%2FsV4LEPyDE7%2FXzxRrO7xTi5apfaGYqBDcpK8rk5zFFraq3j%2FROGFwNrHkRLxow1yC7sP4QAFZZLKcLnL5XweooDeLLyLLu2cHMwNiCzZMlGLTQQWCiu5kSMg0qAWCHdfJik5x98awosEBbgjY6IhKXoJ3h28FsGf7J%2FVKq7OM7e0%2BxFmuCK1i5noSY3PNh1NahxUhqaOjSronYOVjMPs5y%2FP%2FVjSoG6tdbqSTXVIKdVa1XHhd9%2BY30%2Fg3IwbGpedc5xjylgxqOlL29M5pf6oVJEuGJaHIbAYFQ5nz1JtHG9JVJDRZojZex8JTvxK%2BQhjvr9T%2B1eLlOxEdCCEP2QGXI0PqOon6%2FgEuOicZQoloEV6jCrUw1%2BnlmgY6qQE3B%2BQtUualMH2Ixl95BJT87xRhVWVvaZ%2FwXD40hgqVuqSVvEs7Dhw7PNi%2BHiOE0TbmrxRNQku7TwWOTJxatTf5FEhzpHjygUBPaCcm1ddTcwS3Olc6tHXe9%2BnoU7Dpuq%2FaJ1T%2Fh8u4rGmYlNyuaBNke2ADjGgERZGHQzC5i%2FJguf%2F4uclDs0qLXvLGxFPkd%2Brs%2BgFm0hv%2BNXHXm%2FZrbp65N%2BVN1PkMYt1t&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221026T183857Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXKNSM77X%2F20221026%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=241dc4a4ca3d6787dd4c3e89f6048105e270789faafa5262ad5639f6ec255055&hash=b6bc51c30d8d7eb84f1fa6b479e970cc48ac7ebec47765b8ca50ed09cb5f7f16&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0022247X69900614&tid=spdf-dff6d7fe-deb8-4a7b-ac6c-31c7b89d75f0&sid=92cd07c07aa42147465a99d2fb501d1ec7ccgxrqa&type=client&ua=51545e535b0705565705&rr=76054fd85f3707ee)

The behaviour of points/data in high dimensions has a profound impact in data analysis. Many data analysis and machine learning algorithm rely on the concept of neighbors - points within a certain distance. If we consider the n-ball around a fixed point, as n increases the chances that a another point will land within the n-ball (i.e. become neighbors) decreases very fast. The whole notion of neighborhood needs to be revisited. 

In this repository we will discuss the volume and area of an n-ball both analytically and with a Monte Carlo simulation. Below, we see a simulation of the MC computation of the ratio of the volume n-ball vs n-cube for different values of p-norm [more](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Dimensionality.md)

![Ratio Volume Ball and Cube](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/RatioVolumeCubeBallAnimation.gif)


- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Dimensionality.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryDimensionLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryDimensionLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Dimensionality%20Montecarlo%20Test.ipynb))

### Fractals: Henon Map and box-counting dimension estimation

> Noninteger dimensions provide explanation for recursion and scale-invariance in complex systems—biological, physical, or engineered—and explain fractal behavior, examples of which include the Mandelbrot set, patterns in Romanesco broccoli, snowflakes, the Nautilus shell, complex computer networks, brain structure, as well as the filament structure and distribution of matter in cosmology. [Information theory and dimensionality of space, Kak (2020)](https://www.nature.com/articles/s41598-020-77855-9)

In this section we will construct an example of a fractal: the Henon Map and we will compute its dimension. Below is an animation of the Henon Map for n*1000 iterations (n=1,1000) [more](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Fractals.md). 
![Henon Map Animation](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/HenonAnimation.gif)

The link between fractals and information theory is the notion of information dimension: 
> The information dimension, D1, measures the fractal dimension of a probability distribution, and relates the growth of Shannon entropy
to how the system under study is discretized [A new fractal index to classify forest disturbance and
anthropogenic change](https://assets.researchsquare.com/files/rs-1934944/v1/3127293a-a750-405a-b427-000c0fb99f94.pdf?c=1660059355)

> The direct physical relevance of the information dimension is in measurement. Knowledge of the information dimension of an attractor allows an observer to estiamte the information gaines when a measurement is made at a given level of precision [Information Dimension and the Probabilistic Structure of Chaos, Farmer (1982)](https://zfn.mpdl.mpg.de/data/Reihe_A/37/ZNA-1982-37a-1304.pdf)

- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Fractals.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryFractalsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryFractalsLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Fractal%20Dimensions%20Test%20Cases.ipynb)

### Clustering: kMeans and Free Energy
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Clustering.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryClusteringLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryClusteringLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Clustering%20kMeans%20Test.ipynb)

### Flock Dynamics: Vicsek Model and Particle Swarm Optimization
> It does not seem a too-large leap of logic to suppose that some same rules underlie animal social behavior,includingherds,schools,andflocks,andthatofhumans. AssociobiologistE.0.Wilson[9] has written, in reference to fish schooling, “In theory at least, individual members of the school can profit from the discoveries and previous experience of al other members of the school during the searchforfood. Thisadvantagecanbecome decisive,outweighingthedisadvantagesof competition for food items, whenever the resource is unpedictably distributed in patches” (p.209). This statement suggeststhat social sharingof informationamong conspeciatesoffers an evolutionaryadvantage:this hypothesiswas fundamentalto the developnmt of particle swarm optimization [Particle Swarm Optimization, Kennedy and Eberhart (1995)](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf). 

In the animation below, we can see the particle swarm optimization algorithm at work, finidn the minimum point in a non-linear, continuous 2D surface. WE can see how the particles start in a random configuration and as they collect the particle best and share them to find the global best after a few iterations the swarm lands very close to the optimal value. 

![PSO](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/PSO.gif)

> Physical space of course affects informational inputs, but it is arguably a trivial component of psychological experience. Humans learn to avoid physical collision by an early age, navigation of n-dimensional psychosocial space requires and many of us never seem to acquire quite al the skills we need [Particle Swarm Optimization, Kennedy and Eberhart (1995)](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)

> Holland’s chapter on the “optimum allocation of trials” [5] reveals the delicate balance between conservative testing of known regions versus risky exploration of the unknown. It appears that the current version of the paradigm allocates trials nearly optimally. The stochastic factors allow thorough search of spaces between regions that have been found to be relatively good, and the momentum effect caused by nmhfying the extant velocities rather than replacing them results in overshooting,or exploration of unknown regions of the problem domain. [Particle Swarm Optimization, Kennedy and Eberhart (1995)](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)

- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/FlockDynamics.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryFlockDynamicsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryFlockDynamics_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Flock%20Dynamics%20and%20entropy%20Test.ipynb)

### Ant Colony Optimization

> Ant Colony Optimization algorithms, that is, instance of the ACO metaheuritics (...) use a population of ants to collectively solve the optimization problem under consideration (...). Information collected by the ants during the search process is stored in pheromone trails $\tau_{i,j}$ associated to connections $l_{i,j}$. Pheromone trails encode a long-term memory about the whole ant search process. [Anto Colony Optimization: A new Metaheuristic Dorigo and Dicaro (1999)](http://staff.washington.edu/paymana/swarm/dorigo99-cec.pdf)

![ACO toy model](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/ACO_toymodel_2.gif)
- [notebook](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Ant%20Colony%20-%20Simulations.ipynb)

### Reinforcement Learning and Networks
> Q-learning (Watkins, 1989) is a simple way for agents to learn how to act optimally in controlled Markovian domains. It amounts to an incremental method for dynamic programming which imposes limited computational demands. It works by successively improving its evaluations of the quality of particular actions at particular states [Q-Learning, Watkins and Dayan, 1992](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)

![Maze Path RL](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/Maze_Path_Animation.gif)

- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/ReinforcementLearning.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryQLearningLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryQLearning_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Reinforcement%20Learning%20-%20Shortest%20Path%20in%20Network%20Test%20Case.ipynb)

### Largest Empty Circle Problem: Blue Noise and Voronoi Diagrams

Suppose we need to find the location of the next store for a national retail chain withthe condition of being located in the largest area without presence; or suppose we want to locate a waste facility that is as far as possible from current houses. The Largest Empty Circle answers those questions. 

> The largest empty circle (LEC) problem is defined on a set P and consists of finding the largest circle that contains no points in P and is also centered inside the convex hull of P[The Largest Empty Circle Problem, Schuster (2008)](https://www.cs.umd.edu/~mount/Papers/crc05-prox.pdf).

In order to facilitate the visualization, we'd like to locate random points that are not clustered or very close to each other. We use a Possion Disk distribution to simulate a blue noise sample pattern. 

> Blue noise sample patterns—for example produced by Poisson disk distributions, where all samples are at least distance r apart for some user-supplied density parameter r—are generally considered ideal for many applications in rendering. [Fast Poisson Disk Sampling in Arbitrary Dimensions, Bridson (2007)](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf)

![Largest Empty Circle Example](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/LEC.png)

![Voronoi Example](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/Voronoi_Animation.gif)

### Levenshtein Distance and word paths
> [For those channels] by analogy to the combinatorial problem of construction optimal codes capable of correcting s reversals, we will consider the problem of construction optimal codes capable of correcting deletions, insertions, and reversals. [Binary Codes Capable of Correcting Deletions, Insertions and Reversals. Leveinshten (1966)](https://bibbase.org/network/publication/levenshtein-binarycodescapableofcorrectingdeletionsinsertionsandreversals-1966)

In this example we will consider a corpus of words in the english language and compute the Levenshtein distance to build word paths from a source word to a target words. Each move from word to word corresponds to a insertion a delation or a reversal (exchange). Below we see a representation of the word network revealing a central cluster of connected words and an outer ring of disconnected words. 

![Word Path Example](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/WordPath_well_to_john.gif)

The animation shows a path between the word well and the word john. For this particular case, we see only exhanges since all words have the same length.
![Word Network](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/WordNetwork_4_log12.png)

> The problem of multidimensional scaling, broadly stated, is to find n points whose interpoint distances match in some sense the experimental dissimilarities of n objects. Instead of dissimilarities the experimental measurements may be similarities, confusion probabilities, interaction rates between groups, correlation coefficients, or other measures of proximity or dissociation of the most diverse kind. Whether a large value implies closeness or its opposite is a detail and has no essential significance. What is essential is that we desire a monotone relationship, either ascending or descending, between the experimental measurements and distances in the configuration [MULTIDIMENSIONAL SCALING BY OPTIMIZING GOODNESS
OF FIT TO A NONMETRIC HYPOTHESIS, Kruskal (1964)](http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964a.pdf)

We have used a multidimensional scaling to represent the words from the entire corpus and the path from source to target. 
![Word MDS](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/WordMDS_4_log12.png)

## Applications
### Customer Lifetime Value - BG/NBD model
We employ maximum likelihood estimation (MLE) to find the parameters that best fit a model built from first principles that describes the probability of purchase for a Recency and Frequency model. The model is a combination of a beta-gemoetric and negative binomial distributions, with exact and analitically explicit equations. The model is highly interpretable with acceptable accuracy (~2% total forecast) 
[wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/CLV_BG_NBD_model.md)
[notebook](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/CLT%20-%20BG%20NBD%20Model.ipynb)

### Click Through Rate and Maximum Entropy
> Despite almost 10 billion ad impressions per day in our dataset, hundreds of millions of unique user ids, millions of unique pages, and millions of unique ads, combined with the lack of easily generalizable features, makes sparsity a significant problem [Simple and Scalable Response Prediction for Display Advertising, Chapelle et al, (2014)](http://wnzhang.net/share/rtb-papers/ctr-chapelle.pdf)


