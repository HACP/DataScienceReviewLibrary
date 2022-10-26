# Data Science and Information Theory Review: Down the rabbit hole 

This repository is a collection of observations, codes and references from a personal approach to data science from the persective of Information Theory. The process to develop these notes was non-linear, often following a chain of somewhat connected topics related to Information Theory. Many of the techniques, codes and approaches that we employed can be used across different problems. That was the motivation to attempt to write a "library" of functions. From exploratory data analysis (EDA), feature engineering, model training, model selection, evaluation, pipeline and story telling, among others, we will explore connections with concepts and practical aspects of Information Theory. We will explore a few applications in network theory, flowck dynamics, reinforcement learning etc. Real world applications will be highlighted.

We will use a set of well-known benchmark datasets (e.g [Titanic data set](https://www.kaggle.com/competitions/titanic) from [Kaggle](https://www.kaggle.com))  to build our use cases and will develop a library of functions in python to streamline the approach. We will write the librabry following a basic form of Test driven development (TDD) and [pytest](https://docs.pytest.org/en/7.1.x/). Additionally, a wiki with in-depth explanation of concepts, tools and algorithms will develop some concepts and details further.


## Contents
### Information Theory - Concepts and applications
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/InformationTheory.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryMetricsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryMetricsLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Mutual%20Information%20Test.ipynb)
### Dimensionality: Volume and surface area of n-balls; Curse of Dimensionality
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Dimensionality.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryDimensionLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryDimensionLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Dimensionality%20Montecarlo%20Test.ipynb))
### Fractals: Henon Map and box-counting dimension estimation
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Fractals.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryFractalsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryFractalsLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Fractal%20Dimensions%20Test%20Cases.ipynb)
### Clustering: kMeans and Free Energy
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/Clustering.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryClusteringLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryClusteringLib_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Clustering%20kMeans%20Test.ipynb)
### Flock Dynamics: Vicsek Model and Particle Swarm Optimization
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/FlockDynamics.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryFlockDynamicsLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryFlockDynamics_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Flock%20Dynamics%20and%20entropy%20Test.ipynb)
### Reinforcement Learning and Networks
- [wiki](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/ReinforcementLearning.md)
- [python library](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryQLearningLib.py) and [tests](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryQLearning_test.py)
- [jupyter notebooks](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Reinforcement%20Learning%20-%20Shortest%20Path%20in%20Network%20Test%20Case.ipynb)

