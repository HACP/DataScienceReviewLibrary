# Data Science and Information Theory Review Library

In this repo, we will explore common strategies and tools to attack data science problems from the lense of Information Theory. From exploratory data analysis (EDA), feature engineering, model training, model selection, evaluation, pipeline and story telling, among others, we will explore connections with concepts and practical aspects of Information Theory. 

We will use the [Titanic data set](https://www.kaggle.com/competitions/titanic) from [Kaggle](https://www.kaggle.com)  as our use case and will build a library of functions in python to streamline the approach. Additionally, a wiki with in-depth explanation of concepts, tools and algorithms will develop some concepts and details further. We will write the librabry following a basic form of Test driven development (TDD) and [pytest](https://docs.pytest.org/en/7.1.x/).

This repo is intended as a personal review of data science concept connected to Information Theory aiming to delvelop a python package that can be reused and expanded over time. 

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
