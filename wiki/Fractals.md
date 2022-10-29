# Fractals, Dimensions and Information Theory

Mandelbrot provided a beautiful definition of fractals as "shapes whose roughness and fragmentation neither tend to vanish nor fluctuate up and down,
but remain essencially unchanged when one zooms in continually and examination is refined. Hence the structure of every piece holds the key to the 
whole structure"

## Henon Map

A Henon Map is a discrete-time dynamical system that exhibits chaotic behavior. With parameters $a=1.4, b=0.3, x_0=0, y_0=0$ the map follows these rules
```math
x_{n+1} = 1 - a x_n + y_n; 
y_{n+1} = b x_n
```

After several iterarions the map or attractor looks like this
![Henon Map](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/Henon.png)

This is an animation of the Henon Map for n*1000 iterations (n=1,1000)
![Henon Map Animation](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/HenonAnimation.gif)


We can use the box-counting method to estimate the dimension of the attractor 1.26 which reveals the non-interger nature of the fractal dimension.
![Henon Map Dimension](https://github.com/HACP/DataScienceReviewLibrary/blob/main/wiki/assets/figures/fractal_dimension.png)

## Contribution of this project
In the file [InformationTheoryClusteringLib](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/src/InformationTheoryFractalsLib.py), we develop two main functions: 
- [Henon Map](https://github.com/HACP/DataScienceReviewLibrary/blob/05b8f37e62ddda3d692919f1b1b5d19dc0b72b8a/code/src/InformationTheoryFractalsLib.py#L6): An implementation of the Henon Map, simple discrete-time dynamic system with chaotic and fractal features
- [box counting dimension](https://github.com/HACP/DataScienceReviewLibrary/blob/05b8f37e62ddda3d692919f1b1b5d19dc0b72b8a/code/src/InformationTheoryFractalsLib.py#L64): Estimates the fractal dimension using the box counting method. 

### Tesing
We used a TDD approach to build the library and the tests cases are [here](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/test/InformationTheoryFractalsLib_test.py). Additionally, we explored the approach further in a [juypiter notebook](https://github.com/HACP/DataScienceReviewLibrary/blob/main/code/notebooks/Fractal%20Dimensions%20Test%20Cases.ipynb)


## References
[1] https://users.math.yale.edu/mandelbrot/web_pdfs/fractalGeometryWhatIsIt.pdf
