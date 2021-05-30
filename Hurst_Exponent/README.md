<h1 align='center'> Exposant de Hurst : H </h1>

[<h1 align='center'>![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gruz77/Deep-Learning-in-Finance/blob/main/Hurst_Exponent/Hurst_exponent.ipynb)</h1>

The Hurst exponent is directly linked to the notion of fractional movement, it is an indicator of long memory of time series. 
- For 0 < H < 0.5, we are in an anti-persistent trend, the mean-reverting principle, 
- For H = 0.5, we have a standard Brownian motion (Wiener process),
- For 0.5 < H < 1, we have a persistent trend (positive long-term autocorrelation).

## Project goals 
- Implementation of the CNN of [H.Stone's paper, QF (2020)](https://arxiv.org/pdf/1812.05315v3.pdf) for the estimation of H
- Our goal is to find a robust architecture of a "simple" dense network that can do better than the previous CNN for estimating values of H < 0.5, from an MBE, RMSE and MAE point of view.
- Training Set (CNN and ANN): 
  - For 10 values of H ([0,1]), generation of 10,000 fractional Browninen motion time series of length T = 100 (library *fBm*)
- Test Set:
  - For 100 values of H ([0.01,0.99]), generation of 1000 time series of length T = 100

## Comparaison with wavelets
- We also add a comparison with the Wavelets estimators via the *liftLRD* package of R (*rpy2* allows to import R functions in Python).

## Conclusion: 
- The dense ANN network is slightly better than the CNN for the estimation of H < 0.4. This is sufficient to save the ANN model for future use (recent research showing that the Hurst coefficient for financial series is around 0.1: Volatility is rough, J.Gatheral, M.Rosenbaum](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1393551)).
- The wavelets are better for 0.8 < H < 1, but not outside.
