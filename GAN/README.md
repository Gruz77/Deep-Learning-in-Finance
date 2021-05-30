<h1 align='center'> GANs: Time Series Generation </h1>

[<h1 align='center'>![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gruz77/Deep-Learning-in-Finance/blob/main/GAN/GAN.ipynb)</h1>

The applications of GANs are as diverse as they are [impressive](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/). 
Here we are interested in the generation of time series. This can be extremely useful in the case of strategy backtesting, to avoid overfitting. This would allow access to an almost infinite universe of possibilities, and thus to have strategies whose significance could be all the more telling. ("Train on fake, trade on real"). One very important thing to verify that GAN has generated valid financial series is to check the stylised facts (fat tails, volatility with long memory, ... [see here](https://github.com/Gruz77/Physics-of-Markets/tree/main/Stylized_Facts))

- We use the daily data of the S&P500 index since 1928. We work as usual on log returns.

## Construction and training
- We construct the GAN : 
  - The generator will have an input of size D = 10 and an output of size T = 200
  - The discriminator has an input of size T (generator output) and a single sigmoid activation output: in order to discriminate (financial series or not)
  - Our GAN object will thus have as input that of the generator, and as output that of the discriminator. The goal is that the discriminator does not know how to differentiate the true time series from the false ones, and thus returns a probability of 0.5 once trained, for each vector provided in input.

- Training the GAN for each batch (of size 32 here):
  - We generate M subsamples of size T of our log returns (Xreal, size MxT)
  - M samples of noise vectors of dimension D are generated (Noise, size MxD)
  - With these noise samples, we use the generator to predict M yield vectors (Xgen, size MxT), which we concatenate in rows to Xreal to get Xrealgen (size 2MxT)
  - We define the vector Yrealgen = (1,...,1,0,...,0) (size 2M), which we shuffle so that the network does not learn from the order of the lines
  - We train the discriminator
  - We create our noise matrix to fool the discriminator (Noise', size M'xD, with M'=M here)
  - Define our vector Yfake = (1,...,1) (size M') pretending that the above noise samples are real time series -> fool the discriminator
  - We train the GAN (so Noise' as input to the generator -> output Xfake (size M'xT) which will be input to the discriminator)
  - After looping on each batch, we made an epoch, and as surprising as it may seem, we consider that it is sufficient here.

## Conclusion/Tests  
- For 3 input vectors, the discriminator predictions are 0.5, but we see that the series obtained are almost identical. 
- In fact, when testing with an input vector (0,...,0) of size D, the output series is also exactly the same. 
- The generator has therefore only learned from the bias.
- It is thus important to check with a zero input vector, and to **have the argument use_bias=False for the generator**.
- After reconstruction of the model without the bias, the time series are less similar and more respectful of the stylized facts.

## Next step 
- Make the architecture of the generator and discriminator much more robust in order to have time series that are more different from each other.
