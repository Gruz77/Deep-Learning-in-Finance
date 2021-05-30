<h1 align='center'> Call & Put Prediction </h1>

[<h1 align='center'>![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gruz77/Deep-Learning-in-Finance/blob/main/Call_Put_Prediction/Call_Put_prediction.ipynb)</h1>

- Project goals:
  - Our goal here is to predict the market data price of options (mid_option rescaled by the mid of the underlying) using Deep Learning
  - And to see the increase of the prediction according to the added predictors.
  - Note that we have puts and calls, but we are interested in the call case here. 
    - For the put, it is totally similar. If we want to process both at the same time, we need to add binary dummy variables to tell the network whether the sample is a call or a put.

- Data : 
  - Weekly data (about 400 per week) from the beginning of 2009 to the end of 2014 containing:
    - mid (of the underlying), 
    - bid (of the option),
    - ask (of the option), 
    - strike,
    - type of option (call or put),
    - expiry date,
    - Days to expiry. 

  - We remove the data to add the following columns, mostly rescaled by the mid of the underlying, to have relative values to the price of the underlying:
    - mid_option (bid+ask/2),
    - rescaled mid_option (mid_option/mid),
    - rescheduled spread ((ask-bid)/mid),
    - k: rescheduled strike (strike/mid),
    - T_Scaled: Days to expiration in fraction of a year (days_to_expiry/365),
    - volatility measured over 7, 14 and 49 previous weeks (calculated on weekly log returns)
    - Price prediction by Black & Scholes formula (multiplying the calculated flight by sqrt(52) for consistency, and taking r = 0)

- We added the price predicted by Black & Scholes because we assume that helping Deep Learning with known financial models can be a good initiative.

- We have a total of more than 100 000 samples, we decide to split (with shuffle) in 80% train and 20% test. (The validation set will be done automatically with keras during training).

- We will only consider the call case in this tutorial. The case of the put is totally similar. If we want to treat both at the same time, we must add two columns of dummy binary variables to specify the type of option to the network.

## DNN

- Architecture:
  - 3 hidden layers of 150, 100, 50 neurons each with read-back activation function for each
  - 1 output layer with reread activation (as the option price is always positive)
  - adam optimizer and MSE loss.

## Different features

### DNN1: k + T_Scaled
- We use in the first place only the rescaled strike k and the number of days until expiration
- MAE = 8.0e-5 | RMSE = 5.89e-3
- We see that with only the strike and the days to expiry, the network already predicts very well 

### DNN2 : k + T_scaled + relative bd/ask spread
- We add the relative bid/ask spread and try to see if it improves our prediction
- MAE = 6.6e-5 | RMSE = 5.49e-3
- We see a slight improvement, but not exceptional either.
  
### DNN3: k + T_scaled + relative bid/ask spread + flight (49 weeks)
- We add the flight and try to see if it improves our prediction
- MAE = 2.3e-5 | RMSE = 3.49e-3
- This time the improvement is almost a factor of 2!

### DNN4 : k + T_scaled + relative bid/ask spread + flight (49 weeks) + Price by Black&Scholes
- We add the price predicted by Black&Scholes to see if this improves the prediction, and to validate the fact that helping Deep Learning with classical financial models is a good initiative.
- MAE = 1.9e-5 | RMSE = 2.83e-3
- Accuracy is further improved and we verify the hypothesis of supporting Deep Learning with classical models. 
  
