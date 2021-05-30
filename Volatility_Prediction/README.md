<h1 align='center'> LSTMs : Volatility prediction </h1>

[<h1 align='center'>![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gruz77/Deep-Learning-in-Finance/blob/main/Volatility_Prediction/Volatility_prediction.ipynb)</h1>

The aim here is to use an LSTM to predict volatility with different predictors, and to compare them in terms of RMSE and MAE.
We will see if Deep Learning models can be "helped" by current financial models. 

Data: .parquet file of variance series from the Oxford-Man Institute for different assets, from 2000 to 2021.

- Note that throughout the project we estimate the variance, then switch to annualised volatility to make comparisons.

## Rough model
- We will take the code from the following link on Rough Volatility: [rough_volatility_with_python](https://tpq.io/p/rough_volatility_with_python.html)

## Deep Learning
  - Training and prediction are done on sliding windows
  - We use an LSTM with a layer of 100 neurons and in stateful (keeps the state hidden for each batch) with only the variance realized in input:
  - 100 timesteps for our predictor matrix,
  - batch_size of 64 and each training window represents T_in = 20*64 (1280) data.
  - We train the LSTM with a validation_set of 20% on the T_in data, with 20 epochs,
  - We predict the next 100 values, with a new LSTM object whose weights are defined (with set_weights) as those of the just trained model (this comes from the fact that our prediction batch_size (100) is different from the training one (64))
  - Process:
    - The first training iteration will be on the first 1280 data, 
    - We predict the next 100 (1280-1380)
    - We train again the data 100-1380, then we predict the data 1380-1480, etc...

## Rough model + Deep Learning 
  - Here we will proceed in the same way but add as a feature the prediction of the Rough model 
  - Then we study another possibility: to use as a second feature the residuals (difference between the realized log variance and the log variance predicted by the rough model)
  - We note that in these two cases the LSTMs are not trained on the same number of data: 
    - the first 501 values are not predicted via the rough model, and as we add it as a second feature we have to start from the 501st also for the first feature

## Rough model + Deep Learning + Other predictors
  - The roll of futures and the expiration of options often takes place on the 3rd Friday of the month, synonymous with strong fluctuations/volatility. We will see if adding the fact that the corresponding timestep day is a 3rd Friday of the month or not improves our predictive power, or not.

## Comparison
  - The rough volatility model is an excellent model and we can see even graphically that it has a predictive power far exceeding those of the LSTMs networks.
  - At the LSTM level: 
    Here is a screenshot representing the comparison: 
    <img src="img/comparison_pred_vol.png" width="1000"> 
    
  - Unsurprisingly the rough volatility model is the best model from a prediction point of view. 

  - From a Deep Learning point of view, we can see that : 
    - LSTM + rough model is better than LSTM alone
    - learning residuals instead of the predicted variance does not decrease the error rate
    - adding the third Friday of the month boolean does not add any predictive power.

  - We therefore verify that using deep learning with the help of models is one of the best approaches.


## Next steps
  - Optimise the hyperparameters (with a GridSearch for example)
