# Prediction de prix d'options

- Buts du projet :
  - Notre but ici est de prédire le prix de market data d'options (mid_option réechelonné par le mid du sous-jacent) à l'aide du Deep Learning
  - Et de voir l'augmentation de la prédiction en fonction des prédicteurs ajoutés.
  - On note que l'on dispose de puts et de calls, mais nous nous intéressons au cas du call ici. 
    - Pour le put, cela est totalement similaire. Si nous voulons traiter les deux en même temps, il faut rajouter des dummy variables binaires permettant de préciser au réseau si l'échantillon est un call ou un put.

- Data : 
  - données hebdomadaires (environ 400 par semaine) de début 2009 à fin 2014 contenant :
    - mid (du sous-jacent), 
    - bid (de l'option),
    - ask (de l'option), 
    - strike,
    - type d'option (call ou put),
    - date d'expirations,
    - Jours jusqu'a expiration. 

  - Nous retraitons les données afin d'ajouter les colonnes suivantes, la plupart réechelonnées par le mid du sous-jacent, afin d'avoir des valeurs relatives par rapport au prix du sous-jacent :
    - mid_option (bid+ask/2),
    - mid_option réechelonné (mid_option/mid),
    - spread réechelonné ((ask-bid)/mid),
    - k : strike réechelonné (strike/mid),
    - T_Scaled : Jours jusqu'a expiration en fraction d'année (days_to_expiry/365),
    - volatilité mesurée sur 7, 14 et 49 semaines précedentes (calculée sur les log rendements hebdomadaires)
    - Prévision des prix par la formule de Black & Scholes (en multipliant la vol calculée par sqrt(52) pour cohérence, et en prenant r = 0)

- On a rajouté le prix prédit par Black & Scholes car on part du principe qu'aider le Deep Learning par des modèles financiers connus peut être une bonne initiative.

- Nous avons au total un peu plus de 100 000 échantillons, nous décidons de splitter (avec shuffle) en 80% train et 20% test. (Le validation set se fera automatiquement avec keras pendant l'entrainement).

- On ne se place que dans le cas du call dans ce TP. Le cas du put est totalement similaire. Si on veut traiter les deux en même temps, il faut rajouter deux colonnes de dummy variables binaires pour spécifier le type d'option au réseau.

## DNN

- Architecture:
  - 3 hidden layers de 150, 100, 50 neurones chacunes avec fonction d'activation relu pour chaque
  - 1 couche de sortie avec activation relu 
  - optimiseur adam et loss MSE.

## Différentes features

### DNN1 - k + T_Scaled
- On utilise en premier lieur seulement le strike réechelonné k et le nombre de jours jusqu'a expiration
- MAE = 8.0e-5 | RMSE = 5.89e-3
- Nous voyons qu'avec seulement le strike et les jours jusqu'a expiration, le réseau prédit déja très bien 

### DNN2 - k + T_scaled + spread bd/ask relatif
- On ajoute le spread bid/ask relatif et on essaie de voir si cela améliore notre prédiction
- MAE = 6.6e-5 | RMSE = 5.49e-3
- Nous voyons une légère amélioration, mais pas non plus exceptionnelle.
  
### DNN3 - k + T_scaled + spread bid/ask relatif + vol (49 semaines)
- On ajoute la vol et on essaie de voir si cela améliore notre prédiction
- MAE = 2.3e-5 | RMSE = 3.49e-3
- Cette fois l'amélioration est presque d'un facteur 2 !

### DNN4 - k + T_scaled + spread bid/ask relatif + vol (49 semaines) + Prix par Black&Scholes
- On ajoute le prix prédit par Black&Scholes afin de voir si cela améliore la prédiction, et de valider le fait qu'aider le Deep Learning avec des modèles classiques financiers est une bonne initiative.
- MAE = 1.9e-5 | RMSE = 2.83e-3
- La précision est encore améliorée et nous vérifions l'hypothèse d'aider le Deep Learning via des modèles classiques. 
