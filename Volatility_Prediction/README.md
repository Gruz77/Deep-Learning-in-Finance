# LSTMs : Prédiction de la volatilité 

Le but ici est d'utiliser un LSTM afin de prédire la volatilité avec différents prédicteurs, et de les comparer en terme de RMSE et MAE.
Nous allons voir si les modèles de Deep Learning peuvent être "aidés" par des modèles financiers actuels. 

Data : fichier .parquet de séries de variances réalisées de l'Oxford-Man Institute pour différents actifs, de 2000 à 2021.

- On note que dans tout le projet nous estimons la variance, puis nous passons à la volatilité annualisée pour effectuer les comparaisons.

## Modèle rough
- Nous allons reprendre le code du lien suivant sur la Rough Volatility : [rough_volatility_with_python](https://tpq.io/p/rough_volatility_with_python.html)

## Deep Learning
  - L'entrainement et la prédiction sont faits sur fenêtre glissantes
  - On utilise un LSTM avec une couche de 100 neurones et en stateful (garde l'état caché pour chaque batch) avec seulement la variance réalisée en input :
  - 100 timesteps pour notre matrice de prédicteurs,
  - batch_size de 64 et chaque fenêtre d'entrainement représente T_in = 20*64 (1280) données.
  - On entraine le LSTM avec un validation_set de 20% sur les T_in données, avec 20 epochs,
  - On prédit les 100 valeurs suivantes, avec un nouvel objet LSTM dont on définit les poids (avec set_weights) comme ceux du modèle tout juste entrainé (cela vient du fait que notre batch_size de prédiction (100) est différent de celui d'entrainement (64))
  - Process :
    - La première itération d'entrainement sera sur les 1280 premières données, 
    - On prédit les 100 suivantes (1280-1380)
    - On entraine de nouveau les données 100-1380, puis on prédit les données 1380-1480, etc...

## Modèle rough + Deep Learning 
  - Ici nous allons procéder de la même manière mais en ajoutant comme feature la prévision du modèle Rough 
  - Puis on étudie une autre possibilitée : utiliser en seconde feature les résidus (différence de la log variance réalisée et de la log variance prédite par le modèle rough)
  - On note que dans ces deux cas les LSTMs ne sont pas entrainés sur le même nombre de données : 
    - les 501 premières valeurs ne sont pas prédites via le modèle rough, et comme nous l'ajoutons en deuxième feature nous devont partir de la 501ème également pour la première feature

## Modèle rough + Deep Learning + Autres prédicteurs
  - Le roll des futures et l'expiration des options a souvent lieu le 3ème vendredi du mois, synonyme de fortes fluctuations/volatilité. Nous allons voir si rajouter le fait que le jour du timestep correspondant est un 3ème Vendredi du mois ou non améliore notre pouvoir prédictif, ou pas.

## Comparaison
  - Le modèle de volatilité rough est un excellent modèle et nous pouvons voir même graphiquement que celui-ci a un pouvoir prédictif dépassant de loin ceux des réseaux LSTMs.
  - Au niveau des LSTMs : 
    Voici une capture d'écran représentant la comparaison : 
    

## Next steps
  - Optimiser les hyperparamètres (avec un GridSearch par exemple)
  
