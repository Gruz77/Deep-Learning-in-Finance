# Deep-Learning-in-Finance

<img src="img/opening.jpeg" width="500"> 

## Exposant de Hurst : H

L'exposant de Hurst est directement lié à notion de mouvement fractionnaire, c'est un indicateur de longue mémoire des séries temporelles. 
- Pour 0 < H < 0.5, nous sommes dans une trend anti-persistente, le principe de mean-reverting, 
- Pour H = 0.5, nous avons un mouvement brownien standard (processus de Wiener),
- Pour 0.5 < H < 1, nous avons une trend persistente (autocorrélation positive à long-terme).

Buts du projet :
- Implémentation du CNN de l'[article de H.Stone, QF (2020)](https://arxiv.org/pdf/1812.05315v3.pdf) pour l'estimation de l'exposant
- But : trouver une architecture robuste d'un "simple" réseau dense permettant de faire mieux que le CNN précédent pour l'estimation de valeurs de H < 0.5, d'un point de vue MBE, RMSE et MAE.
- Training Set (CNN et ANN) : 
  - Pour 10 valeurs de H ([0,1]), génération de 10 000 séries temporelles de mouvement browninen fractionnaire de longueur T = 100 (librairie *fBm*)
- Test Set :
  - Pour 100 valeurs de H ([0.01,0.99]), génération de 1000 séries temporelles de longueur T = 100
- Conclusion : le réseau dense ANN est meilleur que le CNN pour l'estimation de H < 0.4. Cela est suffisant pour sauvegarder le modèle ANN pour une utilisation ultérieure (recherche récente démontrant que le coefficient de Hurst pour les séries financières est aux alentours de 0.15).
- Next step : comparer les résultats avec les estimateurs Wavelets via le package *liftLRD* de R (*rpy2* permet d'importer des fonctions R en Python)

## GANs : Génération de séries temporelles

Les applications des GANs sont autant diverses qu'[impressionantes](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/). 
Ici nous nous intéressons à la génération de séries temporelles. Ce qui peut être extrèmement utile dans le cas de backtesting de stratégie, pour éviter l'overfitting, cela permettrait d'accéder à un univers presque infini de possibilités, et ainsi d'avoir des stratégies dont la significativité pourrait être d'autant plus parlante. ("Train on fake, trade on real").
