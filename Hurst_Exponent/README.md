## Exposant de Hurst : H

L'exposant de Hurst est directement lié à notion de mouvement fractionnaire, c'est un indicateur de longue mémoire des séries temporelles. 
- Pour 0 < H < 0.5, nous sommes dans une trend anti-persistente, le principe de mean-reverting, 
- Pour H = 0.5, nous avons un mouvement brownien standard (processus de Wiener),
- Pour 0.5 < H < 1, nous avons une trend persistente (autocorrélation positive à long-terme).

Buts du projet :
- Implémentation du CNN de l'[article de H.Stone, QF (2020)](https://arxiv.org/pdf/1812.05315v3.pdf) pour l'estimation de H
- But : trouver une architecture robuste d'un "simple" réseau dense permettant de faire mieux que le CNN précédent pour l'estimation de valeurs de H < 0.5, d'un point de vue MBE, RMSE et MAE.
- Training Set (CNN et ANN) : 
  - Pour 10 valeurs de H ([0,1]), génération de 10 000 séries temporelles de mouvement browninen fractionnaire de longueur T = 100 (librairie *fBm*)
- Test Set :
  - Pour 100 valeurs de H ([0.01,0.99]), génération de 1000 séries temporelles de longueur T = 100
- Conclusion : le réseau dense ANN est légèrement meilleur que le CNN pour l'estimation de H < 0.4. Cela est suffisant pour sauvegarder le modèle ANN pour une utilisation ultérieure (recherche récente démontrant que le coefficient de Hurst pour les séries financières est aux alentours de 0.1 : [Volatility is rough, J.Gatheral, M.Rosenbaum](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1393551)).
- Next step : comparer les résultats avec les estimateurs Wavelets via le package *liftLRD* de R (*rpy2* permet d'importer des fonctions R en Python) - Relancer les calculs en local (ram google colab trop faible)