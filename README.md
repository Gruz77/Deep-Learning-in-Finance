# Deep-Learning-in-Finance

<img src="img/opening.jpeg" width="500"> 

## Exposant de Hurst : H

- Implémentation du CNN de l'[article de H.Stone, QF (2020)](https://arxiv.org/pdf/1812.05315v3.pdf) pour l'estimation de l'exposant
- But : trouver une architecture robuste d'un "simple" réseau dense permettant de faire mieux que le CNN précédent pour l'estimation de valeurs de H < 0.5, d'un point de vue MBE, RMSE et MAE.
- Training Set (CNN et ANN) : 
  - Pour 10 valeurs de H ([0,1]), génération de 10 000 séries temporelles de mouvement browninen fractionnaire de longueur T = 100 (librairie *fBm*)
- Test Set :
  - Pour 100 valeurs de H ([0.01,0.99]), génération de 1000 séries temporelles de longueur T = 100
- Conclusion : le réseau dense ANN est meilleur que le CNN pour l'estimation de H < 0.4. Cela est suffisant pour sauvegarder le modèle ANN pour une utilisation ultérieur (recherche récente démontrant que le coefficient de Hurst pour les séries financières est aux alentours de 0.15).
