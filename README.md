# Deep-Learning-in-Finance

<img src="img/opening.jpeg" width="500"> 

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
- Conclusion : le réseau dense ANN est meilleur que le CNN pour l'estimation de H < 0.4. Cela est suffisant pour sauvegarder le modèle ANN pour une utilisation ultérieure (recherche récente démontrant que le coefficient de Hurst pour les séries financières est aux alentours de 0.1 : [Volatility is rough, J.Gatheral, M.Rosenbaum](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1393551)).
- Next step : comparer les résultats avec les estimateurs Wavelets via le package *liftLRD* de R (*rpy2* permet d'importer des fonctions R en Python)

## GANs : Génération de séries temporelles

Les applications des GANs sont autant diverses qu'[impressionantes](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/). 
Ici nous nous intéressons à la génération de séries temporelles. Ce qui peut être extrèmement utile dans le cas de backtesting de stratégie, pour éviter l'overfitting. Cela permettrait d'accéder à un univers presque infini de possibilités, et ainsi d'avoir des stratégies dont la significativité pourrait être d'autant plus parlante. ("Train on fake, trade on real"). Une chose très importante pour vérifier que le GAN a bien générer des séries financières valides est d'en vérifier les faits stylisés (queues grasses, volatilité avec longue mémoire, ... [voir ici]())

- On utilise les données journalières de l'indice S&P500 depuis 1928. Nous travaillons comme d'habitude sur les log-rendements.
- On construit le GAN : 
  - Le générateur aura une entrée de dimension D = 10 et une sortie de dimension T = 200
  - Le discriminateur a une entrée de taille T (sortie du générateur) et une seule sortie d'activation sigmoïde : afin de discriminer (série financière ou non)
  - Notre objet GAN aura donc en entrée de taille D correspondant à celle du générateur, et en sortie celle du discriminateur. Le but est que le discrimnateur ne sache plus différencier les vrais séries temporelles des fausses, et donc renvoie une probabilité de 0.5 une fois entrainé, pour chaque vecteur fourni en entrée.

- Entrainement du GAN pour chaque batch (de taille 32 ici) :
  - On génère M sous-échantillons de taille T de nos log-rendements (Xreal, taille MxT)
  - On génère M échantillons de vecteurs de bruit de dimension D (Noise, taille MxD)
  - Avec ces échantillons de bruit, on utilise le générateur pour prédire M vecteurs de rendements (Xgen, taille MxT), que l'on concatene en lignes à Xreal pour avoir Xrealgen (taille 2MxT)
  - On définit le vecteur Yrealgen = (1,...,1,0,...,0) (taille 2M), que l'on shuffle afin que le réseau n'apprenne pas de l'ordre des lignes
  - On entraine le discriminateur
  - On créé notre matrice de bruit pour tromper le discriminateur (Noise', taille M'xD, avec M'=M ici)
  - On définit notre vecteur Yfake = (1,...,1) (taille M') prétendant que les échantillons de bruit ci-dessus sont de vraies séries temporelles
  - On entraine le GAN (donc Noise' en entrée du générateur -> sortie Xfake (taille M'xT) qui sera en entrée du discriminateur)
  - Après avoir bouclé sur chaque batch, nous avons fait une epoch, et aussi surprenant que cela puisse paraitre, on cosidère que c'est suffisant ici.

- Conclusion : Pour 3 vecteurs d'entrée, les predictions du discriminateur sont de 0.5, mais on voit que les séries obtenues sont presques identiques. En fait, en testant avec un vecteur d'entrée (0,...,0) de taille D, la série en sortie est aussi exactement la même. Le générateur n'a donc appris que du biais. 
