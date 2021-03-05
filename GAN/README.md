## GANs : Génération de séries temporelles

Les applications des GANs sont autant diverses qu'[impressionantes](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/). 
Ici nous nous intéressons à la génération de séries temporelles. Ce qui peut être extrèmement utile dans le cas de backtesting de stratégie, pour éviter l'overfitting. Cela permettrait d'accéder à un univers presque infini de possibilités, et ainsi d'avoir des stratégies dont la significativité pourrait être d'autant plus parlante. ("Train on fake, trade on real"). Une chose très importante pour vérifier que le GAN a bien généré des séries financières valides est d'en vérifier les faits stylisés (queues grasses, volatilité avec longue mémoire, ... [voir ici](https://github.com/Gruz77/Physics-of-Markets/tree/main/Stylized_Facts))

- On utilise les données journalières de l'indice S&P500 depuis 1928. Nous travaillons comme d'habitude sur les log-rendements.
- On construit le GAN : 
  - Le générateur aura une entrée de dimension D = 10 et une sortie de dimension T = 200
  - Le discriminateur a une entrée de taille T (sortie du générateur) et une seule sortie d'activation sigmoïde : afin de discriminer (série financière ou non)
  - Notre objet GAN aura donc en entrée celle du générateur, et en sortie celle du discriminateur. Le but est que le discrimnateur ne sache plus différencier les vrais séries temporelles des fausses, et donc renvoie une probabilité de 0.5 une fois entrainé, pour chaque vecteur fourni en entrée.

- Entrainement du GAN pour chaque batch (de taille 32 ici) :
  - On génère M sous-échantillons de taille T de nos log-rendements (Xreal, taille MxT)
  - On génère M échantillons de vecteurs de bruit de dimension D (Noise, taille MxD)
  - Avec ces échantillons de bruit, on utilise le générateur pour prédire M vecteurs de rendements (Xgen, taille MxT), que l'on concatene en lignes à Xreal pour avoir Xrealgen (taille 2MxT)
  - On définit le vecteur Yrealgen = (1,...,1,0,...,0) (taille 2M), que l'on shuffle afin que le réseau n'apprenne pas de l'ordre des lignes
  - On entraine le discriminateur
  - On créé notre matrice de bruit pour tromper le discriminateur (Noise', taille M'xD, avec M'=M ici)
  - On définit notre vecteur Yfake = (1,...,1) (taille M') prétendant que les échantillons de bruit ci-dessus sont de vraies séries temporelles -> tromper le discriminateur
  - On entraine le GAN (donc Noise' en entrée du générateur -> sortie Xfake (taille M'xT) qui sera en entrée du discriminateur)
  - Après avoir bouclé sur chaque batch, nous avons fait une epoch, et aussi surprenant que cela puisse paraitre, on considère que c'est suffisant ici.

- Conclusion/Tests : 
  - Pour 3 vecteurs d'entrée, les predictions du discriminateur sont de 0.5, mais on voit que les séries obtenues sont presques identiques. 
  - En fait, en testant avec un vecteur d'entrée (0,...,0) de taille D, la série en sortie est aussi exactement la même. 
  - Le générateur n'a donc appris que du biais.
  - Il est ainsi important de vérifier avec un vecteur d'entrée nul, et d'**avoir l'argument use_bias=False pour le générateur**.
  - Après reconstruction du modèle sans le biais, les séries temporelles sont moins similaires et respectent plus les faits stylisés.

- Next step :
  - Rendre l'architecture du générateur et discriminateur bien plus robuste afin d'avoir des séries temporelles d'autant plus différentes l'une de l'autre.
