# INF5171-243-TP3 - Régression linéaire avec CUDA

Une régression linéaire consiste à déterminer les coefficients d'un polynôme de degré 1 qui approxime la tendance d'un nuage de point.

Le programme `fitab` fourni calcule sur le CPU la régression linéaire. Votre but est de réaliser l'implémentation parallèle avec CUDA. Il s'agit essentiellement d'une réduction. L'algorithme est dans la classe `FitSerial`. Les options du programme sont les suivantes:

* -p : 0=série, cuda=1
* -n : nombre de points à générer
* -o : écrire les points dans un fichier

Exemple d'exécution

```
Options used:
   --num-points 100000000
   --method 0
   --output 
régression linéaire... terminé!
temps d'exécution: 0.772772 s
vitesse          : 1.29404e+11 pps
a:    10
b:    2
r:    1
xmean:0.5
ymean:11
Fin normale du programme
```

### Étape 1 : Réalisation

Vous devez implémenter un noyau de calcul CUDA pour réaliser l'algorithme.

* Votre code doit être dans la méthode fit()
* Vous pouvez ajouter les membres de la classe FitCuda comme vous le souhaitez
* Vous devez copier les données vers la carte, faire l'appel à votre kernel, puis copier le résultat sur l'hôte.
* Vous pouvez vous inspirer du kernel de réduction, qui se base sur la mémoire partagée et la technique de `Thread Coarsening`

### Étape 2 : Tests et validation

Les tests sont fournis. Vous pouvez écrire de nouveaux tests, mais c'est facultatif. Lancez le programme `test_cuda` pour vérifier que votre implémentation donne le même résultat que la version série.

Le test génère des points à partir d'une droite connue. Il est attendu que la régression linéaire retrouve les coefficients à l'origine de la génération des points.

Sur la grappe, exécutez votre version série, et la version parallèle avec CUDA pour 100 millions de points. Comparez le temps d'exécution de la version parallèle sur GPU à la version série sur CPU.

### Remise

 * Faire une archive du code de votre solution avec la commande `make remise` ou `ninja remise`
 * Remise sur TEAMS (code seulement), une seule remise par équipe.
 * Identifiez le ou les codes permanents dans le fichier source `fitcuda.cpp`
 * Respect du style: pénalité max 10% (guide de style dans le fichier `.clang-format`)
 * Qualité (fuite mémoire, gestion d'erreur, avertissements, etc): pénalité max 10%
 * Doit s'exécuter correctement sur la grappe
 * Total sur 100

Bon travail !

# Note sur les logiciels externes

Le code intègre les librairies et les programmes suivants.

* https://github.com/catchorg/Catch2
