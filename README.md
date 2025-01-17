# Rapport TP3 - Régression linéaire

## Auteur

- Nom: Dominique Elias
- Code permanent: ELID14019800

## Implémentation fit() avec CUDA

j’ai réalisé toutes les allocations et les appels au kernel à l’intérieur de la fonction fit, sans utiliser une fonction auxiliaire (comme reduce) pour allouer et désallouer la mémoire CUDA, car les deux vecteurs x et y seront utilisés pour plusieurs étapes de calcul.

## Tests

Quelques tests ont été ajoutés pour vérifier le bon fonctionnement de la fonction fit() avec de grands nombres de points.

## Performances (sur la grappe)

On remarque que la version CUDA est plus lente que la version séquentielle sur CPU. Cela est dû principalement au fait que le temps d’allocation et de désallocation de la mémoire CUDA est plus grand que le temps d’exécution du calcul de la régression linéaire.

Temps d'exécution pour 100 000 000 points:

- CPU: 0.308041 s
- GPU: 0.954217 s

```sh
[skimondo@login1 linear-regression-cuda]$ srun -G 1 --mem=16G  build-release/bin/fitab -n 100000000 -p 0
Options used:
   --parallel 0
   --num-points 100000000
   --output 
régression linéaire... terminé!
temps d'exécution: 0.308041 s
vitesse          : 3.24632e+11 pps
a:    10
b:    2
r:    1
xmean:0.5
ymean:11
Fin normale du programme
```

```sh
[skimondo@login1 linear-regression-cuda]$ srun -G 1 --mem=16G  build-release/bin/fitab -n 100000000 -p 1
Options used:
   --parallel 1
   --num-points 100000000
   --output 
régression linéaire... terminé!
temps d'exécution: 0.954217 s
vitesse          : 1.04798e+11 pps
a:    10
b:    2
r:    1
xmean:0.5
ymean:11
Fin normale du programme
```
