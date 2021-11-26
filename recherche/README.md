# Recherche pour coller un squelette à un personnage


### Installation dans un venv

``` bash
# Mise à jour de pip
sudo apt install python3-pip
python3 -m pip install --upgrade pip
# Installation de venv
sudo apt install python3-venv

# Installation de l'environnement
cd /le/dossier/de/personnages3d/recherche
# Création du dossier environnement si pas encore créé
python3 -m venv mon_env
# Activation
source mon_env/bin/activate
# Upgrade pip if needed
python3 -m pip install --upgrade pip
# Installation des packages
python3 -m pip install -r requirements.txt
```

Ca installe un lien vers python du système, numpy et opencv dans le dossier mon_env de recherche, avec les dépendances.

### Recherche

Vous ne devez modifier que le fichier personnages3d_test_play.py
La fenêtre OpenCV qui s'ouvre est la vue de dessus, avec les personnes captées représentée par un cercle de couleur.

#### Exécution du script

Dans le dossier:
``` bash
cd /le/dossier/de/personnages3d/recherche

./mon_env/bin/python3 personnages3d_test_play.py
```
Ce script doit fonctionner, si ce n'est pas le cas, abandonnez.

#### Enoncé du problème

La détection des squelettes des personnes dans l'image de la caméra se traduit par une liste de squelette, dans un ordre au hazard. Cette liste est rarement dans le même ordre.

Par exemple, si 3 personnes captées, il peut y avoir une liste de 0, 1, 2, 3 voire 4 suelettes (des faux faux), cette liste est le résultat d'un calcul tensoriel. N'ayez pas peur des mots compliqués, je les place pour épater le quidam !

Objectif:
Appliquer un squelette à la même personne, cela se traduit par une couleur constante dans la vue de dessus. Par exemple, la personne rouge doit toujours rester rouge pendant ses déplacements.

### Quelques explications sur le script

* Utilise le json défini dans if __name__ == '__main__':
* La tempo de défilement des frames est réglable dans run() avec cv2.waitKey(tempo) tempo en ms
* Problèmes connus:
    * si le nombre de squelettes détectés est supérieur à 4, c'est non testé !
* Lorsque les squelettes se croisent, lorsque quelqu'un sort du champ, ça marche mal.

### Des idées en l'air
* Plutot que de créer un historique, il faudrait deviner la suite des points probable
* Faire un recherche avec une distance de plus en plus grande.
