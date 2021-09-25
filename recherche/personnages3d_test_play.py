

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

"""

FICHIER = './json/cap_2021_09_24_11_33.json'

import os
import json
from time import time, sleep

import numpy as np
import cv2

from my_config import MyConfig


COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]



class Personnage:
    """Permet de stocker facilement les attributs d'un personnage,
    et de les reset-er.
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.len_histo = int(self.config['pose']['len_histo'])
        self.reset()

    def reset(self):
        self.who = None
        self.xys = None
        self.points_3D = None
        self.center = [100000]*3

        # 10x et 10y et 10z soit 1 seconde
        self.historic = [0]*3

        self.historic[0] = [0]*self.len_histo
        self.historic[1] = [0]*self.len_histo
        self.historic[2] = [0]*self.len_histo
        self.stability = 0

    def add_historic(self, centre):
        """Ajout dans la pile, suppr du premier"""

        for i in range(3):
            self.historic[i].append(centre[i])
            del self.historic[i][0]



class Personnages3D:

    def __init__(self, **kwargs):
        """Les paramètres sont définis dans le fichier personnages3d.ini"""


        self.json_data = read_json(FICHIER)

        self.config = kwargs
        print(f"Configuration:\n{self.config}\n\n")

        # Distance de rémanence pour attribution des squelettes
        self.distance = float(self.config['pose']['distance'])


        # Nombre deif k == 32:  # space personnes à capter
        self.person_nbr = min(int(self.config['pose']['person_nbr']), 4)

        self.whos = [0]*self.person_nbr

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WND_PROP_FULLSCREEN)
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")


        # Toutes les datas des personnages dans un dict self.personnages
        self.personnages = []
        for i in range(self.person_nbr):
            self.personnages.append(Personnage(**self.config))
        self.skelet_nbr = 0
        self.new_centers = None

        self.loop = 1

    def main_frame_test(self, persos_2D, persos_3D):

        # Récup de who, apply to self.perso
        if persos_3D:
            # Ceci n'est pas une bonne idée, les bons pourrait etre dans les supprimés
            self.skelet_nbr = min(len(persos_3D), 4)

            self.who_is_who(persos_3D)
            self.apply_to_personnages(persos_2D, persos_3D)
            self.draw_all_personnages()

    def update_centers(self, persos_3D):
        """
        last_centers = liste des centres tirée de l'historique des piles de centre
        self.new_centers = liste des centres des squelettes de la frame
        """

        self.last_centers = []
        for i in range(self.person_nbr):
            self.last_centers.append(self.personnages[i].center)

        self.new_centers = []
        for i in range(self.skelet_nbr):
            self.new_centers.append(get_center(persos_3D[i]))

    def who_is_who(self, persos_3D):

        # Préliminaire
        self.update_centers(persos_3D)
        print("\nNombre de squelette  ------------------------>", self.skelet_nbr)

        # Parcours des squelettes pour calculer les distances par rapport
        # aux centres des personnages
        # dists[0] = liste des distance entre: squelette 0 et les personnages
        dists = {}
        for skel in range(self.skelet_nbr):
            dists[skel] = []
            # Recherche des perso proche de ce skelelet
            for perso in range(self.person_nbr):
                dist = get_distance(self.new_centers[skel], self.last_centers[perso])
                if dist > 100000:
                    dist = 100000
                dists[skel].append(dist)

        print("distances:", dists)  # {0: [41, 41, 41, 41]}

        # Attibution avec le perso le plus proche du squelette
        whos, TODO = self.attribution_with_nearest(dists)
        self.whos = self.default_attribution(whos, TODO)

    def default_attribution(self, whos, TODO):
        """ Attribution par défaut si pas attribué avant

        whos: [1, None, None, None] TODO: 2

        objectif --> [1, 0, 2, None]

        liste des déjà attribués: done = [1]

        à attribuer 0 et 2:
            possible = [0, 2, 3]
            moins whos
            liste des numéros à attribuer: dispo = [0, 2]

        len(dispo) = TODO

        """

        done = [x for x in whos if x is not None]
        dispo = [x for x in range(self.person_nbr) if x not in whos]

        print("whos avec nearest:", whos, "TODO:", TODO, "done:", done, "dispo", dispo)

        # Attribution importante
        d = 0
        while TODO > 0:
            for i, who in enumerate(whos):
                if who is None:
                    whos[i] = dispo[d]
                    TODO -= 1
                    d += 1
                    break

        print("whos final:", whos)
        return whos

    def attribution_with_nearest(self, dists):
        """ Attribution avec le plus près
        Nombre de squelette  ------------------------> 2
        distances: {0: [2, 1091, 1557, 100000], 1: [1092, 3, 1415, 100000]}
        whos: [0, 1, None, None] TODO: 0
        whos final [0, 1, None, None]

        Nombre de squelette  ------------------------> 2
        distances: {0: [1091, 2, 1413, 100000], 1: [3, 1096, 1556, 100000]}
        whos: [1, 0, None, None] TODO: 0
        whos final [1, 0, None, None]
        """
        whos = [None]*self.person_nbr
        gaps = []
        # Nombre de squelette qui reste à attribuer
        TODO = self.skelet_nbr
        for i in range(self.skelet_nbr):
            if i in dists:
                # Le mini dans la liste
                mini = min(dists[i])
                # Position du mini dans la liste
                index = dists[i].index(mini)
                if mini < self.distance:
                    gaps.append(mini)
                    whos[index] = i
                    TODO -= 1
        # Ne sert que pour l'affichage
        try:
            for i in range(len(gaps)):
                self.personnages[whos[i]].gap = gaps[i]
        except:
            pass

        return whos, TODO

    def apply_to_personnages(self, persos_2D, persos_3D):
        """ whos du type [1, 0, None, 2]
                                1 attribué au perso 0
                                0 attribué au perso 1 ... etc ...
        """

        for i in range(self.person_nbr):

            # Data valide
            if self.whos[i] is not None:
                self.personnages[i].who = self.whos[i]
                self.personnages[i].xys = persos_2D[self.whos[i]]
                self.personnages[i].points_3D = persos_3D[self.whos[i]]
                c = get_center(persos_3D[self.whos[i]])
                self.personnages[i].center = c
                self.personnages[i].add_historic(c)

            # Pas de data sur cette frame
            else:
                self.personnages[i].who = None
                self.personnages[i].xys = None
                self.personnages[i].points_3D = None
                self.personnages[i].add_historic(self.personnages[i].center)

    def draw_all_personnages(self):

        self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        cv2.line(self.black, (0, 360), (1280, 360), (255, 255, 255), 2)
        for i, perso in enumerate(self.personnages):
            if perso.center and perso.center[0] and perso.center[2]:

                x = 360 + int(perso.center[0]*160/1000)
                if x < 0: x = 0
                if x > 1280: x = 1280
                y = int(perso.center[2]*160/1000)
                if y < 0: y = 0
                if y > 720: y = 720
                self.draw_personnage(y, x, COLORS[i])

    def draw_personnage(self, x, y, color):
        cv2.circle(self.black, (x, y), 10, (100, 100, 100), -1)
        cv2.circle(self.black, (x, y), 12, color=color, thickness=2)

    def run(self, conn):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        t0 = time()
        nbr = 0

        while self.loop:

            if nbr < len(self.json_data):
                persos_2D = self.json_data[nbr][0]
                persos_3D = self.json_data[nbr][1]
                self.main_frame_test(persos_2D, persos_3D)
            else:
                os._exit(0)

            cv2.imshow('vue du dessus', self.black)
            nbr += 1

            k = cv2.waitKey(100)
            if k == 27:  # Esc
                break

        # Du OpenCV propre
        cv2.destroyAllWindows()



def read_json(fichier):
    try:
        with open(fichier) as f:
            data = json.load(f)
    except:
        data = None
        print("Fichier inexistant ou impossible à lire:")
    return data


def get_distance(p1, p2):
    """Distance entre les points p1 et p2, dans le plan horizontal,
    sans prendre en compte le y qui est la verticale.
    """

    if p1 and p2:
        if None not in p1 and None not in p2:
            d = ((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)**0.5
            return int(d)
    return 100000


def get_center(points_3D):
    """Le centre est le centre de vue du dessus,
        la verticale (donc le y) n'est pas prise en compte.
    """

    center = []
    if points_3D:
        for i in range(3):
            center.append(get_moyenne(points_3D, i))

    return center


def get_moyenne(points_3D, indice):
    """Calcul la moyenne d'une coordonnée des points,
    la profondeur est le 3 ème = z, le y est la verticale
    indice = 0 pour x, 1 pour y, 2 pour z
    """

    somme = 0
    n = 0
    for i in range(17):
        if points_3D[i]:
            n += 1
            somme += points_3D[i][indice]
    if n != 0:
        moyenne = int(somme/n)
    else:
        moyenne = None

    return moyenne


def main():

    ini_file = 'personnages3d.ini'
    config_obj = MyConfig(ini_file)
    config = config_obj.conf

    # Création de l'objet
    p3d = Personnages3D(**config)

    # On tourne, silence, caméra, action !!!
    conn = None
    p3d.run(conn)



if __name__ == '__main__':
    """Excécution de ce script en standalone"""

    main()
