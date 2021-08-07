

"""
Echap pour finir proprement le script

Capture de 1 à 4 squelettes
avec
camera Intel RealSense D455, Google posenet et Google Coral.
Envoi de toutes les coordonnées de tous les points en OSC.
"""


import os
from time import time, sleep
import enum

import numpy as np
import cv2
import pyrealsense2 as rs

from posenet.this_posenet import ThisPosenet
from posenet.pose_engine import EDGES

from myconfig import MyConfig
from osc import OscClient
from filtre import moving_average


COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]


class PoseNetConversion:
    """Conversion de posenet vers ma norme
    1 ou 2 squelettes capturés:

    [Pose(keypoints={
    <KeypointType.NOSE: 0>: Keypoint(point=Point(x=652.6, y=176.6),
                                                            score=0.8),
    <KeypointType.LEFT_EYE: 1>: Keypoint(point=Point(x=655.9, y=164.3),
                                                            score=0.9)},
    score=0.53292614),

    Pose(keypoints={
    <KeypointType.NOSE: 0>: Keypoint(point=Point(x=329.2562, y=18.127075),
                                                            score=0.91656697),
    <KeypointType.LEFT_EYE: 1>: Keypoint(point=Point(x=337.1971, y=4.7381477),
                                                            score=0.14472471)},
    score=0.35073516)]

    Conversion en:
    skeleton1 = {0: (x=652.6, y=176.6),
    et
    skeleton2 = {0: (x=329.2, y=18.1), ... etc ... jusque 16
    soit
    skeleton2 = {0: (329.2, 18.1),

    skeletons = list de skeleton = [skeleton1, skeleton2]
    """

    def __init__(self, outputs, threshold):

        self.outputs = outputs
        self.threshold = threshold
        self.skeletons = []
        self.conversion()

    def conversion(self):
        """Convertit les keypoints posenet dans ma norme"""

        self.skeletons = []
        for pose in self.outputs:
            xys = self.get_points_2D(pose)
            self.skeletons.append(xys)

    def get_points_2D(self, pose):
        """ ma norme = dict{index du keypoint: (x, y), }
        xys = {0: (698, 320), 1: (698, 297), 2: (675, 295), .... } """

        xys = {}
        for label, keypoint in pose.keypoints.items():
            if keypoint.score > self.threshold:
                xys[label.value] = [int(keypoint.point[0]),
                                    int(keypoint.point[1])]
        return xys


class Personnage:
    """Permet de stocker facilement les attributs d'un personnage,
    et de les reset-er.
    Utilise une pile avec fitre EMA pour lisser les centres.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.who = None
        self.xys = None
        self.points_3D = None
        self.center = [0.0]*3
        # 30x et 30y et 30z
        self.historic = [0]*3
        self.historic[0] = [0]*30
        self.historic[1] = [0]*30
        self.historic[2] = [0]*30
        self.moving_center = [0]*3
        self.dist = 0

    def add_historic(self, centre):
        """Ajout dans la pile, suppr du premier"""

        for i in range(3):
            if centre[i]:
                self.historic[i].append(centre[i])
                del self.historic[i][0]
        self.EMA_update()

    def EMA_update(self):
        for j in range(3):
            # #print(self.historic[j])
            self.moving_center[j] = moving_average(self.historic[j], 28,
                                    type='simple')  # 'exponentiel' ou 'simple'


class Personnages3D:
    """ Capture avec  Camera RealSense D455
        Détection de la pose avec Coral USB Stick ou CPU (TODO)
        Calcul des coordonnées 3D de chaque personnage et envoi en OSC.
        La profondeur est le 3ème dans les coordonnées d'un point 3D,
        x = horizontale, y = verticale
    """

    def __init__(self):
        """Les paramètres sont à définir dans le fichier personnages3d.ini
        En principe, rien ne doit être modifié dans les autres paramètres.
        """

        ini_file = 'personnages3d.ini'
        self.config_obj = MyConfig(ini_file)
        self.config = self.config_obj.conf
        print(f"Configuration:\n{self.config}\n\n")

        # Seuil de confiance de reconnaissance du squelette
        self.threshold = int(self.config['pose']['threshold'])/100

        # Nombre de pixels autour du point pour moyenne du calcul de profondeur
        self.around = self.config['pose']['around']

        # Distance de rémanence pour attribution des squelettes
        self.distance = int(self.config['pose']['distance'])/100

        # Taille d'image possible: 1280x720, 640x480 seulement
        # 640x480 est utile pour fps > 30
        # Les modèles posenet imposent une taille d'image
        self.width = self.config['camera']['width_input']
        self.height = self.config['camera']['height_input']

        # Plein écran de la fenêtre OpenCV
        self.full_screen = self.config['camera']['full_screen']

        # Le client va utiliser l'ip et port du *.ini
        self.osc = OscClient(**self.config['osc'])

        self.create_trackbar()
        self.set_pipeline()

        self.this_posenet = ThisPosenet(self.width, self.height)

        # Toutes les datas des personnages dans un dict self.personnages
        self.nombre_de_personnage = int(self.config['pose']['nombre_de_personnage'])
        self.personnages = []
        for i in range(self.nombre_de_personnage):
            self.personnages.append(Personnage())
        self.skelet_nbr = 0
        self.centers = None

    def create_trackbar(self):
        """trackbar = slider
        1 sliders: seuil de confiance
        Le message Depracated est un bug !! de la 4.5.1 (ou 4.5.2)
        corrigé dans la prochaine version.
        """

        if self.full_screen:
            cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('color', cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('threshold', 'color', int(self.threshold*100), 100,
                            self.onChange_threshold)

        cv2.createTrackbar('distance_', 'color', int(self.distance*100), 100,
                            self.onChange_distance)

    def onChange_distance(self, value):
        """distance = 0.01 à 1 pour slider de 0 à 100"""

        if value == 0:
            value = 1
        self.config_obj.save_config('pose', 'distance', value)
        value *= 0.01
        self.distance = value

    def onChange_threshold(self, value):
        """threshold = 0.01 à 1 pour slider de 0 à 100"""

        if value == 0:
            value = 1
        self.config_obj.save_config('pose', 'threshold', value)
        value *= 0.01
        self.threshold = value

    def set_pipeline(self):
        """Crée le flux d'image avec la caméra D455"""

        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
        except:
            print(f'Pas de Capteur Realsense connecté')
            os._exit(0)

        device = pipeline_profile.get_device()
        config.enable_stream(rs.stream.color, self.width, self.height,
                                                            rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                                                            rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        unaligned_frames = self.pipeline.wait_for_frames()
        frames = self.align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        self.depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Affichage de la taille des images
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        print(f"Taille des images:"
              f"     {img.shape[1]}x{img.shape[0]}")

    def get_personnages(self, outputs):
        """ Appelé depuis la boucle infinie, c'est le main d'une frame.
                Récupération de tous les squelettes
                Definition de who
        """

        # Récupération de tous les squelettes
        if outputs:
            # les xys
            persos_2D = PoseNetConversion(outputs, self.threshold).skeletons
        else:
            persos_2D = None

        if persos_2D:
            # Ajout de la profondeur pour 3D
            persos_3D = self.get_persos_3D(persos_2D)

            # Récup de who, apply to self.perso
            if persos_3D:
                self.skelet_nbr = len(persos_3D)
                self.centers = self.update_centers(persos_3D)
                whos, dists = self.who_is_who(persos_3D)
                self.apply_to_personnages(whos, persos_2D, persos_3D, dists)

            # Affichage
            self.draw_all_poses()
            self.draw_all_texts()

    def update_centers(self, persos_3D):
        """
        last_centers = liste des centres tirée de l'historique des piles de centre
        self.new_centers = liste des centres des squelettes de la frame
        """

        self.last_centers = []
        for i in range(self.nombre_de_personnage):
            self.last_centers.append(self.personnages[i].moving_center)

        self.new_centers = []
        # 3 squelettes
        for i in range(len(persos_3D)):
            self.new_centers.append(get_center(persos_3D[i]))

    def who_is_who(self, persos_3D):
        """Un personnage de la liste des personnages se voit attribuer les datas
        d'un des squelettes.
        L'ordre des squelettes détectés n'est pas toujours le même d'une frame
        à l'autre.

        Exemple: 4 personnages, 3 squelettes dans persos_3D
            perso1 = squelette_qui
            perso2 = squelette_qui_d_autre ... etc ...

        Si len(persos_3D) = 3, et len(personnages) = 4:
            personnages[0].who = 1, ses points sont persos_3D[1]
            personnages[1].who = 0, ses points sont persos_3D[0]
            personnages[2].who = None, ses points sont None
            personnages[3].who = 2, ses points sont persos_3D[2]
        """

        print("\nDébut de frame")

        self.update_centers(persos_3D)

        # Attribution avec l'historique
        whos, dists = self.attribution_avec_l_historic()

        # Print pour suivi
        self.some_print(whos)

        # Attribution pour les non-attribués et pour initier
        whos = self.attribution_en_dernier_recours_par_ordre(whos)

        print("----------------------------------------------whos final:", whos)
        return whos, dists

    def attribution_avec_l_historic(self):
        """Un EMA a été appliqué sur l'historique des centres"""

        whos = [None]*self.nombre_de_personnage
        dists = [0]*self.nombre_de_personnage
        for i in range(self.nombre_de_personnage):  # 4
            for j in range(self.skelet_nbr):  # 3
                dist = get_distance(self.centers[j], self.last_centers[i])
                if dist and dist < self.distance:
                    whos[i] = j
                    dists[i] = dist
                    print((f"Squelette {j} attribué avec distance = {dist:.2f}"
                           f"pour self.distance = {self.distance}"))
        return whos,  dists

    def some_print(self, whos):
        for i in range(self.nombre_de_personnage):
            a = self.last_centers[i][0]
            b = self.last_centers[i][1]
            c = self.last_centers[i][2]
            if a and b and c:
                print((f"self.last_centers {self.last_centers[i][0]:.2f}"
                       f"{self.last_centers[i][1]:.2f}"
                       f"{self.last_centers[i][2]:.2f}"))
        for i in range(self.skelet_nbr):
            if self.centers[i][0] and self.centers[i][1] and self.centers[i][2]:
                print((f"self.centers {self.centers[i][0]:.2f}"
                       f"{self.centers[i][1]:.2f} {self.centers[i][2]:.2f}")
        print("whos:", whos)

    def attribution_en_dernier_recours_par_ordre(self, whos):
        """whos = [None]*nombre_de_personnage
        whos = [None, None, None, None] et nombre_de_squelettes = 2
        table_persos     =       [0 1 2 3]
        table_squelettes =       [0 1]
                                    0    1  2   3
        whos =                   [None None 0 None]
        perso_sans_attribution = [  0    1      3]
        """
        # Si des squelettes ne sont pas attribués. Combiens à attribuer?
        nombre_d_attribuer = len([x for x in whos if x is not None])
        nombre_de_sans_attribution = self.skelet_nbr - nombre_d_attribuer

        if nombre_de_sans_attribution > 0:
            index = 0
            for who in whos:
                if who is None:
                    whos[index] = index
                    print("Attribution du squelette", index, "à", index)
                    nombre_de_sans_attribution -= 1
                    if nombre_de_sans_attribution == 0:
                        break
                index += 1

        print("Nombre de squlettes attribués =", nombre_d_attribuer)
        # #print("Personnages sans attribution =", perso_sans_attribution)
        return whos

    def apply_to_personnages(self, whos, persos_2D, persos_3D, dists):
        """ whos du type [1, 0, None, 2]
                                1 attribué au perso 0
                                0 attribué au perso 1 ... etc ...
        """

        for i in range(self.nombre_de_personnage):
            if whos[i] is not None:
                self.personnages[i].who = whos[i]
                self.personnages[i].xys = persos_2D[whos[i]]
                self.personnages[i].points_3D = persos_3D[whos[i]]
                c = get_center(persos_3D[whos[i]])
                self.personnages[i].center = c
                self.personnages[i].add_historic(c)
                self.personnages[i].dist = dists[i]
            else:
                self.personnages[i].reset()

    def get_persos_3D(self, persos_2D):
        persos_3D = []
        for xys in persos_2D:
            pts = self.get_points_3D(xys)
            persos_3D.append(pts)
        return persos_3D

    def get_points_3D(self, xys):
        """Calcul des coordonnées 3D dans un repère centré sur la caméra,
        avec le z = profondeur
        La profondeur est une moyenne de la profondeur des points autour,
        sauf les extrêmes, le plus petit et le plus gand.
        """

        points_3D = [None]*17
        for key, val in xys.items():
            if val:
                # Calcul de la profondeur du point
                # TODO: mettre dans fct
                distances = []
                x, y = val[0], val[1]
                # around = nombre de pixel autour du points
                x_min = max(x - self.around, 0)
                x_max = min(x + self.around, self.depth_frame.width)
                y_min = max(y - self.around, 0)
                y_max = min(y + self.around, self.depth_frame.height)

                for u in range(x_min, x_max):
                    for v in range(y_min, y_max):
                        # Profondeur du point de coordonnée (u, v) dans l'image
                        distances.append(self.depth_frame.get_distance(u, v))

                # Si valeurs non trouvées, retourne [0.0, 0.0, 0.0, 0.0]
                # Remove the item 0.0 for all its occurrences
                dists = [i for i in distances if i != 0.0]
                dists_sort = sorted(dists)
                if len(dists_sort) > 2:  # TODO: voir si else ?
                    # Suppression du plus petit et du plus grand
                    goods = dists_sort[1:-1]
                    # TODO: rajouter un filtre sur les absurdes ?

                    # Calcul de la moyenne des profondeur
                    somme = 0
                    for item in goods:
                        somme += item
                    profondeur = somme/len(goods)

                    # Calcul les coordonnées 3D avec x et y coordonnées dans
                    # l'image et la profondeur du point
                    # Changement du nom de la fonction trop long
                    point_2D_to_3D = rs.rs2_deproject_pixel_to_point
                    point_with_deph = point_2D_to_3D(self.depth_intrinsic,
                                                     [x, y],
                                                     profondeur)
                    points_3D[key] = point_with_deph

        return points_3D

    def draw_all_poses(self):
        for i, perso in enumerate(self.personnages):
            if perso.xys:
                self.draw_pose(perso.xys, COLORS[i])

    def draw_pose(self, xys, color):
        """Affiche les points 2D, et les 'os' dans l'image pour un personnage
        xys = {0: [790, 331], 2: [780, 313],  ... }
        """
        points = []
        for xy in xys.values():
            points.append(xy)

        # Dessin des points
        for point in points:
            x = point[0]
            y = point[1]
            cv2.circle(self.color_arr, (x, y), 5, color=(100, 100, 100),
                                                              thickness=-1)
            cv2.circle(self.color_arr, (x, y), 6, color=color, thickness=1)

        # Dessin des os
        for a, b in EDGES:
            if a not in xys or b not in xys:
                continue
            ax, ay = xys[a]
            bx, by = xys[b]
            cv2.line(self.color_arr, (ax, ay), (bx, by), color, 2)

    def draw_all_texts(self):
        for i, perso in enumerate(self.personnages):
            if perso.center and perso.center[2]:
                text = perso.dist
                x = 30
                y = 200 + i*100
                self.draw_texte(text, x, y, COLORS[i])

    def draw_texte(self, depth, x, y, color):
        """Affichage d'un texte"""
        cv2.putText(self.color_arr,             # image
                    f"{depth:.3f}",             # text
                    (x, y),                     # position
                    cv2.FONT_HERSHEY_SIMPLEX,   # police
                    2,                          # taille police
                    color,                      # couleur
                    4)                          # épaisseur

    def send_OSC(self):
        """Envoi en OSC des points 3D de tous les personnages"""

        for i, perso in enumerate(self.personnages):
            if perso.points_3D:
                self.osc.send_points_bundle(i, perso.points_3D)

    def run(self):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        t0 = time()
        nbr = 0

        while True:
            nbr += 1
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            color = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()
            if not self.depth_frame:
                continue

            color_data = color.as_frame().get_data()
            self.color_arr = np.asanyarray(color_data)

            outputs = self.this_posenet.get_outputs(self.color_arr)
            # Recherche des personnages captés
            self.get_personnages(outputs)

            # Envoi OSC
            # #self.send_OSC()

            # Affichage de l'image
            cv2.imshow('color', self.color_arr)

            # Calcul du FPS, affichage toutes les 10 s
            if time() - t0 > 10:
                print("FPS =", int(nbr/10))
                t0, nbr = time(), 0

            # Pour quitter
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Du OpenCV propre
        cv2.destroyAllWindows()


def get_distance(p1, p2):
    """Distance entre les points p1 et p2"""

    if p1 and p2:
        if None not in p1 and None not in p2:
            d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
            return d**0.5


def get_center(points_3D):

    center = []
    if points_3D:
        for i in range(3):
            center.append(get_moyenne(points_3D, i))

    return center


def get_moyenne(points_3D, indice):
    """Calcul la moyenne d'une coordonnée des points,
    la profondeur est le 3 ème = z
    indice = 0 pour x, 1 pour y, 2 pour z
    """

    somme = 0
    n = 0
    for i in range(17):
        if points_3D[i]:
            n += 1
            somme += points_3D[i][indice]
    if n != 0:
        moyenne = somme/n
    else:
        moyenne = None

    return moyenne


def main():

    # Création de l'objet
    p3d = Personnages3D()

    # On tourne, silence, caméra, action !!!
    p3d.run()


if __name__ == '__main__':
    """Excécution de ce script en standalone"""

    main()


    # #def attribution_avec_le_dernier_proche(self, persos_3D):
        # #self.last_centers = []
        # ## 4 personnages
        # #for i in range(self.nombre_de_personnage):
            # #self.last_centers.append(self.personnages[i].center)

        # #self.new_centers = []
        # ## 3 squelettes
        # #for i in range(len(persos_3D)):
            # #self.new_centers.append(get_center(persos_3D[i]))

        # ## Parcours des 4 personnages et des 3 squelettes
        # ## self.last_centers [[-0.0814, -0.0644, 0.922], [0, 0, 0]]
        # ## self.new_centers [[-0.0864, -0.0363, 0.936]]
        # #whos = [None]*self.nombre_de_personnage
        # #for i in range(self.nombre_de_personnage):  # 4
            # #for j in range(len(persos_3D)):  # 3
                # #dist = get_distance(self.new_centers[j], self.last_centers[i])
                # #if dist and dist < self.distance:
                        # #whos[i] = j
                        # #print(f"Squelette {j} attribué avec distance = {dist:.2f} pour self.distance = {self.distance}")
        # #return whos, self.last_centers, self.new_centers
