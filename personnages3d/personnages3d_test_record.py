

"""
Echap pour finir proprement le script

Capture de 1 à 4 squelettes
avec
camera Intel RealSense D455, Google posenet et Google Coral.
Envoi de toutes les coordonnées de tous les points en OSC.
"""


import os
from time import time, sleep
from datetime import datetime
import json

import numpy as np
import cv2


import pyrealsense2 as rs
from posenet.this_posenet import ThisPosenet
from posenet.pose_engine import EDGES

from myconfig import MyConfig


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
    Moyenne simple des centres.
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


    def add_historic(self, centre):
        """Ajout dans la pile, suppr du premier"""

        for i in range(3):
            self.historic[i].append(centre[i])
            del self.historic[i][0]



class Personnages3D:
    """ Capture avec  Camera RealSense D455
        Détection de la pose avec Coral USB Stick
        Calcul des coordonnées 3D de chaque personnage, puis suite
        dans PostCaptureProcessing.
        La profondeur est le 3ème dans les coordonnées d'un point 3D,
        x = horizontale, y = verticale
    """

    def __init__(self, **kwargs):
        """Les paramètres sont définis dans le fichier personnages3d.ini"""

        # Enregistrement dans un json
        self.all_data = []

        self.config = kwargs
        print(f"Configuration:\n{self.config}\n\n")

        # Seuil de confiance de reconnaissance du squelette
        self.threshold = float(self.config['pose']['threshold'])

        # Nombre de pixels autour du point pour moyenne du calcul de profondeur
        self.around = int(self.config['pose']['around'])

        # Distance de rémanence pour attribution des squelettes
        self.distance = float(self.config['pose']['distance'])

        # Nombre deif k == 32:  # space personnes à capter
        self.person_nbr = min(int(self.config['pose']['person_nbr']), 4)

        self.whos = [0]*self.person_nbr

        # Taille d'image possible: 1280x720, 640x480 seulement
        # 640x480 est utile pour fps > 30
        # Les modèles posenet imposent une taille d'image
        self.width = int(self.config['camera']['width_input'])
        self.height = int(self.config['camera']['height_input'])

        # Plein écran de la fenêtre OpenCV
        self.full_screen = int(self.config['camera']['full_screen'])

        self.create_window()
        self.set_pipeline()
        self.this_posenet = ThisPosenet(self.width, self.height)

        # Toutes les datas des personnages dans un dict self.personnages
        self.personnages = []
        for i in range(self.person_nbr):
            self.personnages.append(Personnage(**self.config))
        self.skelet_nbr = 0
        self.new_centers = None

        self.loop = 1

    def create_window(self):

        cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WND_PROP_FULLSCREEN)
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")

    def set_window(self):
        if self.full_screen:
            cv2.setWindowProperty('color', cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty('vue du dessus', cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(  'color',
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(  'vue du dessus',
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL)

    def set_pipeline(self):
        """Crée le flux d'image avec la caméra D455

        1. (    self: pyrealsense2.pyrealsense2.config,
                stream_type: pyrealsense2.pyrealsense2.stream,
                stream_index: int,
                width: int,
                height: int,
                format: pyrealsense2.pyrealsense2.format = <format.any: 0>,
                framerate: int = 0) -> None

        format=rs.format.z16
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
        except:
            print(f'Pas de Capteur Realsense connecté')
            os._exit(0)

        device = pipeline_profile.get_device()
        config.enable_stream(   rs.stream.color,
                                width=self.width,
                                height=self.height,
                                format=rs.format.bgr8,
                                framerate=30)
        config.enable_stream(   rs.stream.depth,
                                width=self.width,
                                height=self.height,
                                format=rs.format.z16,
                                framerate=30)
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

    def get_body_in_center(self, persos_3D):
        """Recherche du perso le plus près du centre, pour 1 seul perso"""

        who = []
        # Tous les décalage sur x
        all_x_decal = []

        if persos_3D:
            for perso in persos_3D:
                # Le x est la 1ère valeur dans perso
                if perso:
                    decal = get_moyenne(perso, 0)
                    if decal:
                        all_x_decal.append(decal)
                    else:
                        all_x_decal.append(100000)

        if all_x_decal:
            all_x_decal_sorted = sorted(all_x_decal)
            decal_mini  = all_x_decal_sorted[0]
        who.append(all_x_decal.index(decal_mini))

        return who

    def main_frame(self, outputs):
        """ Appelé depuis la boucle infinie, c'est le main d'une frame.
                Récupération de tous les squelettes
                Definition de who
        """

        persos_2D, persos_3D = None, None
        # Récupération de tous les squelettes
        if outputs:
            # les xys
            persos_2D = PoseNetConversion(outputs, self.threshold).skeletons

            if persos_2D:
                # Ajout de la profondeur pour 3D
                persos_3D = self.get_persos_3D(persos_2D)
                self.all_data.append(persos_3D)

        # Récup de who, apply to self.perso
        if persos_3D:
            self.skelet_nbr = min(len(persos_3D), 4)

            # Si détection que d'un seul perso
            if self.person_nbr == 1:
                self.whos = self.get_body_in_center(persos_3D)
            else:
                self.who_is_who(persos_3D)

            self.apply_to_personnages(persos_2D, persos_3D)

            # Affichage
            self.draw_all_poses()
            self.draw_all_textes()
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

    def get_persos_3D(self, persos_2D):
        persos_3D = []
        for xys in persos_2D:
            # #print("xys", xys)
            pts = self.get_points_3D(xys)
            if pts:
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
                profondeur = self.get_profondeur(val)
                if profondeur:
                    # Calcul les coordonnées 3D avec x et y coordonnées dans
                    # l'image et la profondeur du point
                    # Changement du nom de la fonction trop long
                    point_2D_to_3D = rs.rs2_deproject_pixel_to_point
                    point_with_deph = point_2D_to_3D(self.depth_intrinsic,
                                                     [val[0], val[1]],  # x, y
                                                     profondeur)
                    # Conversion des m en mm
                    points_3D[key] = [int(1000*x) for x in point_with_deph]

        if points_3D == [None]*17:
            points_3D = None
        return points_3D

    def get_profondeur(self, val):
        """Calcul la moyenne des profondeurs des pixels auour du point considéré
        Filtre les absurdes et les trop loins
        """
        profondeur = None
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
        if len(dists_sort) > 2:
            # Suppression du plus petit et du plus grand
            goods = dists_sort[1:-1]
            # TODO: rajouter un filtre sur les absurdes ?

            # Calcul de la moyenne des profondeur
            somme = 0
            for item in goods:
                somme += item
            profondeur = somme/len(goods)

        return profondeur

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

    def draw_all_textes(self):
        """
        for i, perso in enumerate(self.personnages):
            if perso.dist:
                if perso.dist != 100000:
                    text = perso.dist
                    x = 30
                    y = 200 + i*100
                    self.draw_texte(text, x, y, COLORS[i])
        """
        text = "Distance:  " + str(self.distance)
        x = 30
        y = 50
        self.draw_texte(text, x, y, COLORS[3])

        text = "Confiance:  " + str(self.threshold)
        x = 30
        y = 100
        self.draw_texte(text, x, y, COLORS[2])

    def draw_texte(self, depth, x, y, color):
        """Affichage d'un texte"""
        cv2.putText(self.color_arr,             # image
                    str(depth),                 # text
                    (x, y),                     # position
                    cv2.FONT_HERSHEY_SIMPLEX,   # police
                    1,                          # taille police
                    color,                      # couleur
                    2)                          # épaisseur

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

        if conn:
            self.receive_thread(conn)

        while self.loop:
            nbr += 1

            frames = self.pipeline.wait_for_frames(timeout_ms=80)

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            color = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

            if not self.depth_frame and not color:
                continue

            color_data = color.as_frame().get_data()
            self.color_arr = np.asanyarray(color_data)

            outputs = self.this_posenet.get_outputs(self.color_arr)

            # Recherche des personnages captés
            self.main_frame(outputs)

            # Affichage de l'image
            cv2.imshow('color', self.color_arr)
            cv2.imshow('vue du dessus', self.black)

            # Calcul du FPS, affichage toutes les 10 s
            if time() - t0 > 10:
                # #print("FPS =", int(nbr/10))
                t0, nbr = time(), 0

            k = cv2.waitKey(1)

            # Enrergistrement si s
            if k == 115:  # s
                save(self.all_data)

            # Space pour full screen or not
            elif k == 32:  # space
                if self.full_screen == 1:
                    self.full_screen = 0
                elif self.full_screen == 0:
                    self.full_screen = 1
                self.set_window()

            # Pour quitter
            elif k == 27:  # Esc
                break

        # Du OpenCV propre
        cv2.destroyAllWindows()

    def receive_thread(self, conn):
        t = Thread(target=self.receive, args=(conn, ))
        t.start()

    def receive(self, conn):
        while self.loop:
            data = conn.recv()
            print("dans run processus =", data)
            if data[0] == 'quit':
                self.loop = 0
            elif data[0] == 'threshold':
                self.threshold = data[1]
            elif data[0] == 'around':
                self.around = data[1]
            elif data[0] == 'distance':
                self.distance = data[1]
            elif data[0] == 'stability':
                self.stability = data[1]
                for i in range(len(self.personnages)):
                    self.personnages[i].stability = self.stability
            sleep(0.001)



def save(all_data):
    dt_now = datetime.now()
    dt = dt_now.strftime("%Y_%m_%d_%H_%M")
    fichier = f"./json/cap_{dt}.json"
    with open(fichier, "w") as fd:
        fd.write(json.dumps(all_data))
        print(f"{fichier} enregistré.")


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
