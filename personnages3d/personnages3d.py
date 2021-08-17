

"""
Echap pour finir proprement le script

Capture de 1 à 4 squelettes
avec
camera Intel RealSense D455, Google posenet et Google Coral.
Envoi de toutes les coordonnées de tous les points en OSC.
"""


import os
from time import time, sleep
from threading import Thread

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
        self.center = [1000]*3
        # 10x et 10y et 10z soit 1 seconde
        self.historic = [0]*3
        self.historic[0] = [0]*10
        self.historic[1] = [0]*10
        self.historic[2] = [0]*10
        # Moyenne lissée
        self.moving_center = [1000]*3
        # Prédiction de la position future
        self.future = [0]*3
        self.dist = 0

    def add_historic(self, centre):
        """Ajout dans la pile, suppr du premier"""

        if centre:
            for i in range(3):
                if centre[i]:
                    self.historic[i].append(centre[i])
                    del self.historic[i][0]
            self.EMA_update()
            self.predict_the_future()

    def predict_the_future(self):
        """Predit la position du personnage à la frame suivante, pour optimiser
        le calcul de distance lors de l'attribution d'un squelette.
        10, 12 prochain = 14 --> 12 + 2 = 14, 12 + 12 - 10 = 24 - 10
        """
        for i in range(3):
            self.future[i] = 2*self.historic[i][-1] - self.historic[i][-2]

    def EMA_update(self):
        """Filtre ave moyenne glissante, crée de la latence,
        Le centre est vu du dessus, pas de verticale=y
        """
        for i in range(3):
            self.moving_center[i] = moving_average(self.historic[i], 4,
                                    type='simple')  # 'exponentiel' ou 'simple'


class Personnages3D:
    """ Capture avec  Camera RealSense D455
        Détection de la pose avec Coral USB Stick
        Calcul des coordonnées 3D de chaque personnage et envoi en OSC.
        La profondeur est le 3ème dans les coordonnées d'un point 3D,
        x = horizontale, y = verticale
    """
    # TODO Calcul sur cpu ?

    def __init__(self, **kwargs):
        """Les paramètres sont définis dans le fichier personnages3d.ini"""

        self.config = kwargs
        print(f"Configuration:\n{self.config}\n\n")

        # 1 pour avoir les print
        self.debug = 0

        # Seuil de confiance de reconnaissance du squelette
        self.threshold = float(self.config['pose']['threshold'])

        # Nombre de pixels autour du point pour moyenne du calcul de profondeur
        self.around = int(self.config['pose']['around'])

        # Distance de rémanence pour attribution des squelettes
        self.distance = float(self.config['pose']['distance'])

        # Taille d'image possible: 1280x720, 640x480 seulement
        # 640x480 est utile pour fps > 30
        # Les modèles posenet imposent une taille d'image
        self.width = int(self.config['camera']['width_input'])
        self.height = int(self.config['camera']['height_input'])

        # Plein écran de la fenêtre OpenCV
        self.full_screen = int(self.config['camera']['full_screen'])

        # Le client va utiliser l'ip et port du *.ini
        self.osc = OscClient(**self.config['osc'])

        self.create_window()
        self.set_pipeline()

        self.this_posenet = ThisPosenet(self.width, self.height)

        # Toutes les datas des personnages dans un dict self.personnages
        self.person_nbr = min(int(self.config['pose']['person_nbr']), 4)
        self.personnages = []
        for i in range(self.person_nbr):
            self.personnages.append(Personnage())
        self.skelet_nbr = 0
        self.new_centers = None
        self.all_persos_3D = []

        # GUI
        self.loop = 1

    def create_window(self):
        if self.full_screen:
            cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('color', cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WINDOW_AUTOSIZE)
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")

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

    def main_frame(self, outputs):
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
            self.all_persos_3D.append(persos_3D)

            # Récup de who, apply to self.perso
            if persos_3D:
                self.skelet_nbr = min(len(persos_3D), 4)
                self.update_centers(persos_3D)
                whos, dists = self.who_is_who(persos_3D)
                self.apply_to_personnages(whos, persos_2D, persos_3D, dists)

            # Affichage
            self.draw_all_poses()
            self.draw_all_textes()
            self.draw_all_personnages()

    def update_centers(self, persos_3D):
        """
        last_centers = liste des centres tirée de l'historique des piles de centre
        self.new_centers = liste des centres des squelettes de la frame
        """
        # TODO remplacer last_centers par future_centers
        self.last_centers = []
        for i in range(self.person_nbr):
            # #self.last_centers.append(self.personnages[i].moving_center)
            self.last_centers.append(self.personnages[i].future)

        self.new_centers = []
        # 3 squelettes
        for i in range(self.skelet_nbr):
            self.new_centers.append(get_center(persos_3D[i]))

        if self.debug:
            print("self.last_centers", self.last_centers )
            print("self.new_centers", self.new_centers)

    def who_is_who(self, persos_3D):

        self.update_centers(persos_3D)

        if self.debug:
            for perso in self.personnages:
                print()
                print("who", perso.who)
                print("xys", perso.xys)
                print("points_3D", perso.points_3D)
                print("center", perso.center)
                print("historic", perso.historic)
                print("moving_center", perso.moving_center)
                print("dist", perso.dist)

        whos = [None]*self.person_nbr
        dists = [0]*self.person_nbr

        # Bazar avec copie de liste
        #   squelette 0     1     2     3
        table = [   [1000, 1000, 1000, 1000],  # squelette possible pour perso 0
                    [1000, 1000, 1000, 1000],  # squelette possible pour perso 1
                    [1000, 1000, 1000, 1000],  # squelette possible pour perso 2
                    [1000, 1000, 1000, 1000]]  # squelette possible pour perso 3

        for i in range(self.person_nbr):  # 3
            for j in range(self.skelet_nbr):  # 4
                if self.debug:
                    print("dist entre", self.new_centers[j], self.last_centers[i])
                d = get_distance(self.new_centers[j], self.last_centers[i])
                if d > 1000: d = 1000  # pour faire joli
                else: table[i][j] = d

        if self.debug:
            print()
            print(table[0])
            print(table[1])
            print(table[2])
            print(table[3])

        # [0.0293, 1000, 1000, 1000]
        # [1000,   1000, 1000, 1000]
        # [1000,   1000, 1000, 1000]
        # [1000,   1000, 1000, 1000]

        # de whos = [None, None, None, None]
        # avec distance < mini --> [None, 0, None, None]  perso1 a le squelette0
        # avec whos = [1, 0, None, None] perso0 a le squelette1

        # Premier passage avec de dist inférieur au mini
        for i in range(self.person_nbr):
            mini = sorted(table[i])[0]
            # Nouvelle position proche de l'ancienne
            if mini < self.distance:
                if i not in whos and i < self.skelet_nbr:
                    index = table[i].index(mini)
                    whos[index] = i
                    dists[index] = mini
        if self.debug:
            print("premier whos", whos)

        # Deuxième passage: nouveau squelette sans historique ou trop loin
        skelet_done = [x for x in whos if x is not None]
        if self.debug:
            print("skelet_done", skelet_done)  # [0]

        skelet_not_done = []
        for i in range(self.skelet_nbr):
            if i not in skelet_done:
                skelet_not_done.append(i)
        if self.debug:
            print("skelet_not_done", skelet_not_done)

        #  skelet_not_done   = [0, 1]
        #  skelet_done       = []
        #  occurence de None = [0,     1,    2,    3]
        #  whos        = [None, None, None, None]
        #        whos        = [1,    0,    None, None]

        # Occurence None in whos:
        None_occurence = []
        for n, val in enumerate(whos):
            if val is None:
                None_occurence.append(n)
        if self.debug:
            print("None_occurenceurence de None:", None_occurence)

        for i, sq in enumerate(skelet_not_done):
            whos[sq] = None_occurence[i]

        if self.debug:
            print("whos final:", whos, "dists:", dists, "\n")
        return whos, dists

    def apply_to_personnages(self, whos, persos_2D, persos_3D, dists):
        """ whos du type [1, 0, None, 2]
                                1 attribué au perso 0
                                0 attribué au perso 1 ... etc ...
        """

        for i in range(self.person_nbr):
            if whos[i] is not None:
                self.personnages[i].who = whos[i]
                self.personnages[i].xys = persos_2D[whos[i]]
                self.personnages[i].points_3D = persos_3D[whos[i]]
                c = get_center(persos_3D[whos[i]])
                self.personnages[i].center = c
                self.personnages[i].add_historic(c)
                self.personnages[i].dist = dists[i]
            # #else:  TODO utile ou pas, crée une rémanence si skelet perdu
                # #print("Reset de:", i)
                # #self.personnages[i].reset()
        pass

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
                profondeur = self.get_profondeur(val)
                if profondeur:
                    # Calcul les coordonnées 3D avec x et y coordonnées dans
                    # l'image et la profondeur du point
                    # Changement du nom de la fonction trop long
                    point_2D_to_3D = rs.rs2_deproject_pixel_to_point
                    point_with_deph = point_2D_to_3D(self.depth_intrinsic,
                                                     [val[0], val[1]],  # x, y
                                                     profondeur)
                    points_3D[key] = point_with_deph

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
        # TODO: voir si else utile

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
        for i, perso in enumerate(self.personnages):
            if perso.dist:
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

    def draw_all_personnages(self):
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        cv2.line(self.black, (0, 360), (1280, 360), (255, 255, 255), 2)
        for i, perso in enumerate(self.personnages):
            if perso.center and perso.center[0] and perso.center[2]:
                x = 360 + int(perso.center[0]*160)
                if x < 0: x = 0
                if x > 1280: x = 1280
                y = int(perso.center[2]*160)
                if y < 0: y = 0
                if y > 720: y = 720
                self.draw_personnage(y, x, COLORS[i])

    def draw_personnage(self, x, y, color):
        cv2.circle(self.black, (x, y), 10, (100, 100, 100), -1)
        cv2.circle(self.black, (x, y), 12, color=color, thickness=2)

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

            # Envoi OSC
            self.send_OSC()

            # Affichage de l'image
            cv2.imshow('color', self.color_arr)
            cv2.imshow('vue du dessus', self.black)

            # Calcul du FPS, affichage toutes les 10 s
            if time() - t0 > 10:
                # #print("FPS =", int(nbr/10))
                t0, nbr = time(), 0

            # Pour quitter
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Du OpenCV propre
        cv2.destroyAllWindows()

    def run_Process(self, conn):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        t0 = time()
        nbr = 0
        self.receive_thread(conn)
        while self.loop:
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
            self.main_frame(outputs)

            # Envoi OSC
            self.send_OSC()

            # Affichage de l'image
            cv2.imshow('color', self.color_arr)
            cv2.imshow('vue du dessus', self.black)

            # Calcul du FPS, affichage toutes les 10 s
            if time() - t0 > 10:
                print("FPS =", int(nbr/10))
                t0, nbr = time(), 0

            # Pour quitter
            if cv2.waitKey(1) & 0xFF == 27:
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
            sleep(0.001)


def get_distance(p1, p2):
    """Distance entre les points p1 et p2, dans le plan horizontal,
    sans prendre en compte le y qui est la verticale.
    """

    if p1 and p2:
        if None not in p1 and None not in p2:
            d = ((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)**0.5
            return d
    return 1000


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

    ini_file = 'personnages3d.ini'
    config_obj = MyConfig(ini_file)
    config = config_obj.conf

    # Création de l'objet
    p3d = Personnages3D(**config)

    # On tourne, silence, caméra, action !!!
    p3d.run()


def run_in_Process(config, conn):

    p3d = Personnages3D(**config)
    p3d.run_Process(conn)


if __name__ == '__main__':
    """Excécution de ce script en standalone"""

    main()
