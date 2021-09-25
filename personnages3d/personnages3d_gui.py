
"""
Interface graphique pour Personnages3D
"""


import os
import sys
from pathlib import Path
from multiprocessing import Process, Pipe

import kivy
kivy.require('2.0.0')

from kivy.core.window import Window
k = 1.0
WS = (int(1280*k), int(720*k))
Window.size = WS

# TODO verifier ceux inutilisés
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock

from personnages3d import run_in_Process


class MainScreen(Screen):
    """Ecran principal, l'appli s'ouvre sur cet écran
    root est le parent de cette classe dans la section <MainScreen> du kv
    """

    # Attribut de class, obligatoire pour appeler root.titre dans kv
    titre = StringProperty("toto")
    enable = BooleanProperty(False)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # Trop fort
        self.app = App.get_running_app()

        # Pour envoyer les valeurs au child_conn
        self.perso = int(self.app.config.get('pose', 'person_nbr'))
        self.threshold = 0.5
        self.around = 1
        self.distance = 200
        self.stability = 3
        self.parent_conn = None

        # Pour ne lancer qu'une fois
        self.block = 0
        self.enable = False

        self.titre = "Personnages 3D"

        Clock.schedule_once(self.set_toggle, 0.5)

        print("Initialisation du Screen MainScreen ok")

    def set_toggle(self, dt):
        """Les objets graphiques ne sont pas encore créé pendant le init,
        il faut lancer cette méthode plus tard
        """

        if self.perso == 1:
            self.ids.p1.state = "down"
        elif self.perso == 2:
            self.ids.p2.state = "down"
        elif self.perso == 3:
            self.ids.p3.state = "down"
        elif self.perso == 4:
            self.ids.p4.state = "down"

    def set_personnages_number(self, num):
        scr = self.app.screen_manager.get_screen('Main')
        print(f"nombre de personnages = {num}")
        self.perso = num
        self.app.config.set('pose', 'person_nbr', num)
        self.app.config.write()
        if scr.parent_conn:
            scr.parent_conn.send(['perso', self.perso])

    def run_personnages3d(self):
        """Lance personnages3D.py"""

        if not self.block:
            # parent_conn pour envoyer à p, child_conn dans p reçoit
            self.parent_conn, child_conn = Pipe()
            print("config dans MainScreen run_personnages3d", self.app.config)
            p = Process(target=run_in_Process, args=(self.app.config, child_conn, ))
            p.start()
            self.block = 1
            self.enable = True


class MySettings(Screen):
    """Some sliders"""

    threshold = NumericProperty(0.5)
    distance = NumericProperty(0.2)
    around = NumericProperty(1)
    stability = NumericProperty(8)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Initialisation du Screen Settings")


        self.app = App.get_running_app()
        self.threshold = float(self.app.config.get('pose', 'threshold'))
        self.around = int(self.app.config.get('pose', 'around'))
        self.distance = float(self.app.config.get('pose', 'distance'))
        self.stability = int(self.app.config.get('pose', 'stability'))

    def do_slider(self, iD, instance, value):

        scr = self.app.screen_manager.get_screen('Main')

        if iD == 'threshold':
            # Maj de l'attribut
            self.threshold = round(value, 2)
            # Maj de la config
            self.app.config.set('pose', 'threshold', self.threshold)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant
            if scr.parent_conn:
                scr.parent_conn.send(['threshold', self.threshold])

        if iD == 'around':
            self.around = int(value)

            self.app.config.set('pose', 'around', self.around)
            self.app.config.write()

            if scr.parent_conn:
                scr.parent_conn.send(['around', self.around])

        if iD == 'distance':
            self.distance = int(value)
            self.app.config.set('pose', 'distance', self.distance)
            self.app.config.write()

            if scr.parent_conn:
                scr.parent_conn.send(['distance', self.distance])

        if iD == 'stability':
            self.stability = int(value)
            self.app.config.set('pose', 'stability', self.stability)
            self.app.config.write()

            if scr.parent_conn:
                scr.parent_conn.send(['stability', self.stability])


# Variable globale qui définit les écrans
# L'écran de configuration est toujours créé par défaut
# Il suffit de créer un bouton d'accès
# Les class appelées (MainScreen, ...) sont placées avant
SCREENS = { 0: (MainScreen, 'Main'),
            1: (MySettings, 'MySettings')}


class Personnages3DApp(App):
    """Construction de l'application. Exécuté par if __name__ == '__main__':,
    app est le parent de cette classe dans kv
    """

    def build(self):
        """Exécuté après build_config, construit les écrans"""

        # Création des écrans
        self.screen_manager = ScreenManager()
        for i in range(len(SCREENS)):
            # Pour chaque écran, équivaut à
            # self.screen_manager.add_widget(MainScreen(name="Main"))
            self.screen_manager.add_widget(SCREENS[i][0](name=SCREENS[i][1]))

        return self.screen_manager

    def build_config(self, config):
        """Excécuté en premier (ou après __init__()).
        Si le fichier *.ini n'existe pas,
                il est créé avec ces valeurs par défaut.
        Il s'appelle comme le kv mais en ini
        Si il manque seulement des lignes, il ne fait rien !
        """

        print("Création du fichier *.ini si il n'existe pas")

        config.setdefaults( 'camera',
                                {   'width_input': 1280,
                                    'height_input': 720,
                                    'full_screen': 0})

        config.setdefaults( 'pose',
                                {   'threshold': 0.60,
                                    'around': 1,
                                    'person_nbr': 4,
                                    'distance': 100,
                                    'stability': 8,
                                    'len_histo': 100})

        config.setdefaults( 'osc',
                                {   'ip': '127.0.0.1',
                                    'port': 8003})

        config.setdefaults( 'postprocessing',
                                {   'centers': 1,
                                    'all_points': 0})

        print("self.config peut maintenant être appelé")

    def build_settings(self, settings):
        """Construit l'interface de l'écran Options, pour Personnages3D seul,
        Les réglages Kivy sont par défaut.
        Cette méthode est appelée par app.open_settings() dans .kv,
        donc si Options est cliqué !
        """

        print("Construction de l'écran Options")

        data = """[
                    {"type": "title", "title": "OSC"},
                        {   "type": "string",
                            "title": "IP pour envoi en OSC",
                            "desc": "En local: 127.0.0.1",
                            "section": "osc", "key": "ip"   },
                        {   "type": "numeric",
                            "title": "Port pour envoi en OSC",
                            "desc": "8003",
                            "section": "osc", "key": "port"   },

                    {"type": "title", "title": "Camera RealSense"},
                        {   "type": "numeric",
                            "title": "Largeur de l'image caméra",
                            "desc": "1280 ou 640",
                            "section": "camera", "key": "width_input"},
                        {   "type": "numeric",
                            "title": "Hauteur de l'image caméra",
                            "desc": "720 ouy 480",
                            "section": "camera", "key": "height_input"},
                        {   "type": "numeric",
                            "title": "Vue caméra en plein écran",
                            "desc": "0 ou 1",
                            "section": "camera", "key": "full_screen"},

                    {"type": "title", "title": "Détection des squelettes"},
                        {   "type": "numeric",
                            "title": "Seuil de confiance pour la detection d'un keypoint",
                            "desc": "0.01 à 0.99",
                            "section": "pose", "key": "threshold"},
                        {   "type": "numeric",
                            "title": "Nombre de pixels autour du point pour le calcul de la profondeur",
                            "desc": "Entier de 1 à 5",
                            "section": "pose", "key": "around"},
                        {   "type": "numeric",
                            "title": "Nombre de personnes à détecter",
                            "desc": "1 à 4",
                            "section": "pose", "key": "person_nbr"},
                        {   "type": "numeric",
                            "title": "Distance pour suivi des personnes",
                            "desc": "5 à 500",
                            "section": "pose", "key": "distance"},
                        {   "type": "numeric",
                            "title": "Stabilité",
                            "desc": "1 à 50",
                            "section": "pose", "key": "stability"},
                        {   "type": "numeric",
                            "title": "Historique du suivi",
                            "desc": "10 à 100",
                            "section": "pose", "key": "len_histo"},

                    {"type": "title", "title": "Post Processing"},
                        {   "type": "numeric",
                            "title": "Envoi des centres",
                            "desc": "1 pour envoi, 0 sinon",
                            "section": "postprocessing", "key": "centers"},
                        {   "type": "numeric",
                            "title": "Envoi de tous les points",
                            "desc": "1 pour envoi, 0 sinon",
                            "section": "postprocessing", "key": "all_points"}
                   ]"""

        # self.config est le config de build_config
        settings.add_json_panel('Personnages3D', self.config, data=data)

    def on_config_change(self, config, section, key, value):
        """Si modification des options, fonction appelée automatiquement
        menu = self.screen_manager.get_screen("Main")
        Seul les rébglages à chaud sont définis ici !
        """

        if config is self.config:
            token = (section, key)

            if token == ('pose', 'threshold'):
                if value < 0: value = 0
                if value > 0.99: value = 0.99
                self.threshold = value
                self.config.set('pose', 'threshold', value)

            if token == ('pose', 'distance'):
                if value < 5: value = 5
                if value > 500: value = 500
                self.distance = value
                self.config.set('pose', 'distance', value)

            if token == ('pose', 'around'):
                if value < 1: value = 1
                if value > 5: value = 5
                self.around = value
                self.config.set('pose', 'around', value)

            if token == ('pose', 'stability'):
                if value < 1: value = 1
                if value > 50: value = 50
                self.stability = value
                self.config.set('pose', 'stability', value)

    def go_mainscreen(self):
        """Retour au menu principal depuis les autres écrans."""
        self.screen_manager.current = ("Main")

    def do_quit(self):
        print("Je quitte proprement")

        # Fin du processus fils
        scr = self.screen_manager.get_screen('Main')
        if scr.parent_conn:
            scr.parent_conn.send(['quit'])

        # Kivy
        Personnages3DApp.get_running_app().stop()

        # Extinction forcée de tout, si besoin
        os._exit(0)



if __name__ == '__main__':
    """L'application s'appelle Personnages3D
    d'où
    la class
        Personnages3DApp()
    les fichiers:
        personnages3d.kv
        personnages3d.ini
    """

    Personnages3DApp().run()
