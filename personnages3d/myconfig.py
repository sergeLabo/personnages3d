

"""
Charge une configuration à partir d'un fichier *.ini
"""


import ast
from configparser import SafeConfigParser


class MyConfig():
    """
    Charge la configuration depuis le fichier *.ini,
    sauve les changement de configuration,
    enregistre les changements par section, clé.
    """

    def __init__(self, ini_file, verbose=1):
        """
        Charge la config depuis un fichier *.ini
        Le chemin doit être donné avec son chemin absolu
        ou dans le même dossier que ce fichier.
        """

        self.conf = {}
        self.ini = ini_file
        self.verbose = verbose
        self.load_config()


    def load_config(self):
        """Lit le fichier *.ini, et copie la config dans un dictionnaire."""

        parser = SafeConfigParser()
        parser.read(self.ini, encoding="utf-8")

        # Lecture et copie dans le dictionnaire
        for section_name in parser.sections():
            self.conf[section_name] = {}
            for key, value in parser.items(section_name):
                self.conf[section_name][key] = value  #ast.literal_eval(value)

        if self.verbose:
            print(f"\nConfiguration chargée depuis {self.ini}")

        # Si erreur chemin/fichier
        if not self.conf:
            print("Le fichier de configuration est vide")

    def save_config(self, section, key, value):
        """
        Sauvegarde dans le fichioer *.ini  avec section, key, value.
        Uniquement int, float, str
        """

        if isinstance(value, int):
            val = str(value)
        elif isinstance(value, float):
            val = str(value)
        elif isinstance(value, str):
            val = """ + value + """
        else:
            print("Sauvegarde impossible")
            return

        config = SafeConfigParser()
        config.read(self.ini)
        config.set(section, key, val)
        with open(self.ini, "w") as f:
            config.write(f)
        f.close()
        if self.verbose:
            print(f"Section={section} key={key} val={val} saved in {self.ini}\n")
