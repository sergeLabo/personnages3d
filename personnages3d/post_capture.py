

from oscpy.client import OSCClient


class MyOSC:

    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', None)
        self.port = int(kwargs.get('port', None))

        self.client = OSCClient(self.ip, self.port)

    def send_points(self, index, points):
        """index de 0 à 3
        Pour 1 personnage à l'index 1:
            /1/0 à /1/16
        Les coordonnées des points sont arrondies au mm
        """

        if points:
            bund = []
            for i, point in enumerate(points):
                if point:
                    tag = ('/' + str(index) + '/' + str(i)).encode('utf-8')
                    bund.append([tag, point])
            # #print("bund", bund)
            if bund:
                self.client.send_bundle(bund)

    def send_some(self, tag, some):
        """tag = str
        some = list
        Les coordonnées des points sont arrondies au mm
        """
        if some and tag:
            some_mm = [x for x in some if x is not None and x != 100000]
            # #print("some:", tag.encode('utf-8'), some_mm)
            self.client.send_message(tag.encode('utf-8'), some_mm)


class PostCaptureProcessing:
    """Utilisation des points 3D capturés, en fonction du projet global,
    puis envoi des résultats en OSC.
    """

    def __init__(self, **config):
        self.config = config
        self.my_OSC = MyOSC(**config['osc'])
        self.personnages = None


    def update(self, personnages):
        """Le tag pour un centre de personnage est:
            /1/center avec 1 = numéro du personnage
        """
        self.personnages = personnages

        if int(self.config['postprocessing']['centers']):
            for i, personnage in enumerate(self.personnages):
                tag = '/' + str(i) + '/' + 'center'
                if personnage.center != [1000]*3:
                    some = personnage.center
                    self.my_OSC.send_some(tag, some)

        if int(self.config['postprocessing']['all_points']):
            for i, personnage in enumerate(self.personnages):
                self.my_OSC.send_points(i, personnage.points_3D)


if __name__ == "__main__":

    messages = [[3.2, 3, 4], [55.6, 12, 80]]
    cli = MyOSC(**{'ip': '127.0.0.1', 'port': 8003})
    cli.send_points_bundle(0, messages)
