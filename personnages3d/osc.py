

from oscpy.client import OSCClient

class OscClient:

    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', None)
        self.port = kwargs.get('port', None)

        self.client = OSCClient(self.ip, self.port)

    def send_depth(self, depth):
        self.client.send_message(b'/depth', [depth])

    def send_points_bundle(self, index, messages):
        """index de 0 à 3"""

        bund = []

        for i, msg in enumerate(messages):
            tag = ('/' + str(index) + '_' + str(i)).encode('utf-8')
            bund.append([tag, msg])

        self.client.send_bundle(bund)


if __name__ == "__main__":

    messages = [[3.2, 3, 4], [55.6, 12, 80]]
    cli = OscClient(**{'ip': '127.0.0.1', 'port': 8003})
    cli.send_bundle(messages)
