
"""
Ce serveur reçoit les datas envoyée en OSC par personnage3D,
pour essai.
"""


from time import sleep
from oscpy.server import OSCThreadServer


dico = {0: {},
        1: {},
        2: {},
        3: {}}

def on_points(*args):
    # arg (b'/1_0', 0.285, -0.066, 0.871)

    resp = args[0].decode('utf-8')
    tag = int(resp[3:])
    i = int(resp[1])
    dico[i][tag] = args[1:]
    print("dico", dico)


def default_handler(*args):
    print("default_handler", args)

centers =  {0: {},
            1: {},
            2: {},
            3: {}}

def on_center(*args):
    resp = args[0].decode('utf-8')
    # print(resp) /0/center
    i = int(resp[1])
    centers[i] = args[1:]
    print("centers", centers)


server = OSCThreadServer()
server.listen(b'localhost', port=8003, default=True)
server.default_handler = default_handler

for j in range(4):
    for i in range(17):
        tag = ('/' + str(j) + '/' + str(i)).encode('utf-8')
        server.bind(tag, on_points, get_address=True)

for k in range(4):
    tag = ('/' + str(k) + '/' + 'center').encode('utf-8')
    server.bind(tag, on_center, get_address=True)

while 1:
    sleep(0.1)

"""
[
[b'/0_0', [-0.2605190575122833, -0.11665435880422592, 1.3580000400543213]],
[b'/0_1', [-0.24112436175346375, -0.13716775178909302, 1.3700001239776611]],
[b'/0_2', [-0.2772120237350464, -0.14223407208919525, 1.376500129699707]],
[b'/0_3', [-0.16387628018856049, -0.13703861832618713, 1.4144999980926514]],
[b'/0_4', [1000, 1000, 1000]],
[b'/0_5', [-0.09494805335998535, -0.00696173682808876, 1.3260000944137573]],
[b'/0_6', [1000, 1000, 1000]],
[b'/0_7', [-0.14415249228477478, 0.15182898938655853, 1.1069999933242798]],
[b'/0_8', [1000, 1000, 1000]],
[b'/0_9', [1000, 1000, 1000]],
[b'/0_10', [1000, 1000, 1000]],
[b'/0_11', [1000, 1000, 1000]],
[b'/0_12', [1000, 1000, 1000]],
[b'/0_13', [1000, 1000, 1000]],
[b'/0_14', [1000, 1000, 1000]],
[b'/0_15', [1000, 1000, 1000]],
[b'/0_16', [1000, 1000, 1000]]]
"""
