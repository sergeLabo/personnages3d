
from time import sleep
from oscpy.server import OSCThreadServer


def on_tag(*args):
    dico = {}
    tag = int(args[0].decode('utf-8')[1:])
    dico[tag] = args[1:]
    print(dico)

def default_handler(*args):
    print("default_handler", args)

# #def on_bundle(*args):
    # #print(args)

server = OSCThreadServer()
server.listen(b'localhost', port=8003, default=True)
# #server.default_handler = default_handler

for j in range(4):
    for i in range(10):
        tag = ('/' + str(j) + '_'+ str(i)).encode('utf-8')
        server.bind(tag, on_tag, get_address=True)

# #server.bind('/bundle', on_bundle, get_address=True)

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
