
from time import sleep
from oscpy.server import OSCThreadServer

dico = {}
def on_tag(*args):
    print(args)
    tag = int(args[0].decode('utf-8')[1:])
    print(tag)
    dico[tag] = args[1:]
    print(dico)

def default_handler(*args):
    print("default_handler", args)


server = OSCThreadServer()
server.listen(b'localhost', port=8003, default=True)
server.default_handler = default_handler

for i in range(10):
    tag = ('/' + str(i)).encode('utf-8')
    server.bind(tag, on_tag, get_address=True)

while 1:
    sleep(0.1)
