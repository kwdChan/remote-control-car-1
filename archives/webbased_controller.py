from aiohttp import web
import aiohttp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection


    
class WebbaseTwoWheelsControl:

    @staticmethod
    def event2command(event_key, event_type):
        is_active = (event_type=='press') # press or release

        if event_key == 'w':
            command = 'go'

        elif event_key == 's':
            command =  'stop'

        elif event_key == 'e':
            command =  'V+'

        elif event_key == 'q':
            command =  'V-'

        elif event_key == 'a':
            command =  'L+'

        elif event_key =='d':
            command =  'R+'
        else:
            command =  '.'
        return {command: is_active}

    @staticmethod
    def start(port=8765):
        receiver, sender = Pipe(False)

        def callback(event_key, event_type):

            sender.send(WebbaseTwoWheelsControl.event2command(event_key, event_type))

        p = Process(target=get_webbased_wheel_controller, args=(port, [callback]))
        p.start()
        return p, receiver





def get_webbased_wheel_controller(port, callbacks=[]):

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                event = msg.json()
                event_type = event['type']
                key = event['key']
                #print(f"Received {event_type} event: {key}")

                for cb in callbacks:

                    cb(event['key'], event['type'])

                #await ws.send_str(f"Received: {msg.data}")

            elif msg.type == aiohttp.WSMsgType.ERROR:
                print('WebSocket connection closed with exception %s' % ws.exception())

        print('WebSocket connection closed')
        return ws

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)
    web.run_app(app, port=port)





async def index(request):
    return web.Response(text="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Keypress Capture</title>
</head>
<body>
    <h1>Press and release keys to send events to the server</h1>
    <script>
        const ws = new WebSocket('ws://' + window.location.host + '/ws');

        ws.onopen = function(event) {
            console.log('Connection is open');
        };

        ws.onerror = function(error) {
            console.log('WebSocket Error: ' + error);
        };

        document.addEventListener('keydown', (event) => {
            const key = event.key;
            ws.send(JSON.stringify({type: 'press', key: key}));
            console.log(`Key "${key}" pressed`);
        });

        document.addEventListener('keyup', (event) => {
            const key = event.key;
            ws.send(JSON.stringify({type: 'release', key: key}));
            console.log(`Key "${key}" released`);
        });
    </script>
</body>
</html>
""", content_type='text/html')

