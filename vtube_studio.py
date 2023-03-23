import requests
import asyncio # only needed for async example
import websocket
import json
import random
import time

# https://github.com/DenchiSoft/VTubeStudio#authentication refers to this websocket API
class MBIS_vtube:
    def __init__(self, plugin_name="MyBitchIsAi", plugin_developer='HRNPH', port=8001):
        self.websocket = websocket.WebSocket()
        self.port = port
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer 
        self.auth_token = None

        self.msg_template = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"TestInit",
            "messageType": "APIStateRequest"
        }
        auth_status = self.auth()
        print('Authenticated with VTube Studio...')
        print(f'Authentication token: {self.auth_token}')
        print(f"Status: {auth_status}")

    def auth(self, noreturn=False) -> dict:
        if self.auth_token is None:
            msg = {
                "pluginName": f"{self.plugin_name}",
                "pluginDeveloper": f"{self.plugin_developer}"
            }
            result = self.send("AuthenticationTokenRequest", msg)
            self.auth_token = result['data']['authenticationToken'] # update auth token
            if noreturn:
                return
            return result




    def send(self, msg_type:str, msg:dict, noreturn=False) -> dict:
        try:
            self.websocket.connect(f"ws://localhost:{self.port}")
            if self.auth_token != None and msg_type != "AuthenticationRequest":
                # perform login with auth token for authentication
                auth_msg = {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "authenticationToken": self.auth_token,
                }
                result = self.send("AuthenticationRequest", auth_msg, noreturn=noreturn)

        except Exception as e:
            print("Error: ", e)
            print("This usually means that VTube Studio is not running or the public API is not enabled.")
            print("The permissions are not set correctly and/or got rejected.")
            # alert user to start VTube Studio
            raise Exception("Please start VTube Studio and enable the public API in the settings.")
        
        msg_temp = self.msg_template
        msg_temp['messageType'] = msg_type
        if msg is not None:
            msg_temp['data'] = msg
        # self.websocket.send(json.dumps(msg_temp))
        # handle ConnectionAbortedError
        try:
            self.websocket.send(json.dumps(msg_temp))
        except ConnectionAbortedError:
            print('ConnectionAbortedError: Reconnecting...')
            self.websocket.connect(f"ws://localhost:{self.port}")
            self.websocket.send(json.dumps(msg_temp))

        if noreturn:
            return
        return json.loads(self.websocket.recv())
    
    def recv(self):
        return self.websocket.recv()
    
    def close(self):
        self.websocket.close()

## init from mbis
class Char_control(MBIS_vtube):
    # init all the things with super
    def __init__(self, plugin_name="MyBitchIsAi", plugin_developer='HRNPH', port=8001):
        super().__init__(plugin_name, plugin_developer, port)

    def express(self, expression:str, expression_dict=None):
        if expression_dict is None:
            expression_dict = {
                "netural": None,
                "agree": "N1",
                "wonder": "N2",
                "shy": "N3",
                "happy": "N4",
                "sad": "N5",
            } # This need to be updated to match your expressions in VTube Studio
    
        # get expression hotkey ID
        available_hotkey_ids = self.send('HotkeysInCurrentModelRequest', None)['data']['availableHotkeys']
        for each_hotkey in available_hotkey_ids:
            name = each_hotkey['name'].lower()
            expression_dict[name] = each_hotkey['hotkeyID']

        try:
            hotkey_id = expression_dict[expression]
        except KeyError:
            print("Invalid expression")
            available_expressions = expression_dict.keys()
            print(f'Available expressions: {expression_dict.keys()}')
            return available_expressions

        # trigger expression VIA hotkey if expression is not netural
        if hotkey_id is not None:
            msg = {
                "hotkeyID": hotkey_id,
            }
            result = self.send("HotkeyTriggerRequest", msg)
            return result

if __name__ == "__main__":
    waifu = Char_control()
    print('------------------------------------')
    print(waifu.express('happy'))