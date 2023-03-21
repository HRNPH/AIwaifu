import requests
import asyncio # only needed for async example
import websocket
import json
import random
import time

# https://github.com/DenchiSoft/VTubeStudio#authentication refers to this websocket API
class MBIS_vtube:
    def __init__(self, plugin_name="MyBitchIsAi", plugin_developer='HRNPH'):
        self.websocket = websocket.WebSocket()
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.auth_token = None
        try:
            self.websocket.connect("ws://localhost:8001")
            self.msg_template = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": f"TestInit",
                "messageType": "APIStateRequest"
            }
            self.websocket.send(json.dumps(self.msg_template))
            result = self.websocket.recv()
            print(result)
            print('Connected to VTube Studio...')
            auth_status = self.auth()
            print('Authenticated with VTube Studio...')
            print(f'Authentication token: {self.auth_token}')
            print(f"Status: {auth_status['data']['reason']}")

        except Exception as e:
            print("Error: ", e)
            print("This usually means that VTube Studio is not running or the public API is not enabled.")
            print("The permissions are not set correctly and/or got rejected.")
            # alert user to start VTube Studio
            raise Exception("Please start VTube Studio and enable the public API in the settings.")


    def auth(self) -> dict:
        msg = {
            "pluginName": f"{self.plugin_name}",
            "pluginDeveloper": f"{self.plugin_developer}"
        }
        result = self.send("AuthenticationTokenRequest", msg)
        self.auth_token = result['data']['authenticationToken'] # update auth token

        # perform login
        msg = {
            "pluginName": self.plugin_name,
            "pluginDeveloper": self.plugin_developer,
            "authenticationToken": self.auth_token,
        }
        result = self.send("AuthenticationRequest", msg)
        return result

    def send(self, msg_type:str, msg:dict) -> dict:
        msg_temp = self.msg_template
        msg_temp['messageType'] = msg_type
        if msg is not None:
            msg_temp['data'] = msg
        self.websocket.send(json.dumps(msg_temp))
        return json.loads(self.websocket.recv())
    
    def recv(self):
        return self.websocket.recv()
    
    def close(self):
        self.websocket.close()

## init from mbis
class Char_control(MBIS_vtube):
    # init all the things
    def __init__(self):
        super().__init__()


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