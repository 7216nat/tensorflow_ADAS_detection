import json
import kuksa_viss_client



def kuksa_ini():
    config = {"ip": "localhost", "port": 8090, "insecure": False}
    client = kuksa_viss_client.KuksaClientThread(config)
    with open("/usr/lib/python3.10/site-packages/kuksa_certificates/jwt/all-read-write.json.token", "r") as f:
        token = f.read()
    client.start()
    client.authorize(token)
    return client
