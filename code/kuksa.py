import json
import kuksa_viss_client



def kuksa_ini():
    config = {"cacertificate": "/etc/kuksa-val/CA.pem",
        "certificate": "/etc/kuksa-val/Client.pem",
        "key": "/etc/kuksa-val/Client.key"}
    client = kuksa_viss_client.KuksaClientThread(config)
    with open("../kuksa_certificates/super-admin.json.token", "r") as f:
        token = f.read()
    client.start()
    client.authorize(token)
    return client
