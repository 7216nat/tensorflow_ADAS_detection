import json
import kuksa_viss_client



def kuksa_ini():
    config = {"cacertificate": "../kuksa_certificates/CA.pem",
        "certificate": "../kuksa_certificates/Client.pem",
        "key": "../kuksa_certificates/Client.key"}
    client = kuksa_viss_client.KuksaClientThread(config)
    with open("../kuksa_certificates/super-admin.json.token", "r") as f:
        token = f.read()
    client.start()
    client.authorize(token)
    return client
