
import json

dico = {
    "test":1,
    "retest":[
        1,
        2,
        3
    ]
}

directory = "C:/Users/Marie Bienvenu/stage_m2/afac/"
filename="afac.json"

name = directory+filename

with open(name, "w") as outfile:
    json.dump(dico, outfile, indent=4)

with open(name, 'r') as openfile:
    config = json.load(openfile)

print(config)

json_string = json.dumps(dico, indent=4)
print(json_string)

name2 = directory+"afac2.json"

with open(name2, "w") as outfile:
    json.dump(json_string, outfile, indent=4)

with open(name2, 'r') as openfile:
    config = json.load(openfile)

print(config)
