import json

control={"sigma_list": [15, 80, 200],
        "G" : 5.0,
        "b" : 25.0,
        "alpha" : 125.0,
        "beta" : 46.0,
        "low_clip" : 0.01,
        "high_clip" : 0.99
}

json.dump(control,open('config.json','w'),indent=4)

