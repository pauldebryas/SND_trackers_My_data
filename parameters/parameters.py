# This script is meant to be used for writting the parameters of your SND@LHC detector in a more intuitive way than 
# writing it directly in the json file. 

#Each configuration ('DS5', 'SNDatLHC', ..) must be associated to a dataset. 

import json

snd_parameters = {
    "SNDatLHC": {"END_OF_BRICK": -28.5,
                 "X_min": 7.9,
                 "X_max": 49.1,
                 "Y_min": 15.4,
                 "Y_max": 56.6,
                 "TT_POSITIONS": [[-18.11, -13.08],
                                  [-7.71, -2.68],
                                  [2.68, 7.71],
                                  [13.08, 18.11],
                                  [23.48, 28.5]]
                },
    "DS5": {"END_OF_BRICK": -3041.0,
            "X_min": -26.0,
            "X_max": 26.0,
            "Y_min": -21.5,
            "Y_max": 21.5,
            "TT_POSITIONS": [[-3041.0, -3037.0],
                             [-3032.0, -3027.0],
                             [-3022.0, -3017.0],
                             [-3012.0, -3007.0]]},
    "RESOLUTION": 2050
}

with open('parameters.json', 'w') as json_file:
    json.dump(snd_parameters, json_file)
