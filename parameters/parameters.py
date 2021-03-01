# parameters.py

# This script is meant to be used for writting the geometry parameters of your SND@LHC detector.
# Once you have written the parameters, run "parameters.py" and it will write the parameters values in "parameters.json".

#Each configuration is associated to a dataset:
#'DS5' is PG configuration (4 planes)
#'SNDatLHC' is SNDatLHC detector configuration (5 planes)

#"END_OF_BRICK": z position of the front of the detector
#"X_min":        x position of the lowest point of the Scifi planes
#"X_max":        x position of the highest point of the Scifi planes
#"Y_min":        y position of the lowest point of the Scifi planes
#"Y_max":        y position of the highest point of the Scifi planes
#"TT_POSITIONS": z position of the Scifi planes (front and end of each plane)

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
