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
#"RESOLUTION":   spatial resolution of the tracker in micrometer

import json

snd_parameters = {
    "SNDatLHC": {"SciFi_tracker": {
                      "RESOLUTION": 2050,
                      "END_OF_BRICK": -28.5,
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
                 "Mu_tracker_upstream":{
                      "RESOLUTION": 66200,
                      "END_OF_BRICK": -28.5,
                      "X_min": -2.3,
                      "X_max": 59.3,
                      "Y_min": 3.2,
                      "Y_max": 69.4,
                      "TT_POSITIONS": [[47.4, 49.5],
                                       [69.4, 71.5],
                                       [91.4, 93.5],
                                       [113.4, 115.5],
                                       [135.4, 137.5]]
                 },
                 "Mu_tracker_downstream":{
                      "RESOLUTION": 10000,
                      "END_OF_BRICK": -28.5,
                      "X_min": -2.3,
                      "X_max": 59.3,
                      "Y_min": 3.2,
                      "Y_max": 69.4,
                      "TT_POSITIONS": [[157.4, 159.5],
                                       [179.4, 181.5],
                                       [201.4, 203.5]]
                 } 
                }
}

with open('parameters.json', 'w') as json_file:
    json.dump(snd_parameters, json_file)
