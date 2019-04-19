# Use Case 1
## Determine phase transition (or energy shift) temperature of materials in a 96 well plate as the plate is heated.

Components:
```
* Determine edges of the well plate
* Tile well plate
* Read temperature at origin of tiled image
* Loop through all 96 images for 1 frame
* Loop through all frames desired for video
* Plot the temperature of solvent v temperature of well plate close to solvent across the entire video
* Using slope software determine temperature where the phase transition occurred
* Organize data in a good way (idk how rn)
* Output of the melting point temperatures in a table
```

# Use Case 2
## Allow the user to request a specific well plate or to modify properties of well plate

Components:
```
* Prompts for well plate number and new well plate dimensions
* Run normal process for temperature loops using new dimensions and only for specific well plate number
```

# Use Case 3
## Using a machine learning algorithm, collected data, and validated data, be able to predict the melting point of a given solvent.

Components
```
* Working module (use case 1)
* Data from running with validation from DSC.
* Machine learning algorithm between component makeup, ratios, melting temp, (maybe - molecular structure)
* Softer: algorithm can predict the desired frames
```

# Use Case 4
## Be able to determine when there is a failure in the sample (burning or incorrectly measured or the like)

Components
```
* working module (use case 1)
* machine learning model (?) that can estimate the approximate change points
* OR some buffers on expected data - smooths and marks jumps, decreases, or unexpected spikes
```
