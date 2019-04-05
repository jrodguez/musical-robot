# Use Case 1
## Determine phase transition (or energy shift) temperature of materials in a 96 well plate as the plate is heated.

Components:
```
* Determine edges of the well plate
* Tile well plate
* Read temperature at origin of tiled image
* Loop through all 96 images for 1 frame
* Loop through all frames desired for video
* Plot the temperature v time for each well plate across the entire video
* Using slope software determine temperature where the phase transition occurred
* Organize data in a good way (idk how rn)
```

# Use Case 2
## Allow the user to request a specific well plate or to modify properties of well plate

Components:
```
* Prompts for well plate number and new well plate dimensions
* Run normal process for temperature loops using new dimensions and only for specific well plate number
```
