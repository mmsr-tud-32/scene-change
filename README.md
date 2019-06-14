# MMSR-TUD-32 Scene Change

## Steps

### Finding the perspective

For both the foreground and background images.

1. Resize
1. Hough transform to detect lines
1. Sample lines
1. Find intersections

### Align & scale foreground

The horizon of both images should be sufficiently close to each other in order to create a believable image.

### Fix white balance

### Sources
The code in this repo is originally inspired by the work performed in [this repo](https://github.com/SZanlongo/vanishing-point-detection), but has been heavily edited in the process. As such we opted not to fork the repository, as the script no longer reflects the goal of the original script.
