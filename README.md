# ifm_GUI
Developmental code for continuous acquisition of interferometer absorption images and automatic population of fringes.

Run ifm_GUI.py and navigate to Acquisition Page. Image controls (rotation angle, Gaussian blur kernel size, pixel size, etc) are available as a text entry field. Below are checkboxes for what the program fits to, i.e. the full image region and/or ports for both x and y axes of integration. 

To begin, click "Select First Image" and open the image file. The image should populate and you will be prompted to select the region where the atoms are in the image. Then you will be asked to define regions for the two output ports of the interferometer. To select a region, simply click and drag on the window. Pressing ENTER saves the bounds.

After the first image is populated, pressing START initializes a loop that continuously monitors for new images in the specified directory (this might be broken though... works for sure if in main dir with main file). You can pause aquisition at any time by pressing PAUSE.

Each loaded image shows the image and the integrated optical densities in subplots to the right and below the main image. To view the Gaussian fits, select the image from the dropdown menu and click SHOW FITS. To change the bounding boxes, click CHANGE BOUNDS, and reselect the port bounds in the same manner as the first image. There is a known bug where the new fit params aren't handled well, will fix in upcoming versions. 

One can also flag an image and enter the assosiated phase with each image with the entry fields below the image frame. You need to click UPDATE for the changes to save. If successful, the terminal should reflect the changes.

To view the fringe from the loaded images, click SHOW FRINGE. Flagged datapoints are in red. The top subplot shows the raw image number for the various channels that have been fit to. The bottom shows the asymmetry in the number between port one and port two. Again, flagged images are displayed in red. On either plot, clicking a datapoint closes the fringe window and shows the clicked image. One can then change the bounds, check the fits, or update a flag for said image. 

This commit is Version 1.0.0 so expect bugs, especially when operating on Windows. Updates will come as bugs and functionality is fixed/added.
