# 2023-image-processing
A project for the image processing course at BME.

## Downloading images

Just run download_images.py, it will get all the images from the urls.

It will also create a new database file, with the file names.

Note: In the database there are duplicated rows.

**TODO:**
- need some way to see the accuracy of the model

## Extract license plates from the raw images

Run license_plate_extraction.py

It is currently not the best solution, but it's acceptable.

## Segmenting and transforming license plates

**TODO:**
- transform license plate for the same horizontal orientation
- scale the images, so they are the same dimension
- segment license plate into the images of each character

## Character recognition

**TODO:**
- create neural network
- train and test
etc.