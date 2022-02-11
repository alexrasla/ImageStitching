# ImageStitching
This program creates panoramic images using image stitching techniques. Given an image directory, this program outputs a pano_img.jpg that stitching its images together.

# Running Program
The image directory must have at least 9 images to stitch together. If the projective image is too large to compute and creates a base image that is too small, then an error is thrown and the image is shown. To run the program, execute the command:
```
python3 stitching --image [image directory]
```
