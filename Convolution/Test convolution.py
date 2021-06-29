from MainFunctions import *

image = Image.open('image_original.png')
pixels = list(image.getdata())
kernel = None  # Masque par default est 1/9 (3 x 3) un flou gaussien
new_matrix = apply_convolution(pixels, image.size, kernel)
image_out = Image.fromarray(new_matrix)
if kernel is None:
    image_out.save('Gaussian blur.png')
else:
    image_out.save('Convolution.png')
