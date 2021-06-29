from MainFunctions import *

image = Image.open('image_original.png')
pixels = list(image.getdata())
factor = 0.5     # Facteur de Redimensionnement   Testé ( 0.5 , 2 )
new_pixels, new_size = resize(pixels, image.size, factor)
image_out = Image.new(image.mode, new_size)
image_out.putdata(new_pixels)
if abs(factor) > 1:
    image_out.save('Big_image.png')
elif 0 < abs(factor) < 1:
    image_out.save('Small_image.png')
