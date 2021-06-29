from MainFunctions import *

image = Image.open('image_original.png')
pixels = list(image.getdata())
angle = 60     # Angle de Rotation   Testé ( 60 )
new_pixels, new_size = rotate(pixels, image.size, angle)
image_out = Image.new(image.mode, new_size)
image_out.putdata(new_pixels)
image_out.save('Rotated_image.png')
