from MainFunctions import *

image = Image.open('image_original.png')
image_out = Image.new(image.mode, image.size)
pixels = list(image.getdata())
value = -100     # Valeur de changement de luminosité   Testé (-100 , +50 )
new_pixels = modify_image_brightness(pixels, value)
image_out.putdata(new_pixels)
if value >= 0:
    image_out.save('Bright_image.png')
else:
    image_out.save('Dark_image.png')
