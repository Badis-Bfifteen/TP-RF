from MainFunctions import *

image = Image.open("image_original.png")
symbol_extraction(image, threshold=127)
print("Voir Dossier Symbols")
