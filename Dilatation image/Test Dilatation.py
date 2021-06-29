from MainFunctions import *
from UtilityFunctions import *

image = Image.open('image_original.png')
matrix = np.asarray(image).reshape(image.height, image.width, 3)
kernel = None  # Masque par default est  [[010][111][010]]
threshold = 127  # un seuil qui convertira une image en une image binaire de deux valeurs 0 et 255
new_matrix = convert_from_bin(
    add_channels(
        image_dilation(
            remove_channels(
                convert_to_bin(
                    thresholding_wv(matrix, threshold
                                    )
                )
            )
        )
    )
)
image_out = Image.fromarray(new_matrix)
image_out.save('Dilation.png')
