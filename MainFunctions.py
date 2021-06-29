import os
import shutil

import numpy
from PIL import Image

import UtilityFunctions
import matplotlib.pyplot as plt
from UtilityFunctions import *


def show_histogram(pic):
    x = numpy.asarray(pic).reshape(pic.width * pic.height, 3)
    colors = ['red', 'green', 'blue']
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(prop={'size': 10})

    plt.hist(x[:, :], bins=256, histtype='bar', color=colors, label=colors)
    plt.show()
    plt.hist(x[:, 0], bins=256, histtype='bar', color='red', label=colors)
    plt.show()
    plt.hist(x[:, 1], bins=256, histtype='bar', color='green', label=colors)
    plt.show()
    plt.hist(x[:, 2], bins=256, histtype='bar', color='blue', label=colors)
    plt.show()


def modify_image_brightness(pixels, value):
    new_pixels = []
    for pixel in pixels:
        new_pixels.append(modify_pixel_brightness(pixel, value))
    return new_pixels


def apply_convolution(pixels, size, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel = kernel * (1 / 9)
    matrix = UtilityFunctions.list_to_matrix(pixels, size)
    H, W, rgb = matrix.shape
    kernel_H, kernel_W = kernel.shape
    new_matrix = numpy.zeros((H, W, rgb), dtype=numpy.uint8)
    for i in range(H):
        for j in range(W):
            r, g, b = 0, 0, 0
            for x in range(kernel_H):
                for y in range(kernel_W):
                    p = i - kernel_H // 2 + x
                    q = j - kernel_W // 2 + y
                    if 0 <= p < H and 0 <= q < W:
                        r = r + int(kernel[x][y] * matrix[p][q][0])
                        g = g + int(kernel[x][y] * matrix[p][q][1])
                        b = b + int(kernel[x][y] * matrix[p][q][2])

            new_matrix[i][j] = r, g, b
    return new_matrix


def resize(pixels, old_size, factor):
    w, h = old_size
    if factor == 0:
        factor = 1
    new_size = (int(w * abs(factor)), int(h * abs(factor)))
    if old_size == new_size:
        return pixels.copy(), old_size
    else:
        return resize_bilinear(pixels, old_size, new_size), new_size


def rotate(pixels, size, angle):
    angle = -(angle % 360)
    if angle == 0:
        return pixels.copy(), size
    else:
        return rotate_aliasing(pixels, size, angle)


def image_dilation(matrix, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel[0, 0] = kernel[2, 2] = kernel[0, 2] = kernel[2, 0] = 0
    H, W = matrix.shape
    kernel_H, kernel_W = kernel.shape
    new_matrix = numpy.zeros((H, W), dtype=numpy.uint8)
    for i in range(H):
        for j in range(W):
            dilated = False
            for x in range(kernel_H):
                for y in range(kernel_W):
                    p = i - kernel_H // 2 + x
                    q = j - kernel_W // 2 + y
                    if 0 <= p < H and 0 <= q < W:
                        dilated = kernel[x][y] and matrix[p][q]
                        if dilated:
                            break
                if dilated:
                    break
            new_matrix[i][j] = int(dilated)
    return new_matrix


def image_erosion(matrix, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel[0, 0] = kernel[2, 2] = kernel[0, 2] = kernel[2, 0] = 0
    H, W = matrix.shape
    matrix = -(matrix - 1)
    kernel_H, kernel_W = kernel.shape
    new_matrix = numpy.ones((H, W), dtype=numpy.uint8)
    for i in range(H):
        for j in range(W):
            if matrix[i][j] == 0:
                eroded = False
                for x in range(kernel_H):
                    for y in range(kernel_W):
                        p = i - kernel_H // 2 + x
                        q = j - kernel_W // 2 + y
                        if 0 <= p < H and 0 <= q < W:
                            if kernel[x][y] and matrix[p][q]:
                                new_matrix[i][j] = 1
                                eroded = True
                                break
                        else:
                            if kernel[x][y]:
                                new_matrix[i][j] = 1
                                eroded = True
                                break
                    if eroded:
                        break
                if not eroded:
                    new_matrix[i][j] = matrix[i][j]
    return -(new_matrix - 1)


def image_Opening(matrix, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel[0, 0] = kernel[2, 2] = kernel[0, 2] = kernel[2, 0] = 0
    matrix_e = image_erosion(matrix, kernel)
    return image_dilation(matrix_e, kernel)


def image_Closing(matrix, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel[0, 0] = kernel[2, 2] = kernel[0, 2] = kernel[2, 0] = 0
    matrix_d = image_dilation(matrix, kernel)
    return image_erosion(matrix_d, kernel)


def edge_extraction(matrix, kernel=None):
    if kernel is None:
        kernel = numpy.ones((3, 3), dtype=numpy.uint8)
        kernel[0, 0] = kernel[2, 2] = kernel[0, 2] = kernel[2, 0] = 0
    matrix_r = matrix.copy()
    H, W = matrix.shape
    matrix_e = image_erosion(matrix, kernel)
    for i in range(H):
        for j in range(W):
            if matrix_r[i, j] == matrix_e[i, j] == 1:
                matrix_r[i, j] = 0
    x = np.zeros(matrix.shape)
    x[1:-1, 1:-1] = matrix_r[1:-1, 1:-1]
    matrix_r = x
    return matrix_r


def symbol_extraction(picture, threshold):
    """
    :param picture: Objet Image de la libraries pillow
    :param threshold: un seuil qui convertira une image en une image binaire de deux valeurs 0 et 255
    la procédure extraira tous les symboles de l'image et les enregistrera séparément en tant qu'images
    """
    matrix = np.asarray(picture).reshape(picture.height, picture.width, 3)  # convertire l'image en matrice
    matrix = edge_extraction(
        remove_channels(
            convert_to_bin(
                thresholding_wv(matrix, threshold)  # l'utilisation d'un seuil (threshold) convertira une
                # image en une image binaire de deux valeurs 0 et 255 ( l'image originale a trois canaux la
                # nouvelle image aura les mêmes trois canaux RVB mais ils auront la même valeur)
            )  # convertir 255 en 1
        )  # supprimer les canaux redondants RVB
    )  # Extraction des contours
    symbols = []
    inp = []
    H, W = matrix.shape
    for i in range(H):
        for j in range(W):
            if matrix[i, j] == 1 and not_in_Symbols(symbols, (i, j)):
                points = [(i, j)]
                get_neighbor_there(matrix, points, (i, j))
                symbols.append(make_boundary(points))
                inp.append(points)
    try:
        os.mkdir("Symbols")
    except OSError as error:
        print(error)
        for filename in os.listdir("Symbols"):
            file_path = os.path.join("Symbols", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    for i, symbol in enumerate(symbols):
        x1, y1, x2, y2 = symbol
        img = picture.crop((x1 + 2, y1 + 2, x2 + 2, y2 + 2))
        img.save("Symbols/symbol_" + str(i + 1) + ".png")
