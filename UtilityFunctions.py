import math
import numpy as np


def hue_to_rgb(p, q, t):
    if t < 0.0:
        t += 1.0
    if t > 1.0:
        t -= 1.0
    if t < 1.0 / 6.0:
        return p + (q - p) * 6.0 * t
    if t < 1.0 / 2.0:
        return q
    if t < 2.0 / 3.0:
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0
    return p


def v_to_255(v):
    return int(min(255, 256 * v))


def hsl_to_rgb(hsl):
    h, s, l = hsl
    _H, _S, _L = h / 60.0, s / 100.0, l / 100.0
    _H /= 6.0  # H from [0 ... 6] to [0 ... 1]
    if _S == 0:
        r = g = b = 1
    else:
        q = l < 0.5 if l * (1 + s) else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1.0 / 3.0)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1.0 / 3.0)
    return v_to_255(r), v_to_255(g), v_to_255(b)


def rgb_to_hsl(rgb):
    r, g, b = rgb
    mv_RGB = 255.0
    _R, _G, _B = r / mv_RGB, g / mv_RGB, b / mv_RGB
    _Min = min(min(_R, _G), _B)
    _Max = max(max(_R, _G), _B)
    _Delta = _Max - _Min
    _H, _S, _L = 0.0, 0.0, (_Max + _Min) / 2.0

    if _Delta != 0:
        if _L < 0.5:
            _S = float(_Delta / (_Max + _Min))
        else:
            _S = float(_Delta / (2.0 - _Max - _Min))

        if _R == _Max:
            _H = (_G - _B) / _Delta
        elif _G == _Max:
            _H = 2.0 + (_B - _R) / _Delta
        elif _B == _Max:
            _H = 4.0 + (_R - _G) / _Delta
    # _H = [0 ... 6]*60 degree-- _S = [0 ... 1]*100 percent-- _L =  [0 ... 1]*100 percent
    _H *= 60
    _S *= 100
    _L *= 100
    if _H < 0:
        _H += 360
    return _H, _S, _L


def get_ar_height(width, ratio):
    ratio_width, ratio_height = ratio
    return width * ratio_height / ratio_width


def get_ar_width(height, ratio):
    ratio_width, ratio_height = ratio
    return height * ratio_width / ratio_height


def list_to_matrix(_list, size):
    _2dList = []
    width, height = size
    for i in range(0, len(_list), width):
        _2dList.append(_list[i:i + width])
    array = np.array(_2dList)
    return array


def matrix_to_list(matrix):
    w, h = matrix.shape[1], matrix.shape[0]
    pixels = []
    array = matrix.reshape(w * h, 3)
    for i in range(w * h):
        pixels.append((int(array[i][0]), int(array[i][1]), int(array[i][2])))
    return pixels


def modify_pixel_brightness(pixel, value):
    r, g, b = pixel
    r, g, b = r + value, g + value, b + value
    if r < 0:
        r = 0
    elif r > 255:
        r = 255

    if g < 0:
        g = 0
    elif g > 255:
        g = 255

    if b < 0:
        b = 0
    elif b > 255:
        b = 255

    return r, g, b


def linear_interpolate(a, b, l_ay, l_ab):
    r_A, g_A, b_A = a
    r_B, g_B, b_B = b
    r_Y = int(r_A + l_ay * (r_B - r_A) / float(l_ab))
    g_Y = int(g_A + l_ay * (g_B - g_A) / float(l_ab))
    b_Y = int(b_A + l_ay * (b_B - b_A) / float(l_ab))
    return r_Y, g_Y, b_Y


def resize_bilinear(pixels, size, new_size):
    width, height = size
    new_width, new_height = new_size
    new_pixels = []
    x_ratio = float(width - 1) / new_width
    y_ratio = float(height - 1) / new_height
    for i in range(new_height):
        for j in range(new_width):
            x = int(x_ratio * j)
            y = int(y_ratio * i)
            x_diff = (x_ratio * j) - x
            y_diff = (y_ratio * i) - y
            index = y * width + x
            a_r, a_g, a_b = pixels[index]
            b_r, b_g, b_b = pixels[index + 1]
            c_r, c_g, c_b = pixels[index + width]
            d_r, d_g, d_b = pixels[index + width + 1]

            y_b = a_b * (1 - x_diff) * (1 - y_diff) + b_b * x_diff * (1 - y_diff) + c_b * y_diff * (
                    1 - x_diff) + d_b * (x_diff * y_diff)
            y_g = a_g * (1 - x_diff) * (1 - y_diff) + b_g * x_diff * (1 - y_diff) + c_g * y_diff * (
                    1 - x_diff) + d_g * (x_diff * y_diff)
            y_r = a_r * (1 - x_diff) * (1 - y_diff) + b_r * x_diff * (1 - y_diff) + c_r * y_diff * (
                    1 - x_diff) + d_r * (x_diff * y_diff)
            new_pixels.append((int(y_r), int(y_g), int(y_b)))
    return new_pixels


def anti_aliasing_000(pixels):
    new_pixels = []
    black = (0, 0, 0)
    for i, pixel in enumerate(pixels):
        if pixels[i - 1] != black and pixels[i + 1] != black and pixel == black:
            new_pixels.append(linear_interpolate(pixels[i - 1], pixels[i + 1], 1, 2))
        else:
            new_pixels.append(pixel)
    return new_pixels


def rotate_aliasing(pixels, size, angle):
    arbitrary = False
    if abs(angle) != 90 or abs(angle) != 180 or abs(angle) != 270 or abs(angle) != 0:
        arbitrary = True
    matrix = list_to_matrix(pixels, size)
    width, height = size  # définir la largeur et la hauteur de l'image
    # Définir les variables les plus courantes
    angle = math.radians(angle)  # conversion de degrés en radians
    cosine = math.cos(angle)
    sine = math.sin(angle)
    # Définir la hauteur et la largeur de la nouvelle image à former
    new_height = round(abs(height * cosine) + abs(width * sine)) + 1
    new_width = round(abs(width * cosine) + abs(height * sine)) + 1
    # définir une autre variable d'image de dimensions de new_height et new_width remplie de zéros
    matrix_copy = np.zeros((new_height, new_width, matrix.shape[2]))
    # Trouver le centre de l'image autour duquel nous devons faire pivoter l'image
    original_centre_height = round(((height + 1) / 2) - 1)  # par rapport à l'image originale
    original_centre_width = round(((width + 1) / 2) - 1)  # par rapport à l'image originale
    # Trouver le centre de la nouvelle image qui sera obtenue
    new_centre_height = round(((new_height + 1) / 2) - 1)  # par rapport à la nouvelle image
    new_centre_width = round(((new_width + 1) / 2) - 1)  # par rapport à la nouvelle image

    for i in range(height):
        for j in range(width):
            # coordonnées du pixel par rapport au centre de l'image originale
            y = height - 1 - i - original_centre_height
            x = width - 1 - j - original_centre_width

            new_y = round(-x * sine + y * cosine)
            new_x = round(x * cosine + y * sine)

            '''
                puisque l'image sera tournée, le centre changera aussi,
                donc pour s'adapter à cela, nous devrons changer new_x et new_y par rapport au nouveau centre
            '''
            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x

            if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x >= 0 and new_y >= 0:
                # écriture des pixels vers la nouvelle destination dans l'image de sortie
                matrix_copy[new_y, new_x, :] = matrix[i, j, :]
    pixels = matrix_to_list(matrix_copy)
    if arbitrary:
        pixels = anti_aliasing_000(pixels)

    return pixels, (new_width, new_height)


def convert_bin(pixels):
    """
    :param pixels: list de pixel
    :return: Bool success
    """
    bin_new_pixels = []
    for pixel in pixels:
        if pixel[0] == pixel[1] == pixel[2]:
            if pixel[0] == 255:
                bin_new_pixels.append((1, 1, 1))
            elif pixel[0] == 0:
                bin_new_pixels.append((0, 0, 0))
            else:
                return False, None
        else:
            return False, None
    return True, bin_new_pixels


def convert_to_bin(matrix):
    new_matrix = np.zeros(matrix.shape, dtype=matrix.dtype)
    H, W, rgb = new_matrix.shape
    for i in range(H):
        for j in range(W):
            if matrix[i, j, 0] == matrix[i, j, 1] == matrix[i, j, 2] == 255:
                new_matrix[i, j] = (1, 1, 1)
            else:
                new_matrix[i, j] = (0, 0, 0)

    return new_matrix


def convert_from_bin(matrix):
    new_matrix = np.zeros(matrix.shape, dtype=matrix.dtype)
    H, W, rgb = new_matrix.shape
    for i in range(H):
        for j in range(W):
            if matrix[i, j, 0] == matrix[i, j, 1] == matrix[i, j, 2] == 1:
                new_matrix[i, j] = (255, 255, 255)
            else:
                new_matrix[i, j] = (0, 0, 0)

    return new_matrix


def thresholding_wv(matrix, value):
    new_matrix = np.zeros(matrix.shape, dtype=matrix.dtype)
    H, W, rgb = new_matrix.shape
    for i in range(H):
        for j in range(W):
            if int(matrix[i, j, 0] * 0.21 + matrix[i, j, 1] * 0.72 + matrix[i, j, 2] * 0.07) >= value:
                new_matrix[i, j] = (255, 255, 255)
    return new_matrix


def color_to_grey(matrix):
    new_matrix = matrix.copy()
    H, W, rgb = new_matrix.shape
    for i in range(H):
        for j in range(W):
            r, g, b = new_matrix[i, j]
            v = int(0.21 * r + 0.72 * g + 0.07 * b)
            new_matrix[i, j] = v, v, v
    return new_matrix


def remove_channels(matrix):
    new_matrix = np.zeros((matrix.shape[0], matrix.shape[1])).astype(np.uint8)
    H, W = new_matrix.shape
    for i in range(H):
        for j in range(W):
            new_matrix[i, j] = matrix[i, j, 0]
    return new_matrix


def add_channels(matrix):
    new_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 3)).astype(np.uint8)
    H, W = matrix.shape
    for i in range(H):
        for j in range(W):
            new_matrix[i, j, 0] = new_matrix[i, j, 1] = new_matrix[i, j, 2] = matrix[i, j]
    return new_matrix


def get_neighbor_there(matrix, points, point):
    x, y = point
    if matrix[x + 1, y + 1] == 1:
        if (x + 1, y + 1) not in points:
            points.append((x + 1, y + 1))
            get_neighbor_there(matrix, points, (x + 1, y + 1))
    if matrix[x + 1, y] == 1:
        if (x + 1, y) not in points:
            points.append((x + 1, y))
            get_neighbor_there(matrix, points, (x + 1, y))
    if matrix[x, y + 1] == 1:
        if (x, y + 1) not in points:
            points.append((x, y + 1))
            get_neighbor_there(matrix, points, (x, y + 1))
    if matrix[x - 1, y + 1] == 1:
        if (x - 1, y + 1) not in points:
            points.append((x - 1, y + 1))
            get_neighbor_there(matrix, points, (x - 1, y + 1))
    if matrix[x + 1, y - 1] == 1:
        if (x + 1, y - 1) not in points:
            points.append((x + 1, y - 1))
            get_neighbor_there(matrix, points, (x + 1, y - 1))
    if matrix[x - 1, y - 1] == 1:
        if (x - 1, y - 1) not in points:
            points.append((x - 1, y - 1))
            get_neighbor_there(matrix, points, (x - 1, y - 1))
    if matrix[x, y - 1] == 1:
        if (x, y - 1) not in points:
            points.append((x, y - 1))
            get_neighbor_there(matrix, points, (x, y - 1))
    if matrix[x - 1, y] == 1:
        if (x - 1, y) not in points:
            points.append((x - 1, y))
            get_neighbor_there(matrix, points, (x - 1, y))


def make_boundary(points):
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min - 2, y_min - 2, x_max + 2, y_max + 2


def not_in_Symbols(symbols, point):
    y, x = point
    for s in symbols:
        x1, y1, x2, y2 = s
        if x1 - 3 < x < x2 + 3 and y1 - 3 < y < y2 + 3:
            return False
    return True
