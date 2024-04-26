from google.colab.patches import cv2_imshow
import numpy as np
import cv2

def haar_transform(image):
    rows, cols = image.shape
    transformed_image = np.zeros((rows, cols))

    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            if i + 1 < rows and j + 1 < cols:
                a = (image[i, j] + image[i, j + 1] + image[i + 1, j] + image[i + 1, j + 1]) / 4
                b = (image[i, j] + image[i, j + 1] - image[i + 1, j] - image[i + 1, j + 1]) / 2
                c = (image[i, j] - image[i, j + 1] + image[i + 1, j] - image[i + 1, j + 1]) / 2
                d = (image[i, j] - image[i, j + 1] - image[i + 1, j] + image[i + 1, j + 1]) / 4
                transformed_image[i, j] = a
                transformed_image[i, j + 1] = b
                transformed_image[i + 1, j] = c
                transformed_image[i + 1, j + 1] = d
    return transformed_image, image - transformed_image


def multi_level_haar_transform(image, levels):
    vertical_coefficients = []
    horizontal_coefficients = []
    diagonal_coefficients = []
    approximation_coefficients = []

    for _ in range(levels):
        transformed_image, difference_coefficients = haar_transform(image)
        vertical = transformed_image.copy()
        vertical[:, 1::2] = 0
        horizontal = transformed_image.copy()
        horizontal[1::2, :] = 0
        diagonal = transformed_image - vertical - horizontal

        vertical_coefficients.append(vertical)
        horizontal_coefficients.append(horizontal)
        diagonal_coefficients.append(diagonal)
        approximation_coefficients.append(difference_coefficients)
        image = difference_coefficients

    return vertical_coefficients, horizontal_coefficients, diagonal_coefficients, approximation_coefficients

def inverse_haar(vertical_coeffs, horizontal_coeffs, diagonal_coeffs, approx_coeffs):
    reconstructed_image = np.zeros_like(approx_coeffs[0])

    for i in range(len(approx_coeffs)):
        reconstructed_image += approx_coeffs[i]
        reconstructed_image += vertical_coeffs[i]
        reconstructed_image += horizontal_coeffs[i]
        reconstructed_image += diagonal_coeffs[i]
    return reconstructed_image

def threshold_coefficients(coefficients, threshold):
    thresholded_coeffs = coefficients.copy()
    thresholded_coeffs[np.abs(thresholded_coeffs) < threshold] = 0
    return thresholded_coeffs

image = cv2.imread('/content/cricket.jpg', cv2.IMREAD_GRAYSCALE)
levels = 1
vertical_coeffs, horizontal_coeffs, diagonal_coeffs, approx_coeffs = multi_level_haar_transform(image.astype(np.float32), levels)

for i in range(len(approx_coeffs)):
    approx_coeffs[i] = threshold_coefficients(approx_coeffs[i], threshold=17)


reconstructed_image = inverse_haar(vertical_coeffs, horizontal_coeffs, diagonal_coeffs, approx_coeffs)
cv2_imshow(image)
for i in range(levels):
    cv2_imshow(np.uint8(approx_coeffs[i]))
    cv2_imshow(np.uint8(vertical_coeffs[i]))
    cv2_imshow(np.uint8(horizontal_coeffs[i]))
    cv2_imshow(np.uint8(diagonal_coeffs[i]))
cv2_imshow(reconstructed_image)






