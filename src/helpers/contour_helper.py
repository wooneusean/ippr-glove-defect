import cv2 as cv
import numpy as np


def find_skin_contours(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # skin_lower = np.array([120, 130, 130])
    # skin_higher = np.array([255, 255, 255])
    skin_lower = np.array([45, 105, 85])
    skin_higher = np.array([180, 160, 110])
    skin_extracted = cv.inRange(img_lab, skin_lower, skin_higher)

    # floodfill from point (0, 0)
    h, w = skin_extracted.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(skin_extracted, mask, (0, 0), 255)
    # invert floodfilled image
    skin_extracted = cv.bitwise_not(skin_extracted)

    # cv.imshow("skin_extracted", skin_extracted)

    a_erode = cv.erode(skin_extracted, None, iterations=2)
    # cv.imshow("a_erode", a_erode)

    a_dilate = cv.dilate(a_erode, None, iterations=2)
    # cv.imshow("a_dilate", a_dilate)

    a_fill = cv.morphologyEx(a_dilate, cv.MORPH_CLOSE,
                             np.ones((7, 7), dtype=np.uint8))
    # cv.imshow("a_fill", a_fill)

    a_canny = cv.Canny(a_fill, 0, 255)
    # cv.imshow("a_hole_canny", a_canny)

    # find contours
    a_contours, _ = cv.findContours(
        a_canny,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    # cv.drawContours(img, a_contours, -1, (0, 255, 0), 3)
    # cv.imshow("img", img)

    return a_contours


def find_latex_contour(img):
    latex_lower = np.array([0, 0, 0])
    latex_higher = np.array([255, 120, 255])
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    latex_extracted = cv.inRange(img_hsv, latex_lower, latex_higher)
    latex_extracted = cv.bitwise_not(latex_extracted, latex_extracted)
    latex_extracted = cv.erode(latex_extracted, None, iterations=1)
    latex_extracted = cv.dilate(latex_extracted, None, iterations=1)

    latex_contours, hierarchy = cv.findContours(
        latex_extracted,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    # cv.imshow("latex_extracted", latex_extracted)
    latex_extracted = cv.bitwise_not(latex_extracted)
    largest_contour = None
    largest_contour_area = -1
    for i, cnt in enumerate(latex_contours):
        # only get parentless contours
        if hierarchy[0][i][3] != -1:
            continue

        current_contour_area = cv.contourArea(cnt)

        if largest_contour is None:
            largest_contour = cnt
        elif current_contour_area > largest_contour_area:
            largest_contour = cnt
            largest_contour_area = cv.contourArea(cnt)

    # cv.drawContours(img_copy, [largest_contour], -1, (0, 255, 0), 2)

    latex_extracted = cv.bitwise_not(latex_extracted)
    # cv.imshow("latex_hole_extracted", img_copy)
    return largest_contour


def find_stain_contours(img):
    # Ballpoint pen stains
    stain_one_lower = np.array([115, 125, 90])
    stain_one_higher = np.array([145, 155, 105])
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    strel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    stain_one_extracted = cv.inRange(
        img_lab,
        stain_one_lower,
        stain_one_higher
    )
    stain_one_extracted = cv.morphologyEx(
        stain_one_extracted,
        cv.MORPH_CLOSE,
        strel
    )
    # cv.imshow("stain_one_extracted", stain_one_extracted)
    stain_one_extracted = cv.erode(
        stain_one_extracted,
        strel,
        iterations=1
    )
    stain_one_extracted = cv.dilate(
        stain_one_extracted,
        strel,
        iterations=3
    )

    # Black marker stains
    stain_two_lower =   np.array([65, 115, 110])
    stain_two_higher =  np.array([90, 135, 125])

    stain_two_extracted = cv.inRange(
        img_lab,
        stain_two_lower,
        stain_two_higher
    )

    # stain_two_extracted = cv.morphologyEx(
    #     stain_two_extracted,
    #     cv.MORPH_CLOSE,
    #     strel
    # )
    stain_two_extracted = cv.erode(
        stain_two_extracted,
        strel,
        iterations=1
    )
    # cv.imshow("stain_two_extracted", stain_two_extracted)
    stain_two_extracted = cv.dilate(
        stain_two_extracted,
        strel,
        iterations=3
    )
    # cv.imshow("stain_two_extracted", stain_two_extracted)

    stain_combined_extracted = cv.bitwise_or(
        stain_one_extracted,
        stain_two_extracted
    )

    stain_contours, hierarchy = cv.findContours(
        stain_combined_extracted,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    # cv.drawContours(img, stain_contours, -1, (0, 255, 0), 2)
    # cv.imshow("stain_combined_extracted", stain_combined_extracted)
    return stain_contours


# find_stain_contours(cv.imread("img/blue_glove_stain_1.jpg"))
# cv.waitKey(0)

def find_cloth_contours(img):
    cloth_lower = np.array([])
    cloth_higher = np.array([])
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cloth_extracted = cv.inRange(img_hsv, cloth_lower, cloth_higher)
    cloth_extracted = cv.bitwise_not
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cloth_extracted = cv.morphologyEx(cloth_extracted, cv.MORPH_OPEN, kernel)

find_cloth_contours(cv.imread("../img/black_fabric.png"))