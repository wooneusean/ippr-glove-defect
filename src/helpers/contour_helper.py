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
    # cv.imshow("latex_extracted", latex_extracted)

    latex_contours, hierarchy = cv.findContours(
        latex_extracted,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    # cv.imshow("latex_extracted", latex_extracted)
    largest_contour = None
    largest_contour_area = -1
    for i, cnt in enumerate(latex_contours):
        # only get parentless contours
        if hierarchy[0][i][3] != -1:
            continue

        current_contour_area = cv.contourArea(cnt)

        # cv.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        # cv.putText(
        #     img,
        #     str(current_contour_area),1
        #     tuple(cnt[0][0] + (10, 10)),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     1,
        #     cv.LINE_AA
        # )

        if largest_contour is None:
            largest_contour = cnt
            largest_contour_area = current_contour_area
        elif current_contour_area > largest_contour_area:
            largest_contour = cnt
            largest_contour_area = current_contour_area

    # cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
    # cv.imshow("latex_hole_extracted", img)
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
    stain_one_extracted = cv.Canny(stain_one_extracted, 0, 255)
    stain_one_extracted = cv.morphologyEx(
        stain_one_extracted,
        cv.MORPH_CLOSE,
        strel
    )
    # cv.imshow("stain_one_extracted", stain_one_extracted)

    # Black marker stains
    stain_two_lower = np.array([35, 125, 105])
    stain_two_higher = np.array([85, 135, 125])
    # stain_two_lower = np.array([40, 115, 100])
    # stain_two_higher = np.array([85, 135, 120])

    stain_two_extracted = cv.inRange(
        img_lab,
        stain_two_lower,
        stain_two_higher
    )
    stain_two_extracted = cv.morphologyEx(
        stain_two_extracted,
        cv.MORPH_CLOSE,
        strel
    )
    stain_two_extracted = cv.erode(
        stain_two_extracted,
        strel,
        iterations=1
    )
    stain_two_extracted = cv.dilate(
        stain_two_extracted,
        strel,
        iterations=3
    )
    # cv.imshow("stain_two_extracted", stain_two_extracted)
    stain_two_extracted = cv.Canny(stain_two_extracted, 0, 255)
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
    # cv.imshow("stain_combined_extracted", img)
    return stain_contours

# if __name__ == "__main__":
#     img = cv.imread("img/blue_glove_hole_5.jpg")
#     cv.imshow("img", img)
#     find_latex_contour(img)
#     cv.waitKey(0)
def find_oven_contours(img):
    oven_lower = np.array([220, 126, 0])
    oven_upper = np.array([255, 132, 130])
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    oven_extracted = cv.inRange(img_lab, oven_lower, oven_upper)
    cv.bitwise_not(oven_extracted, oven_extracted)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # oven_extracted = cv.morphologyEx(oven_extracted, cv.MORPH_CLOSE, kernel)

    # cv.imshow("oven_extracted", oven_extracted)

    oven_contours, hierarchy = cv.findContours(
        oven_extracted,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )

    largest_contour = None
    largest_contour_area = -1
    for i, contour in enumerate(oven_contours):
        if hierarchy[0][i][3] != -1:
            continue

        current_contour_area = cv.contourArea(contour)

        if largest_contour_area is None:
            largest_contour_area = current_contour_area
        elif current_contour_area > largest_contour_area:
            largest_contour = contour
            largest_contour_area = current_contour_area

    # cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 4)
    # cv.imshow("oven_hole_extracted", img)
    # cv.waitKey(0)

    return largest_contour

def find_frosting_contour(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    frosting_lower = np.array([40, 120, 50])
    frosting_upper = np.array([205, 175, 105])
    frosting_extracted = cv.inRange(img_lab, frosting_lower, frosting_upper)

    # cv.imshow("frosting_extracted", frosting_extracted)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # frosting_closed = cv.morphologyEx(frosting_extracted, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    frosting_dilated = cv.dilate(frosting_extracted, kernel, iterations=2)

    # cv.imshow("frosting_dilated", frosting_dilated)

    contours, hierarchy = cv.findContours(
        frosting_dilated,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )

    new_contours = []
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area > 75:
            new_contours.append(contour)

    # cv.drawContours(img, contours, -1, (255, 0, 0), 3)
    # cv.imshow("contours", img)
    # cv.waitKey(0)

    return new_contours


def find_burn_contour(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    burn_lower = np.array([0, 0, 0])
    burn_upper = np.array([25, 255, 255])
    burn_extracted = cv.inRange(img_lab, burn_lower, burn_upper)

    # cv.imshow("burn_extracted", burn_extracted)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    burn_closed = cv.morphologyEx(burn_extracted, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(
        burn_closed,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )

    new_contours = []
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area > 1300:
            new_contours.append(contour)

    # cv.drawContours(img, new_contours, -1, (255, 0, 0), 3)
    # cv.imshow("contours", img)
    # cv.waitKey(0)

    return new_contours


def find_flour_contour(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    flour_lower = np.array([160, 115, 110])
    flour_upper = np.array([255, 130, 150])
    flour_extracted = cv.inRange(img_lab, flour_lower, flour_upper)

    # cv.imshow("flour_extracted", flour_extracted)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    flour_morph = cv.dilate(flour_extracted, kernel, iterations=4)
    # flour_morph = cv.erode(flour_morph, kernel, iterations=2)

    # cv.imshow("flour_closed", flour_morph)

    contours, hierarchy = cv.findContours(
        flour_morph,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE
    )


    new_contours = []
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area > 1000 and contour_area < 10000:
            print(contour_area)
            new_contours.append(contour)

    # cv.drawContours(img, new_contours, -1, (255, 0, 0), 3)
    # cv.imshow("contours", img)
    # cv.waitKey(0)

    return new_contours



# img = cv.imread("../img/oven_burn_1.png")
# img = cv.resize(img, (500, 500))
# find_oven_contours(img)
# find_frosting_contour(img)
# find_burn_contour(img)
# find_flour_contour(img)
