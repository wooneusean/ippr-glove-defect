import cv2 as cv

def resize_image(image, percentage):

    old_width = image.shape[0]
    old_height = image.shape[1]
    new_width = int(old_width*percentage)
    new_height = int(old_height*percentage)
    new_image = cv.resize(image, (new_width, new_height))
    return new_image