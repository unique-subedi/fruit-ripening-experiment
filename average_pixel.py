import numpy as np
import cv2
import sys
import os


def find_avg_pixel(image, edge_detect=True):
    '''
    Given a filepath to a banana image, find the average color of the banana. Can use edge detection to find the edge
    of the banana, if not then will need to manually black out all parts of the picture that is not the banana. The 
     averaging does not take into account rgb values that sum to 0.
    Arguments:
        image: filepath to the banana image
        edge_detect: Whether or not we should use Canny edge detect to find the edge of the banana. If so, sets all 
            non-banana pixels to (0,0,0) and crops to the smallest rectangle containing the banana. Saves the new
            image as [image_name]_cropped_and_blacked.png and runs the rest of the function on this image.

    Returns:
        array[3]: average red, green, and blue values of the non-black portions of the image ran.
    '''
    #finding where image was saved and where to save to
    slash_idx = image.rfind("/")
    if slash_idx == -1:
        path = "./"
        name = image
    else:
        path = image[:slash_idx+1]
        name = image[slash_idx+1:]

    #reading image
    img = cv2.imread(image)
    img_for_avg = img

    #if edge_detect 
    if edge_detect:
        #doing edge dectection
        blurred = cv2.blur(img, (3,3))
        canny = cv2.Canny(blurred, 20, 100)

        #cropping image based on edges found
        pts = np.argwhere(canny>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)
        cropped = img[y1:y2, x1:x2]
        canny_cropped = canny[y1:y2, x1:x2]
        
        #setting all pixels outside of edges (hopefully banana edges) to be pure black
        black = [0,0,0]
        for y in range(cropped.shape[0]):
            for x in range(cropped.shape[1]):
                if canny_cropped[y,x] == 0:
                    cropped[y,x] = black
                else:
                    break
            for x in range(cropped.shape[1]-1,-1,-1):
                if canny_cropped[y,x] == 0:
                    cropped[y,x] = black
                else:
                    break

        #saving image for manual inspection if needed
        cv2.imwrite(f"{image[:-4]}_cropped_and_blacked.png", cropped)
        img_for_avg = cropped

    #getting average pixel value
    rgb = [0,0,0]
    total = 0
    for y in range(img_for_avg.shape[0]):
        for x in range(img_for_avg.shape[1]):
            bgr = img_for_avg[y,x]
            if sum(bgr) >= 1:
                rgb[0] += bgr[2]
                rgb[1] += bgr[1]
                rgb[2] += bgr[0]
                total += 1
    return np.array(rgb)/total


#to use run python average_pixel.py [T/F] arg1 arg2 ...
#where arg1 can be a filepath to an image or a directory. If a directory, it will run find_avg_pixel on all .png, .jpg,
#or .webp files in the directory. The T or F is for whether or not to use edge_detection or just run the pictures as is.
#example: python average_pixel.py T ./directions/to/nearest/grocery/store/
#example: python average_pixel.py F ./that_is_bananas.jpg ./b-a-n-a-n-a-s.png ./aint-no-hollaback-girl/
if __name__ == "__main__":
    argv = sys.argv
    edge = argv[1]
    if edge[0] in ["T", "t"]:
        edge = True
    else:
        edge = False
    for arg in argv[2:]:
        if arg[len(arg)-1] == "/":  #directory
            for name in os.listdir(arg):
                if name[-3:] not in ["ebp", "jpg", "png"]:
                    continue
                image = arg + name
                avg_pix = find_avg_pixel(image, edge_detect=edge)
                print(f"{image} average color: {avg_pix}")
        else:
            avg_pix = find_avg_pixel(arg, edge_detect=edge)
            print(f"{arg} average color: {avg_pix}")
