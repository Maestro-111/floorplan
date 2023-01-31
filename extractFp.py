import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import time
import pytesseract
#import easyocr
import keras_ocr


BOX_LIMITER = 4
BOX_MIN_NUMBER = 2
BOX_MAX_NUMBER = 8
# tuplify

# reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
# ocr_reader = easyocr.Reader(['en'])
ocr_pipeline = keras_ocr.pipeline.Pipeline()


def tup(point):
    return (point[0], point[1])


def isOverlapping1D(xmin1, xmax1, xmin2, xmax2):
    return xmax1 >= xmin2 and xmax2 >= xmin1

# returns true if the two boxes overlap


def overlap(source, target):
    # unpack points
    tl1, br1, _wh1 = source
    tl2, br2, _wh2 = target

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True

# returns all overlapping boxes


def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps


# filename1 = '/Users/fred/Documents/RealMaster/floorplan/Burano_image_1_1.jpeg'
filename1 = '/Users/fred/work/floorplan/images/IMG_6892.JPG'


def getImageAndBoxes(filename):
    img = cv2.imread(filename)
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    imgChannels = img.shape[2]
    orig = np.copy(img)
    blue, green, red = cv2.split(img)

    def medianCanny(img, thresh1, thresh2):
        median = np.median(img)
        img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
        return img

    blue_edges = medianCanny(blue, 0, 1)
    green_edges = medianCanny(green, 0, 1)
    red_edges = medianCanny(red, 0, 1)

    edges = blue_edges | green_edges | red_edges

    # I'm using OpenCV 3.4. This returns (contours, hierarchy) in OpenCV 2 and 4
    #contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # go through the contours and save the box edges
    max_area = imgHeight * imgWidth * 0.9
    allBoxes = []
    boxes = []  # each element is [[top-left], [bottom-right]]
    hierarchy = hierarchy[0]
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            allBoxes.append([[x, y], [x+w, y+h], [w, h]])
            # filter out excessively large boxes, and boxes with extreme aspect ratios
            if w*h > max_area:
                print(f'area too large {w*h} {max_area} {x} {y} {w} {h}')
                continue
            if (h > imgHeight * 0.9 and (x == 0 or (x + w) >= imgWidth)) or \
               (w > imgWidth * 0.9 and (y == 0 or (y + w) >= imgHeight)):
                print(f'narrow box close to edges {x} {y} {w} {h}')
                continue
            boxes.append([[x, y], [x+w, y+h], [w, h]])

    # go through the boxes and start merging
    merge_margin = 15

    # this is gonna take a long time
    finished = False
    highlight = [[0, 0], [1, 1]]
    points = [[[0, 0]]]
    while not finished:
        # set end con
        finished = True

        # check progress
        print("Len Boxes: " + str(len(boxes)))
        if len(boxes) < BOX_MAX_NUMBER:
            print(str(boxes))
        if len(boxes) <= BOX_MIN_NUMBER:
            break

        # draw boxes # comment this section out to run faster
        copy = np.copy(orig)
        for box in boxes:
            cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0, 200, 0), 1)
        cv2.rectangle(copy, tup(highlight[0]), tup(
            highlight[1]), (0, 0, 255), 2)
        for point in points:
            point = point[0]
            cv2.circle(copy, tup(point), 4, (255, 0, 0), -1)
        cv2.imshow(f"Copy {merge_margin}", copy)
        # key = cv2.waitKey(1);
        # if key == ord('q'):
        #     break;

        # time.sleep(0.1)

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]

            # add margin
            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= merge_margin
            tl[1] -= merge_margin
            br[0] += merge_margin
            br[1] += merge_margin

            # get matching boxes
            overlaps = getAllOverlaps(boxes, [tl, br, [0, 0]], index)
            if len(boxes) < 10:
                print(
                    f'overlaps {index} {curr} {str([tl,br])}:' + str(overlaps))

            # check if empty
            if len(overlaps) > 0:
                # combine boxes
                # convert to a contour
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br, wh = boxes[ind]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)

                # get bounding rect
                x, y, w, h = cv2.boundingRect(con)

                # stop growing
                w -= 1
                h -= 1
                merged = [[x, y], [x+w, y+h], [w, h]]

                # highlights
                highlight = merged[:]
                points = con

                # remove boxes from list
                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]
                boxes.append(merged)

                # set flag
                finished = False
                break

            # increment
            index -= 1
            # time.sleep(0.5)

        if finished and (len(boxes) > BOX_MAX_NUMBER and merge_margin < 100):
            merge_margin += 1
            finished = False
            print(f'new mergin: {merge_margin}')

    # draw final boxes
    cv2.destroyAllWindows()

    # show final
    copy = np.copy(orig)
    # sort by area
    boxes.sort(key=lambda x: x[2][0] * x[2][1], reverse=True)
    for box in boxes:
        cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0, 200, 0), 1)

    # plt.figure('image')
    # plt.imshow(copy)
    # plt.title(f'Binary contours {datetime.now()}')
    # plt.show()

    cv2.imshow("Final", copy)
    # cv2.waitKey(0);
    extendedBoxes = []
    padding = 5
    for box in boxes:
        tl = box[0][:]
        br = box[1][:]
        wh = box[2][:]
        tl[0] -= padding
        tl[1] -= padding
        br[0] += padding
        br[1] += padding
        tl[0] = max(0, tl[0])
        tl[1] = max(0, tl[1])
        br[0] = min(imgWidth - 1, br[0])
        br[1] = min(imgHeight - 1, br[1])
        wh[0] = br[0] - tl[0]
        wh[1] = br[1] - tl[1]
        extendedBoxes.append([tl, br, wh])
    return orig, extendedBoxes


def removeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)


def saveBoxes(filename, boxes, texts):
    removeFile(filename)
    with open(filename, 'w') as f:
        for i, box in enumerate(boxes):
            f.write(
                f'{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]},{box[2][0]},{box[2][1]}:{texts[i]}\n')


def saveSubImages(img, boxes, foldername, prefix):
    pathname = os.path.join(foldername, f'{prefix}_orig.jpg')
    removeFile(pathname)
    cv2.imwrite(pathname, img)
    texts = []
    for i, box in enumerate(boxes):
        sub = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        # the first box is the biggest one, shall be the floorplan
        pathname = os.path.join(
            foldername, f'{prefix}_floorplan.jpg' if i == 0 else f'{prefix}_{i}.jpg')
        removeFile(pathname)
        try:
            cv2.imwrite(pathname, sub)
        except:
            print(f'error writing {pathname} {sub.shape} {box}')
        text1 = pytesseract.image_to_string(sub)
        # text2s = ocr_reader.readtext(sub)
        prediction_groups = ocr_pipeline.recognize([sub], cls=True)
        print(f'{i} {text1} {prediction_groups}')
        texts.append([text1, prediction_groups])
    saveBoxes(os.path.join(foldername, f'{prefix}_boxes.txt'), boxes, texts)

# walk through folder


def walkFolder(foldername, targetFolder):
    for root, dirs, files in os.walk(foldername):
        for file in files:
            if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPEG"):
                filename = os.path.join(root, file)
                print(filename)
                orig, boxes = getImageAndBoxes(filename)
                # extract the filename without extension
                prefix = os.path.splitext(file)[0]
                # save boxes
                saveSubImages(orig, boxes, targetFolder, prefix)


foldername = '/Users/fred/work/floorplan/images'
walkFolder(foldername, '/Users/fred/work/floorplan/target')
