# TODO: add description of this file
# DEPRECATED, use as reference only
import imagehash
from PIL import Image

import os
import re
import cv2
import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import time
import fitz
import platform
import pytesseract



if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    import easyocr
    # import keras_ocr
    ocr_easy = easyocr.Reader(['en']) #,gpu = False)  # (['ch_sim','en'])
    # ocr_keras = keras_ocr.pipeline.Pipeline()
    TRY_OCR = True
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    TRY_OCR = False

BOX_MIN_NUMBER = 2
BOX_MAX_NUMBER = 8
HASH_SIZE = 12


def encodeNumber(number: int, namemap: str):
    """Encode 36 based number to string."""
    encoded = ''
    number = int(number)
    length = len(namemap)
    if number == 0:
        return namemap[0]
    while number > 0:
        n = number % length
        encoded = namemap[n] + encoded
        number = number // length
    if encoded.strip() == '':
        raise Exception(f'number {number} encoded to empty string')
    return encoded


def removeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)


def makeSureFolderExists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def replaceImageFile(filename, img):
    removeFile(filename)
    cv2.imwrite(filename, img)


def filename2info(fullpath, delimiter='-'):
    dirname = os.path.dirname(fullpath)
    dirname = os.path.basename(dirname)
    print(f'try to get info from dirname {dirname} with delimiter {delimiter}')
    parts = dirname.split(delimiter)
    if len(parts) <= 1:
        filename = os.path.basename(fullpath)
        print(
            f'try to get info from filename {dirname} with delimiter {delimiter}')
        filename = os.path.splitext(filename)[0]
        parts = filename.split(delimiter)
    if len(parts) <= 1:
        return None, None, None
    address = parts[0].strip()
    addr = camelCase(address)
    project = parts[1].strip()
    return addr, address, project


def camelCase(s):
    return re.sub(r"(_|-)+", " ", s).title().replace(" ", "")


def saveInfo(filename, data):
    removeFile(filename)
    with open(filename, 'w') as outfile:
        try:
            json.dump(data, outfile, indent=4, sort_keys=True)
        except Exception:
            print(f'error {filename}')
            print(data)

def loadInfo(filename, default=None):
    if not os.path.exists(filename):
        return default
    with open(filename, 'r') as f:
        return json.load(f)


def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center, theta, 1)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img, M, (bound_w, bound_h),
                             borderValue=(255, 255, 255))
    return rotated


def tup(point):
    return (point[0], point[1])


def overlap(source, target):
    """ returns true if the two boxes overlap"""
    # unpack points
    tl1, br1, _wh1 = source
    tl2, br2, _wh2 = target

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True


def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps


def medianCanny(img, thresh1, thresh2):
    median = np.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img


def getImageAndBoxes(img):
    """ returns a list of boxes and the total width and height of all boxes """
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    imgInfo = {}
    imgInfo['area'] = imgHeight * imgWidth
    imgInfo['wh'] = [imgWidth, imgHeight]
    imgChannels = img.shape[2]
    orig = np.copy(img)
    blue, green, red = cv2.split(img)

    blue_edges = medianCanny(blue, 0, 1)
    green_edges = medianCanny(green, 0, 1)
    red_edges = medianCanny(red, 0, 1)

    edges = blue_edges | green_edges | red_edges

    # I'm using OpenCV 3.4. This returns (contours, hierarchy) in OpenCV 2 and 4
    # contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 1)
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
        # print("Len Boxes: " + str(len(boxes)))
        # if len(boxes) < BOX_MAX_NUMBER:
        #     print(str(boxes))
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
        # cv2.imshow(f"Copy {merge_margin}", copy)
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
            # if len(boxes) < 10:
            #     print(
            #         f'overlaps {index} {curr} {str([tl,br])}:' + str(overlaps))

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
            # print(f'new mergin: {merge_margin}')

    # draw final boxes
    # cv2.destroyAllWindows()

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

    # cv2.imshow("Final", copy)
    # cv2.waitKey(0);
    extendedBoxes = []
    padding = 5
    widthTotal = 0
    heightTotal = 0
    index = 0
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
        widthTotal += wh[0]
        heightTotal += wh[1]
        extendedBoxes.append({
            'index': index,
            'area': wh[0]*wh[1],
            'xywh': [tl, br, wh]
        })
        index += 1
    imgInfo['contours'] = extendedBoxes
    return imgInfo, widthTotal, heightTotal, copy


def pdf2images(pdf_file, output_dir):
    imageFiles = []

    # open the file
    pdf_doc = fitz.open(pdf_file)

    # STEP 3
    # iterate over PDF pages
    for page_index in range(len(pdf_doc)):

        filename = f'p{page_index}.jpg'

        # get the page itself
        page = pdf_doc[page_index]
        # image_list = page.getImageList()

        img = page.get_pixmap(matrix=fitz.Identity, dpi=500,
                              colorspace=fitz.csRGB, clip=None, alpha=True, annots=True)
        filepath = os.path.join(output_dir, filename)
        removeFile(filepath)
        img.save(filepath)  # save file
        print(f'page {page_index} saved to {filepath}')
        imageFiles.append(filename)
    return imageFiles


def getProjectInfo(file, folders, baseInfo) -> dict:
    addr, address, project = filename2info(file)
    if addr is None:
        print(f'cannot parse address from {file}')
        return None
    # check if the address is already processed
    projectInfo = None
    if os.path.exists(os.path.join(folders['inter'], addr)):
        projectInfo = loadInfo(os.path.join(
            folders['inter'], addr, 'info.json'))
    if projectInfo is None:
        projectInfo = {
            'id': baseInfo['total'],
            'publicName': encodeNumber(baseInfo['total'], baseInfo['namemap']),
            'addr': addr,
            'address': address,
            'project': project,
            'original': file,
            'pageCount': 0,
            'pages': {},
            'avgContours': 0,
            'contours': 0,
            'ts': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        }
        baseInfo['total'] += 1
        baseInfo['last'] = projectInfo['address']
        baseInfo['ts'] = projectInfo['ts']
    return projectInfo


def saveProjectInfo(folders, projectInfo: dict):
    projectInfo['pageCount'] = len(projectInfo['pages'])
    projectInfo['avgContours'] = projectInfo['contours'] / \
        projectInfo['pageCount']
    projectInfo['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saveInfo(os.path.join(
        folders['inter'], projectInfo['addr'], 'info.json'), projectInfo)


def getImagesBoxesInRightDirection(img):
    imgInfo, w, h, imgWithBoxes = getImageAndBoxes(img)
    rotation = 0
    if h > w:
        rotation = 270
        img = rotate(img, rotation)
        imgInfo, w, h, imgWithBoxes = getImageAndBoxes(img)
    imgInfo['rotation'] = rotation
    return img, imgInfo, imgWithBoxes


def processSubImages(img, folders, baseInfo, projectInfo, imgInfo):
    global TRY_OCR
    imageInterFolder = os.path.join(
        folders['inter'], projectInfo['addr'], str(imgInfo['index']))
    makeSureFolderExists(imageInterFolder)
    projectPublicFolder = os.path.join(
        folders['public'], projectInfo['publicName'])
    makeSureFolderExists(projectPublicFolder)

    # save the original image to inter
    pathname = os.path.join(imageInterFolder, 'original.jpg')
    replaceImageFile(pathname, img)
    # go through the boxes
    boxes = imgInfo['contours']
    contourCount = len(boxes)
    for i, contour in enumerate(boxes):
        box = contour['xywh']
        sub = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        tesseractTxt = pytesseract.image_to_string(sub)
        contour['text'] = {
            'te': {'text': tesseractTxt},
        }
        if TRY_OCR:
            easyTxtList = ocr_easy.readtext(sub)
            if len(easyTxtList) > 0:
                easyTxt = ' '.join(easyTxtItem[1]
                                   for easyTxtItem in easyTxtList)
            else:
                easyTxt = ''
            contour['text']['er'] = {'text': easyTxt} #, 'orig': easyTxtList}
            #kerasTxtList = ocr_keras.recognize([sub])
            #if len(kerasTxtList[0]) > 0:
            #    kerasTxt = ' '.join(kerasTxtItem[0]
            #                        for kerasTxtItem in kerasTxtList[0])
            #else:
            #    kerasTxt = ''
            #contour['text']['kr'] = {'text': kerasTxt} #, 'orig': kerasTxtList}
        else:
            easyTxt = ''
            #kerasTxt = ''
        print(f'{i}: te:{tesseractTxt}\n er:{easyTxt}')
        if i == 0:  # floorplan
            # save the floorplan to inter
            replaceImageFile(os.path.join(
                imageInterFolder, 'floorplan.jpg'), sub)
            # save the floorplan to public
            replaceImageFile(os.path.join(
                projectPublicFolder, imgInfo['publicName'] + '_floorplan.jpg'), sub)
        # need ML to recognize the keyplate and arrow
        # elif i == 1:  # keyplate
        #     # save the keyplate to inter
        #     replaceImageFile(os.path.join(
        #         imageInterFolder, 'keyplate.jpg'), sub)
        #     # save the keyplate to public
        #     replaceImageFile(os.path.join(
        #         projectPublicFolder, imgInfo['publicName'] + '_keyplate.jpg'), sub)
        # elif i == contourCount-1:  # arrow
        #     # save the arrow to inter
        #     replaceImageFile(os.path.join(imageInterFolder, 'arrow.jpg'), sub)
        #     # save the arrow to public
        #     replaceImageFile(os.path.join(
        #         projectPublicFolder, imgInfo['publicName'] + '_arrow.jpg'), sub)
        else:  # other
            # save the other to inter
            replaceImageFile(os.path.join(imageInterFolder, f'{i}.jpg'), sub)
    # save image info
    extractUnitInfo(imgInfo)
    saveInfo(os.path.join(imageInterFolder, 'info.json'), imgInfo)
    projectInfo['contours'] += contourCount


def extractUnitInfo(imgInfo):
    '''Extract sqft/balcany/totalSqft/br/br_plus/bath/floors/unit/name from the image'''
    pass


def processImage(fullpath, folders, baseInfo: dict, projectInfo: dict = None):
    hasProjectInfo = True
    if projectInfo is None:
        hasProjectInfo = False
        projectInfo = getProjectInfo(file, folders, baseInfo)
    if projectInfo is None:
        print(f'cannot parse address from {file}')
        return
    img = cv2.imread(fullpath)
    if img is None:
        print(f'cannot read image from {fullpath}')
        return
    img, imgInfo, imgWithBoxes = getImagesBoxesInRightDirection(img)
    imgHash = str(imagehash.average_hash(
        Image.fromarray(img), hash_size=HASH_SIZE))
    if imgHash in projectInfo['pages']:
        print(f'already processed {fullpath}')
        return
    imgInfo['index'] = projectInfo['pageCount']
    projectInfo['pageCount'] += 1
    imgInfo['hash'] = imgHash
    imgInfo['publicName'] = encodeNumber(
        projectInfo['pageCount'], baseInfo['namemap'])
    imgInfo['original'] = os.path.basename(fullpath)
    projectInfo['pages'][imgHash] = imgInfo['index']
    filename = os.path.basename(fullpath)
    imgInfo['filename'] = filename
    # save the image to middle dir
    projectMiddleFolder = os.path.join(
        folders['middle'], projectInfo['addr'])
    makeSureFolderExists(projectMiddleFolder)
    # save the original image to middle
    pathname = os.path.join(projectMiddleFolder, imgInfo['filename'])
    replaceImageFile(pathname, imgWithBoxes)
    # save boxes
    processSubImages(img, folders, baseInfo, projectInfo, imgInfo)
    # save image info
    saveInfo(os.path.join(
        folders['inter'], projectInfo['addr'], str(imgInfo['index']), 'info.json'), imgInfo)
    # save project info
    if not hasProjectInfo:
        saveProjectInfo(folders, projectInfo)


def processPDFFile(file, folders, baseInfo: dict):
    projectInfo = getProjectInfo(file, folders, baseInfo)
    if projectInfo is None:
        print(f'cannot parse address from {file}')
        return
    print(f'processing project {projectInfo}')
    output_dir = os.path.join(folders['middle'], projectInfo['addr'])
    makeSureFolderExists(output_dir)
    images = pdf2images(file, output_dir)
    for i in range(len(images)):
        imgfilepath = os.path.join(output_dir, images[i])
        processImage(imgfilepath, folders, baseInfo, projectInfo)
    # save project info
    saveProjectInfo(folders, projectInfo)
    print(f'{file} done')


def processFile(fullpath, folders, baseInfo: dict):
    file = os.path.basename(fullpath)
    _filename, ext = os.path.splitext(file)
    ext = ext.lower()
    if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
        processImage(fullpath, folders, baseInfo)
    if ext == '.pdf':
        processPDFFile(fullpath, folders, baseInfo)


# walk through folder
def walkFolder(foldername, folders, baseInfo: dict):
    for root, dirs, files in os.walk(foldername):
        for file in files:
            fullpath = os.path.join(root, file)
            processFile(fullpath, folders, baseInfo)
        for dir in dirs:
            dirpath = os.path.join(root, dir)
            walkFolder(dirpath, folders, baseInfo)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        # file = r'/Users/fred/work/floorplan/Toronto/1 _ 3 Market St - Market Wharf/Market-Wharf-floorplans.compressed.pdf'
        file = r'/Users/fred/work/floorplan/Toronto/135 & 155 Dalhousie St - The Merchandise Building Original Lofts/Merchandise-Lofts-FloorPlan.pdf'
        if platform.system() == 'Windows':
            file = r'C:\Users\qfxia\work\floorplan\Toronto\135 & 155 Dalhousie St - The Merchandise Building Original Lofts\Merchandise-Lofts-FloorPlan.pdf'
    middleDir = r'/Users/fred/work/floorplan/middle'
    internalDir = r'/Users/fred/work/floorplan/internal'
    publicDir = r'/Users/fred/work/floorplan/public'
    if platform.system() == 'Windows':
        middleDir = r'C:\Users\qfxia\work\floorplan\middle'
        internalDir = r'C:\Users\qfxia\work\floorplan\internal'
        publicDir = r'C:\Users\qfxia\work\floorplan\public'
    dirs = {
        "middle": middleDir,
        "inter": internalDir,
        "public": publicDir
    }
    baseInfo = loadInfo(
        os.path.join(dirs['inter'], 'info.json'),
        {
            "namemap": 'bkcz9fhdi6p0uv5gr3n1qjwal4syo7mte82x',
            "total": 0,
            "last": 0,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    if os.path.isdir(file):
        # walk the directory
        walkFolder(file, dirs, baseInfo)
    elif os.path.isfile(file):
        # process the file
        processFile(file, dirs, baseInfo)
    else:
        print(f"file not found or not a file. {file}")
        exit - 1
    # save base info
    saveInfo(os.path.join(dirs['inter'], 'info.json'), baseInfo)
    print('done')
