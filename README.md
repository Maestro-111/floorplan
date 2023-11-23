# Seneca Floorplan Project


---
- Phase 1 progress 1%
- Phase 2 progress 0
- Phase 3 progress 20%
- Phase 4 progress 0
- Phase 5 progress 0

## Floorplan processing (phase 3)

- extract address and project name to DB and/or folders
- read source folder:
  - read target info.json to get previous info
    - establish initial page index
  - split into single page of images if pdf file
  - process each image as a page if it's an image file
- split each page to
  - floorplan: biggest image. Feed for future ML
  - keyplate: may have multiple. Feed for future ML
  - unit name
  - sqft (+balcany sqft)
  - floors
  - remarks
  - disclaimers
- save all info to DB and/or folders

## Folder structures:

1. Source folder, can be nested. The name must follow the following rules:
   1. pdf project file or folder name: 'address - project name(.pdf)'
   2. image file name: 'address - project name - unit.jpg'
   3. New project must give a unique virtual address which is close to the real address. The address must be unique in the whole database. The address must be in the following format: 'St No.' + 'Street name without spaces in camelcase'
2. Working folder, to save middle images for debug purpose
   1. each project in a separate sub-folder. named as 'St No.' + 'Street name without spaces in camelcase'(addr)
3. Target internal folder. To hold each building in separate sub-folders. Floorplan/keyplate/drawing/remark-info as different files in that folder.

   1. info.json:
      1. namemap: a string which hold 36 characters in a random order. Used to encode project id to a string.
      2. total: projects
      3. last: last project
      4. ts: timestamp
   2. project sub-folder name: 'St No.' + 'Street name without spaces in camelcase'
      1. info.json
         1. id: project id, used for folder name in public folder
         2. folder: project folder name. Used for folder name in public folder.
         3. addr: address without spaces in camelcase
         4. address: address
         5. project: project name
         6. original: original file name
         7. pageCount: total pages
         8. pages: {avgerage_hash: page index}
         9. avgContour: average contours per page
         10. contours: total contours. also served as next contour index for public filename.
         11. ts: timestamp
      2. unit sub-folder name: page index number
         1. original file: 'original.jpg'
         2. floorplan(largest): 'floorplan.jpg'
         3. keyplates(probably second largest): 'keyplate1..n.jpg'
         4. direction(smallest): 'direction.jpg'
         5. others: 1..n.jpg
         6. info.json
            1. index: page index
            2. hash: average hash of the page
            3. publicName: file name from index, used for public folder
            4. orig: original file name
            5. area: area of the page. pixels
            6. wh: width and height of the page. pixels
            7. rotation: rotation angle
            8. sqft: sqft of the unit
            9. balcany: sqft of the balcany
            10. totalSqft: total sqft of the unit
            11. br: number of bedrooms
            12. br_plus: number of bedrooms plus
            13. bath: number of bathrooms
            14. floors[]: floor numbers
            15. unit: unit number
            16. name: unit name
            17. units: all possible units number in the building
            18. contours:
                1. index: contour index
                2. filename: internal file name
                3. public: public file name if has one. Encoded with namemap, ending with .jpg
                4. area: area of the contour. pixels
                5. xywh: [[x1,y1],[x2,y2],[w,h]]
                6. text: text in the contour
                   1. te: tessaract text
                      1. text: text
                      2. orig: original text
                   2. er: easyocr text
                      1. text: text
                      2. orig: original text result
                   3. kr: kerasocr text
                      1. text: text
                      2. orig: original text result

4. Target public folder:
   1. Project subfolder name: encoded(id). To hold each building in separate sub-folders. Floorplan/keyplate/drawing/remark-info as different files in that folder.
      1. each floorplan, keyplate, direction arrow as a separate file

### Database structure (mongo)
TODO

# setup enviroment
```
pip install keras-ocr
pip install pytesseract
pip install opencv-python
pip install easyocr
pip install pdf2image
pip install imagehash
pip install PyMuPDF
pip install fitz
```
