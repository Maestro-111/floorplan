# floorplan
Floorplan processing

- extract address and project name to DB & folder name
- split into single page of images
- split each page to 
    - floorplan: biggest image. Feed for future ML
    - keyplate: may have multiple. Feed for future ML
    - unit name
    - soft (+balcany sqft)
    - floors
    - remarks
    - disclaimers
- save all info to DB


Folder structures:
1. Source folder, can be nested
2. Working folder, to save middle images for debug purpose
3. Target folder. To hold each building in separate sub-folders. Floorplan/keyplate/drawing/remark-info as different files in that folder.
