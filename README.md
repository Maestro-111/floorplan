# Seneca Floorplan Project

this repo is for RM floor project.

---

- Phase 1 progress 95%
- Phase 2 progress 0
- Phase 3 progress 20%
- Phase 4 progress 0
- Phase 5 progress 0

# Contribute

- install poetry, see https://python-poetry.org/docs/
- install dependencies by `poetry install`

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

### Database structure (mongo)

TODO:

# setup enviroment

```
poetry install
pip install keras-ocr
pip install pytesseract
pip install opencv-python
pip install easyocr
pip install pdf2image
pip install imagehash
pip install PyMuPDF
pip install fitz
```

# TF and PyTorch setup

- Use python3.11 and TF 2.13 on office 3090 machine
- Use the latest PyTorch stable version
- https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba

  - change 515 to 545 # line 34

- add follow to .bash_profile or .zshrc and source it.
  ```
  export PATH=/usr/local/cuda-11.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
  ```

setup python environment

```
python3.11 -m venv py311
source py311/bin/activate

pip install --upgrade pip
pip install tensorflow[and-cuda]
pip install --upgrade tensorrt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Copyright/License/Disclamer

apache-2.0 license.
TODO:
