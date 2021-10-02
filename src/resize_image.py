import os
import glob
from PIL import Image

img_files = glob.glob(os.path.join("images", '*'))
width = 500
height = 500

for f in img_files:
    img = Image.open(f)
    img_resize = img.resize((width, height))
    fname, fext = os.path.splitext(f)
    img_resize.save(fname + '_500x500' + fext)
