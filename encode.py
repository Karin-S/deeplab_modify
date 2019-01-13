import numpy as np
from tqdm import tqdm
import PIL
import os
import skimage.io


'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


n_classes = 21
label_colours = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                            [0, 64, 128]])

sourcedir = 'F:\\pingan\\VOCdevkit\\VOC2012\\TV'
targetdir = 'F:\\pingan\\VOCdevkit\\VOC2012\\TV'
mask_list = os.listdir(sourcedir)
mask_list = list(map(lambda x: os.path.join(sourcedir, x), mask_list))
tbar = tqdm(mask_list, desc='\r')
for mask in tbar:
    id = mask[-15:-4]
    mask = PIL.Image.open(mask)
    label_mask = np.array(mask)
    mask = np.array(mask).astype(int)

    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(label_colours):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
    skimage.io.imsave(os.path.join(targetdir, str(id) + '.png'), label_mask)

