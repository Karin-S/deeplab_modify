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

sourcedir = 'F:\\pingan\\VOCdevkit\\VOC2012\\val_res_original'
targetdir = 'F:\\pingan\\VOCdevkit\\VOC2012\\val_res_original'
img_list = os.listdir(sourcedir)
img_list = list(map(lambda x: os.path.join(sourcedir, x), img_list))

tbar = tqdm(img_list, desc='\r')
for img in tbar:
    id = img[-15:-4]
    img = PIL.Image.open(img)
    label_mask = np.array(img)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    skimage.io.imsave(os.path.join(targetdir, str(id) + '.png'), rgb)