import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import PIL
import os

n_classes = 21
label_colours = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                            [0, 64, 128]])
sourcedir = 'F:\\pingan\\VOCdevkit\\VOC2012\\SegmentationClassAug'
targetdir = 'C:\\Users\\Shuang\\Desktop\\new'
img_list = os.listdir(sourcedir)
img_list = list(map(lambda x: os.path.join(sourcedir, x), img_list))
tbar = tqdm(img_list)
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
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    plt.imshow(rgb)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(targetdir, str(id) + '.png'))





