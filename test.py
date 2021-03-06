from torch.utils.data import Dataset
import argparse
import os
from modeling.deeplab import *
import PIL
from modeling.sync_batchnorm.replicate import patch_replication_callback
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


# class testset(Dataset):
#     """
#     test dataset
#     """
#     NUM_CLASSES = 21
#
#     def __init__(self):
#         """
#         :param dir: path to test dataset directory
#         """
#         super().__init__()
#         self.dir = "E:\img"
#         self.img_list = list(map(lambda x: os.path.join(self.dir, x), os.listdir(self.dir)))
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, index):
#         _img = PIL.Image.open(self.img_list[index]).convert('RGB')
#         _img = self.transform(_img)
#         return _img



def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default="xception_trans_model.pth",
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()

    # train_loader, val_loader, arg_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    #
    # for i, sample in enumerate(tbar):
    #     image = sample[0]
    #     id = sample[1]
    #
    #
    #     if self.args.cuda:
    #         image = image.cuda()
    #     with torch.no_grad():
    #         output = self.model(image)
    #         prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    #         prediction = prediction.astype('uint8')
    #         im = PIL.Image.fromarray(prediction)
    #         im.save(id[0])
    #     break

    # args.cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)


    # composed_transforms = transforms.Compose([transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    torch.manual_seed(args.seed)
    # test_ds = testset()
    # test_load = DataLoader(test_ds, batch_size=1)
    model = DeepLab(backbone='xception', output_stride=8, num_classes=21, sync_bn=True, freeze_bn=True)

    checkpoint = torch.load(args.resume)

    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # tbar = tqdm(test_loader, desc='\r')
    img = plt.imread("F:/pingan/VOCdevkit/VOC2012/test\\2008_006894_.jpg")
    h, w, _ = img.shape
    ratio = 513. / np.max([w, h])
    resized = cv2.resize(img, (int(ratio * w), int(ratio * h)))
    resized = resized / 127.5 - 1.
    if w < h:
        pad_x = int(513 - resized.shape[1])
        resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
    elif w > h:
        pad_y = int(513 - resized.shape[0])
        resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')

    resized2 = resized2.transpose((2, 0, 1))
    resized3 = np.zeros((1, 3, 513, 513), dtype=np.float32)
    resized3[0, ...] = resized2
    resized4 = torch.from_numpy(resized3)
    # print(resized4[0, 0, :5, :5])
    # print(resized4.shape)
    with torch.no_grad():
        output = model(resized4)
    prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    prediction = prediction.astype('uint8')
    im = PIL.Image.fromarray(prediction)
    # print(output[0, 1, 20:25, 20:25])
    # print(output)
    im.save(os.path.join('F://123.png'))


if __name__ == "__main__":
    main()
