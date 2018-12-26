from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
import os
from modeling.deeplab import *
import PIL
from torch.autograd import Variable

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''

class testset(Dataset):
    """
    test dataset
    """
    NUM_CLASSES = 21

    def __init__(self, dir, transform):
        """
        :param dir: path to test dataset directory
        """
        super().__init__()
        self.dir = "E:\img"
        self.img_list = list(map(lambda x: os.path.join(self.dir, x), os.listdir(self.dir)))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        _img = PIL.Image.open(self.img_list[index]).convert('RGB')
        _img = self.transform(_img)
        return _img

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
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
    parser.add_argument('--batch-size', type=int, default=2,
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
    args.cuda = False
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
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
    print(args)
    composed_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    torch.manual_seed(args.seed)
    test_ds = testset(dir, composed_transforms)


    test_load = DataLoader(test_ds, batch_size=1)

    model = DeepLab(backbone='xception', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False)

    checkpoint = torch.load(args.resume)

    if args.cuda:
        model = model.cuda()
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # print(model)

    for i, image in enumerate(test_load):

        if args.cuda:
            image = image.cuda()
        image = Variable(image, volatile=True)#.cuda()

        output = model(image)

        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        prediction = prediction.astype('uint8')
        im = PIL.Image.fromarray(prediction)
        im.save(os.path.join('test', 'result', str(i) + '.png'))

if __name__ == "__main__":
    main()
