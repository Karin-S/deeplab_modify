from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        train_hard_mining_set = pascal.VOCSegmentation(args, split='train_hard_mining')
        val_set = pascal.VOCSegmentation(args, split='val')
        val_save_set = pascal.VOCSegmentation_save(args, split='val')
        arg_set = pascal.VOCSegmentation(args, split='arg')
        test_set = pascal.VOCSegmentationtest(args, split='test')
        compare_set = pascal.VOCSegmentation_compare(args, split='val')

        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_hard_mining_loader = DataLoader(train_hard_mining_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_save_loader = DataLoader(val_save_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        arg_loader = DataLoader(arg_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True, **kwargs)
        compare_loader = DataLoader(compare_set, batch_size=1, shuffle=True, **kwargs)

        return train_loader, train_hard_mining_loader, val_loader, val_save_loader, arg_loader, test_loader, compare_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError