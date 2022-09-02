import torch
from torch.utils.data import BatchSampler, SequentialSampler
import numpy as np
from glob import glob
import cv2
import os
import imgaug
import imgaug.augmenters as iaa
from tqdm.auto import tqdm
import pandas as pd
import torchvision.transforms as T
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, config):
        self.mask_paths = sorted(glob(os.path.join(video_dir, 'segmentation/*.png')))
        self.img_paths = [p.replace('segmentation', 'rgb') for p in self.mask_paths]
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.aug = config['aug']
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 50% horizontal flip
            iaa.Affine(
                rotate=(-15, 15),
                shear=(-10, 10),
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            ),
        ])
        self.config = config
        self._init_img_preprocess_fn(config)
        self._init_vignetting(config)

    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])[:, :, ::-1]
        mask = cv2.imread(self.mask_paths[idx])[:, :, 0]
        
        img = self.preprocess(img)
        mask = self.preprocess_mask(mask)
        if self.config['vignetting_calibration']:
            img = img
            # img = vignetting_calibration(img, self.vignetting_mask)
        if self.aug:
            img, mask = self.seq(image=img, segmentation_maps=mask)

        return self._to_torch_tensor(img, mask)
    
    def preprocess(self, img):
        img = cv2.resize(img, (self.img_w, self.img_h))
        return img
    
    def preprocess_mask(self, mask):
        mask = imgaug.augmentables.segmaps.SegmentationMapsOnImage(mask.astype(np.int8), 
                                                                   shape=mask.shape)
        mask = mask.resize((self.img_h, self.img_w))
        return mask

    def _init_img_preprocess_fn(self, config):
        model_type = config['model_type']
        if model_type == 'UNet' and config[model_type]['encoder']['pretrained']:
            transform = T.Compose([
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # img = transform(img)
        elif model_type == 'smp':
            encoder_name = config[model_type]['encoder_name']
            pretrained = config[model_type]['pretrained']  # used pretrained
            transform = get_preprocessing_fn(encoder_name, pretrained='imagenet')
        else:
            raise ValueError('Not implemented model type preprocess fn')
        self.transform = transform

    def _to_torch_tensor(self, img, mask):
        model_type = self.config['model_type']
        if model_type == 'UNet':
            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
            img = self.transform(img)

        elif model_type == 'smp':
            img = img / 255.
            img = self.transform(img)
            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        else:
            raise ValueError('Not implemented model type preprocess fn')

        if mask:
            mask = mask.get_arr()  # to np
            mask = torch.tensor(mask, dtype=torch.long)
            return img, mask
        else:
            return img

    def _init_vignetting(self, config):
        return
        # vignetting_mask = np.load(config['vignetting_path'])
        # self.vignetting_mask = cv2.resize(vignetting_mask, (self.img_w, self.img_h))



class ActionDataset(SegDataset):
    def __init__(self, video_dir, config):
        super().__init__(video_dir, config)
        self.video_dir = video_dir
        df = pd.read_csv(os.path.join(video_dir, 'action_discrete.txt'), header=None, names=['frame', 'class'])
        df['frame'] = [f"{i:09d}" for i in df['frame'].tolist()]
        self.df = df
        self.config = config
        self._init_img_preprocess_fn(config)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_id = row['frame']
        frame_path = os.path.join(self.video_dir, 'rgb', f'{frame_id}.png')
        action = int(row['class'].item())
        action = torch.tensor(action, dtype=torch.long)

        frame = self.preprocess(cv2.imread(frame_path)[:, :, ::-1])
        frame_t = self._to_torch_tensor(frame, None)

        # if self.aug:
        #     img, mask = self.seq(image=img, segmentation_maps=mask)

        return frame_t, action

    def __len__(self):
        return len(self.df)

# use segmentation mask as input
class ActionDataset2(SegDataset):
    def __init__(self, video_dir, config):
        super().__init__(video_dir, config)
        self.video_dir = video_dir
        self.set_id = video_dir.split(os.sep)[-2]  # train1 or train2
        self.video_id = video_dir.split(os.sep)[-1]  # video_*
        self.frame_paths = sorted(glob(os.path.join(video_dir, 'segmentation', '*.png')))
        frame_names = [p.split(os.sep)[-1].split('.')[0] for p in self.frame_paths] # '000000000 000000060...'
        df = pd.read_csv(os.path.join(video_dir, 'action_discrete.txt'), header=None, names=['frame', 'class'])
        df['frame'] = [f"{i:09d}" for i in df['frame'].tolist()]
        df = df[df['frame'].isin(frame_names)]  # select action frame with segmentation mask
        self.df = df
        self.config = config
        self._init_img_preprocess_fn(config)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_id = row['frame']
        mask_path = os.path.join(self.video_dir, 'segmentation', f'{frame_id}.png')
        # frame_path = os.path.join(self.video_dir, 'rgb', f'{frame_id}.png')
        action = int(row['class'].item())
        action = torch.tensor(action, dtype=torch.long)
        # print(mask_path)
        mask = cv2.imread(mask_path)[:, :, 0]
        mask = self.preprocess_mask(mask)
        mask = mask.get_arr()  # to np
        mask = torch.tensor(mask, dtype=torch.long)
        mask = torch.nn.functional.one_hot(mask, num_classes=self.config['num_cls'])
        mask = mask.to(dtype=torch.float).permute(2, 0, 1)

        # if self.aug:
        #     img, mask = self.seq(image=img, segmentation_maps=mask)

        return mask, action

    def __len__(self):
        return len(self.df)

class InferenceActionDataset(SegDataset):
    def __init__(self, mask_dir, config):
        super().__init__(mask_dir, config)
        self.video_dir = mask_dir
        self.set_id = mask_dir.split(os.sep)[-2]  # train1 or train2
        self.video_id = mask_dir.split(os.sep)[-1]  # video_*
        self.mask_paths = sorted(glob(os.path.join(mask_dir, 'segmentation', '*.png')))
        self.config = config
        self._init_img_preprocess_fn(config)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path)[:, :, 0]
        mask = self.preprocess_mask(mask)
        mask = mask.get_arr()  # to np
        mask = torch.tensor(mask, dtype=torch.long)
        mask = torch.nn.functional.one_hot(mask, num_classes=self.config['num_cls'])
        mask = mask.to(dtype=torch.float).permute(2, 0, 1)

        # if self.aug:
        #     img, mask = self.seq(image=img, segmentation_maps=mask)

        return mask, mask_path

    def __len__(self):
        return len(self.mask_paths)

class InferenceSegDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, config):
        img_paths_raw = sorted(glob(os.path.join(video_dir, 'rgb/*.png')))[:]
        img_paths = []
        for p in img_paths_raw:
            if int(p.split(os.sep)[-1].split(".")[0]) % 60 == 0:
                img_paths.append(p)
        self.img_paths = img_paths

        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.config = config
        self._init_img_preprocess_fn(config)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]

        img = cv2.imread(path)
        # TODO: sometimes img is corrupted, need handling
        if img is not None:
            img = img[:, :, ::-1]
        else:
            print('corrupted img', path)
            img = np.zeros((self.img_h, self.img_w, 3), dtype=int)

        img = self.preprocess(img)
        return self._to_torch_tensor(img, None), path

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_w, self.img_h))
        return img

    def _init_img_preprocess_fn(self, config):
        model_type = config['model_type']
        if model_type == 'UNet' and config[model_type]['encoder']['pretrained']:
            transform = T.Compose([
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # img = transform(img)
        elif model_type == 'smp':
            encoder_name = config[model_type]['encoder_name']
            pretrained = config[model_type]['pretrained']  # used pretrained
            transform = get_preprocessing_fn(encoder_name, pretrained='imagenet')
        else:
            raise ValueError('Not implemented model type preprocess fn')
        self.transform = transform

    def _to_torch_tensor(self, img, mask):
        model_type = self.config['model_type']
        if model_type == 'UNet':
            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
            img = self.transform(img)

        elif model_type == 'smp':
            img = img / 255.
            img = self.transform(img)
            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        else:
            raise ValueError('Not implemented model type preprocess fn')

        if mask:
            mask = mask.get_arr()  # to np
            mask = torch.tensor(mask, dtype=torch.long)
            return img, mask
        else:
            return img


def _build_seg_dataset(video_dirs, config):
    datasets = []
    for video_dir in tqdm(video_dirs):
        datasets.append(SegDataset(video_dir, config))
    return torch.utils.data.ConcatDataset(datasets)


def _build_act_dataset(video_dirs, config):
    datasets = []
    for video_dir in tqdm(video_dirs):
        datasets.append(ActionDataset2(video_dir, config))
    return datasets

def build_inference_act_dataset(mask_dirs, config):
    datasets = []
    for mask_dir in tqdm(mask_dirs):
        datasets.append(InferenceActionDataset(mask_dir, config))
    print(f'# of test videos dirs: {len(datasets)}')
    return datasets

def build_seg_datasets(config):
    data_config = config['data']
    root_dir = data_config['root_dir']
    train_video_dirs = pd.read_csv(data_config['train_csv'])['video_dir'].tolist()
    val_video_dirs = pd.read_csv(data_config['val_csv'])['video_dir'].tolist()
    train_video_dirs = _add_root_dir(root_dir, train_video_dirs)
    print('train_video_dirs: ', train_video_dirs)
    val_video_dirs = _add_root_dir(root_dir, val_video_dirs)
    print('# of Videos for Segmentation: ', len(train_video_dirs), len(val_video_dirs))
    train_ds = _build_seg_dataset(train_video_dirs, config)
    val_ds = _build_seg_dataset(val_video_dirs, config)
    
    return train_ds, val_ds

def build_act_datasets(config):
    data_config = config['data']
    root_dir = data_config['root_act_dir']
    train_video_dirs = pd.read_csv(data_config['train_csv'])['video_dir'].tolist()
    val_video_dirs = pd.read_csv(data_config['val_csv'])['video_dir'].tolist()
    train_video_dirs = _add_root_dir(root_dir, train_video_dirs)
    val_video_dirs = _add_root_dir(root_dir, val_video_dirs)
    print('# of Action Videos: ', len(train_video_dirs), len(val_video_dirs))
    train_ds = _build_act_dataset(train_video_dirs, config)
    val_ds = _build_act_dataset(val_video_dirs, config)

    return train_ds, val_ds


def _add_root_dir(root_dir, dirs):
    return [os.path.join(root_dir, p) for p in dirs]


def build_loader(datasets, config, args):
    train_ds, val_ds = datasets
    gpus = config['gpu']
    bs = config['bs']
    num_workers = args.workers
    if gpus > 1 and torch.cuda.device_count() > 1:
        bs *= gpus
        num_workers *= gpus
        print(f'multi gpu loader batch size: {bs}, workers: {num_workers}')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=False,
                                               num_workers=num_workers,
                                               drop_last=True)  # avoid BN error
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=False,
                                             num_workers=num_workers)
    return train_loader, val_loader

def build_action_loader(datasets, config, args):
    train_ds, val_ds = datasets
    gpus = config['gpu']
    bs = config['bs']
    pin_memory=config['pin_memory']
    num_workers = args.workers
    if gpus > 1 and torch.cuda.device_count() > 1:
        bs *= gpus
        num_workers *= gpus
        print(f'multi gpu loader batch size: {bs}, workers: {num_workers}')

    train_loaders = [torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, pin_memory=pin_memory,
                                                 # sampler=BatchSampler(SequentialSampler(range(len(ds))), batch_size=bs, drop_last=False),
                                                 sampler=SequentialSampler(range(len(ds))),
                                                 num_workers=num_workers) for ds in train_ds]
    val_loaders = [torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, pin_memory=pin_memory,
                                               # sampler=BatchSampler(SequentialSampler(range(len(ds))), batch_size=bs, drop_last=False),
                                               sampler=SequentialSampler(range(len(ds))),
                                               num_workers=num_workers) for ds in val_ds]

    return train_loaders, val_loaders

def build_inference_action_loader(test_datasets, config, args):
    gpus = 1
    bs = config['bs']
    pin_memory = config['pin_memory']
    num_workers = 0
    if gpus > 1 and torch.cuda.device_count() > 1:
        bs *= gpus
        num_workers *= gpus
        print(f'multi gpu loader batch size: {bs}, workers: {num_workers}')

    test_loaders = [torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, pin_memory=pin_memory,
                                                sampler=SequentialSampler(range(len(ds))),
                                                num_workers=num_workers) for ds in test_datasets]
    return test_loaders

def vignetting_calibration(img, vignetting):
    """
    :param img: (H, W, C) RGN image
    :param vignetting: (H, W) mean intensity of all image
    :return: after calibration
    """
    img2 = vig(img.copy(), vignetting)
    #     img3 = hisEqulColor(img.copy())
    img3 = clahe(img2.copy())
    return img3


def vig(img, vignet):
    threshold = img.mean().astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_v = img_hsv[:, :, 2]
    dark_region = img_v<threshold
    dark_vigenting = vignet<threshold
    dark_mask = np.logical_and(dark_region, dark_vigenting)
    img_v[dark_mask] += (threshold - vignet[dark_mask])
    img_v = np.clip(img_v, 0, 255)
    img_hsv[:, :, 2] = img_v.copy()
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_rgb


# https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
def clahe(rgb):
    gridsize = 8
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


if __name__ == '__main__':
    print('GG')
    DATASET_DIR = ".."
    VIDEO_DIRS = glob(os.path.join(DATASET_DIR, "train*", "video*"))
#     video_dir = VIDEO_DIRS[0]
    config = {'img_h': 480, 'img_w': 640}
    train_ds, val_ds = build_act_datasets(config)
#     ds = SegDataset(config)
#     ds = _build_seg_dataset(VIDEO_DIRS, config)
#     print(len(ds))
