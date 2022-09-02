import argparse
import pandas as pd
import torch
from glob import glob
import imgaug
import torchvision
import os
from tqdm.auto import tqdm
import cv2
from pprint import pprint
import concurrent.futures as cf
import numpy as np

from utils import load_config, check_save_dir, load_yaml
from utils.evaluate_official import eval_official
from utils.logger import init_action_logger
from utils.trainer import state2np, state_to_device
from utils.loss import build_loss
from utils.sample_video import sample_all_video
# from utils.loss import build_loss
from models.action import CNNLSTM, build_action_encoder
from models import build_model

from data import InferenceSegDataset, build_inference_action_loader, build_inference_act_dataset

def pred_seg(loaders, device, model, NUM_JOBS):
    print("Start Segmentation prediction")

    with torch.no_grad():
        with cf.ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
            for loader in tqdm(loaders):
                for i, data in enumerate(tqdm(loader)):
                    imgs, paths = data
                    imgs = imgs.to(device)
                    pred_masks = model(imgs).argmax(1)
                    paths = list(paths)
                    video_ids = [p.split(os.sep)[-3] for p in paths]  # video_*
                    frame_ids = [p.split(os.sep)[-1] for p in paths]  # 0000000*.png
                    for pred_m, video_id, frame_id in zip(pred_masks, video_ids, frame_ids):
                        pred_m = pred_m.cpu().numpy()
                        pred_m = imgaug.augmentables.segmaps.SegmentationMapsOnImage(pred_m.astype(np.int8),
                                                                                     shape=pred_m.shape)
                        pred_m = pred_m.resize((1080, 1920)).get_arr()
                        executor.submit(cv2.imwrite,
                                        os.path.join(args.output_dir, video_id, 'segmentation', frame_id),
                                        pred_m)
    print('Done Segmentation prediction')

def pred_act(device, args):
    config_act = load_yaml(args.act_config)
    encoder = build_action_encoder(config_act)
    model = CNNLSTM(config=config_act, encoder=encoder)
    model.load_state_dict(torch.load(args.act_ckpt, map_location=device))
    model.eval()
    model.to(device)

    video_dirs = sorted(glob(os.path.join(args.output_dir, "video*")))  # video of segmentation output dir
    rgb_dirs = sorted(glob(os.path.join(args.data_dir, "video*")))  # video dir of input data dir
    test_datasets = build_inference_act_dataset(video_dirs, config_act)
    act_loaders = build_inference_action_loader(test_datasets, config_act, args)
    print("Start act pred")
    # temp_state = None
    with torch.no_grad():
        for loader, rgb_dir, output_dir in tqdm(zip(act_loaders, rgb_dirs, video_dirs)):
            print('len(loader.dataset): ', len(loader.dataset))
            mask_ids = []
            pred_actions = []
            temp_state = None  # reset state

            for batch_i, (masks, mask_paths) in enumerate(loader):
                masks = masks.to(device)
                mask_paths = [p.split(os.sep)[-1].split('.')[0] for p in list(mask_paths)]
                # LSTM need previous state for sequence outputs
                pred, new_state = model(masks, state_to_device(temp_state, device)) if batch_i != 0 else model(masks, temp_state)
                temp_state = state2np(new_state)
                pred_cls = list(pred.argmax(1).cpu().numpy())
                mask_ids.extend(mask_paths)
                pred_actions.extend(pred_cls)

            # Process action file
            df = pd.DataFrame({
                "frame_id": mask_ids,
                "label": pred_actions
            })
            rgb_paths = sorted(glob(os.path.join(rgb_dir, 'rgb', '*.png')))
            all_frame_ids = [p.split(os.sep)[-1].split(".")[0] for p in rgb_paths]

            output_frame_ids = []
            output_classes = []
            previous_class = None
            for i in all_frame_ids:
                if int(i)%60 == 0:  # use new action label
                    cls = int(df[df["frame_id"]==i]["label"].values[0])
                    previous_class = cls

                output_frame_ids.append(i)
                output_classes.append(previous_class)

            df_output = pd.DataFrame({
                "frame_id": output_frame_ids,
                "label": output_classes
            })
            df_output.to_csv(os.path.join(output_dir, "action_discrete.txt"), header=None, index=None, sep=',')

    print("Done act pred")


def inference(args):
    print('torch ', torch.__version__)  # 1.11.0
    print('cuda ', torch.version.cuda)  # 11.3
    print('cudnn ', torch.backends.cudnn.version())  # 8200
    NUM_JOBS = 4
    config = load_yaml(args.seg_config)
    # Fix yaml file
    config['smp']['seg_encoder_weights'] = False # disable download encoder w in smp model
    config['smp']['freeze_encoder'] = False
    config['aux'] = False
    # print(config)

    if args.sample_video:
        print('Parse videos')
        sample_all_video(args.data_dir, recursive=True, jobs=NUM_JOBS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    os.makedirs(args.output_dir, exist_ok=True)


    # Segmentation
    if args.pred_seg:
        loaders = build_seg_loaders(args, config)
        model = build_seg_model(args, config, device)
        if args.tta:
            import ttach as tta
            transforms = tta.Compose([tta.HorizontalFlip()])
            model = tta.SegmentationTTAWrapper(model, transforms)
        pred_seg(loaders, device, model, NUM_JOBS)

        # clear graph
        if torch.cuda.is_available():
            print('GPI mem ', torch.cuda.memory_usage())
        del model

    # Action
    if args.pred_act:
        pred_act(device, args)


    if args.eval and args.gt_dir:
        print("Evaluation")
        eval_official(args.gt_dir, args.output_dir)


def build_seg_loaders(args, config):
    video_dirs = glob(os.path.join(args.data_dir, 'video_*'))
    print("video_dirs:", video_dirs)
    # Make prediction dirs
    video_names = [p.split(os.sep)[-1] for p in video_dirs]
    for name in video_names:
        target_dir = os.path.join(args.output_dir, name)
        seg_dir = os.path.join(target_dir, 'segmentation')
        print('create dir : ', target_dir)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
    datasets = [InferenceSegDataset(p, config) for p in video_dirs]
    loaders = [torch.utils.data.DataLoader(ds, batch_size=config['bs'], shuffle=False, pin_memory=False,
                                           drop_last=False) for ds in datasets]
    return loaders


def build_seg_model(args, config, device):
    model = build_model(config)
    saved_dict = torch.load(args.seg_ckpt, map_location=device)
    if config["model_type"] == 'smp':
        # Bug for smp module, key not matched!
        # remove "module." prefix from saved state dict
        from collections import OrderedDict
        updated_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in saved_dict.items()])
        model.load_state_dict(updated_dict)
    else:
        model.load_state_dict(saved_dict)
    print('Load model weights')
    model.eval()
    model.to(device)
    return model


if __name__ == '__main__':
    print('GG')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_config', type=str)
    parser.add_argument('--seg_ckpt', type=str)
    parser.add_argument('--act_config', type=str)
    parser.add_argument('--act_ckpt', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--sample_video', help='parse video frame', action='store_true')
    parser.add_argument('--output_dir', type=str, default='../../data/pred')
    parser.add_argument('--gt_dir', type=str, default='../../data/gt')
    parser.add_argument('--pred_seg', help='predict segmentation mask', action='store_true')
    parser.add_argument('--pred_act', help='predict action', action='store_true')
    parser.add_argument('--eval', help='evaluate metrics', action='store_true')
    parser.add_argument('--tta', help='use TTA module', action='store_true')
    # # parser.add_argument('--config_workspace', type=str, default='./configs/config_aia.yaml')
    # parser.add_argument('--config_encoder', type=str)
    # parser.add_argument('--ckpt_encoder', type=str)
    # parser.add_argument('--workers', type=int, default=0)
    # parser.add_argument('--bs', type=int, default=2)

    args = parser.parse_args()
    print('inference \n', args)

    inference(args)

