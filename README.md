# MICCAI-AIA_Noobs

Team name: AIA-Noobs

Authors: Chih-Yang Li, Shu-Yu Huang


## Docker

ref: https://www.synapse.org/#!Synapse:syn27618412/wiki/618483

### Pull Image
```bash
docker pull taipingeric/miccai2022 
```

### Build from this repo
```bash
docker image build -t taipingeric/miccai2022 .
```

### Run inference
the following command runs 3 steps
1. Sample videos frames in **_data_dir_**
2. Generate mask predictions
3. Generate action predictions from predicted masks in **_2._** 

**NOTE**: Replace the **PATH_ROOT_DATA_DIR** with your path, and put the video dirs into **PATH_ROOT_DATA_DIR/test** dir. 

The predictions will be in **pred** dir.

```
PATH_ROOT_DATA_DIR
├── test
│    ├── video_1
│    │      ├──── video_left.avi
│    │      └──── rgb (generated)
│    │              ├──── 000000000.png
│    │              └──── ...
│    ├── ...
│    └── video_3
│           ├──── video_left.avi
│           └──── rgb (generated)
│                   ├──── 000000000.png
│                   └──── ...
└── pred (generated)
     ├── video_1
     │      ├──── action_discrete.txt
     |      └──── segmentation
     │              ├── 000000000.png
     |              └   ...
     ├── ...
     └── video_3
            ├────action_discrete.txt
            └────segmentation
                    ├── 000000000.png
                    └   ...
```

```bash
docker container run --rm --gpus=all --user=1000:1000 \
        -v PATH_ROOT_DATA_DIR:/data/ \
        taipingeric/miccai2022 \
        python AIA-Noobs/inference.py \
        --data_dir ../../data/test \
        --seg_config ckpt_cfg/20220820-1844.yaml \
        --seg_ckpt ckpt_cfg/20220820-1844-best.pth \
        --act_config ckpt_cfg/20220824-0538.yaml \
        --act_ckpt ckpt_cfg/20220824-0538.pth \
        --sample_video --pred_act --pred_seg --tta \
```


# ROOT_DIR/AIA-Noobs Usage
MICCAI2022 challenge

Official evaluation script: https://github.com/surgical-vision/SAR_RARP50-evaluation

## Training

### Segmentation training

```bash
python train.py --config './configs/config.yaml' --config_workspace './configs/config_aia.yaml' --workers 3 --bs 4
```

#### Action training
```bash
python train_eric.py --config './configs/config.yaml' --config_workspace '.\configs\config_lab.yaml' --workers 0 --bs 64 --config_encoder '../miccai_config/20220812-0040.yaml' --ckpt_encoder '../miccai_ckpt/20220812-0040-best.pth'
```


## Dataset EDA

* Videos
    * num: 45
    * frame sample rate: 6
* Segmentation
    * num of masks: 13324
    * num class (w/ bg): 10
```json
{
    "Tool clasper": 1,
    "Tool wrist": 2,
    "Tool shaft": 3,
    "Suturing needle": 4,
    "Thread": 5,
    "Suction tool": 6,
    "Needle Holder": 7,
    "Clamps": 8,
    "Catheter": 9
}
```
    
* Action:
    * num of frames: 133036
    * num class: 8

        ```txt
        0 Other
        1 Picking-up the needle
        2 Positioning the needle tip
        3 Pushing the needle through the tissue
        4 Pulling the needle out of the tissue
        5 Tying a knot
        6 Cutting the suture
        7 Returning/dropping the needle
        ```

    * label format: action_discrete.txt
        ```txt
        000000000,0
        000000006,0
        000000012,0
        000000018,0
        000000024,0
        000000030,0
        000000036,0
        000000042,0
        ```
