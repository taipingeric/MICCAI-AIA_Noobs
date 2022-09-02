# AIA-Noobs
MICCAI2022 challenge

Official evaluation script: https://github.com/surgical-vision/SAR_RARP50-evaluation

###Training

### Segmentation

```bash
python train.py --config './configs/config.yaml' --config_workspace './configs/config_aia.yaml' --workers 3 --bs 4
```

### Action
```bash
python train_action.py --config './configs/config.yaml' --config_workspace '.\configs\config_lab.yaml' --workers 0 --bs 64
```

#### Action inner temp code
```bash
python train_eric.py --config './configs/config.yaml' --config_workspace '.\configs\config_lab.yaml' --workers 0 --bs 64 --config_encoder '../miccai_config/20220812-0040.yaml' --ckpt_encoder '../miccai_ckpt/20220812-0040-best.pth'
```

### Docker

ref: https://www.synapse.org/#!Synapse:syn27618412/wiki/618483

#### Clone official repo
```bash
git clone https://github.com/surgical-vision/SAR_RARP50-evaluation && cd ./SAR_RARP50-evaluation
```
#### Build docker image
```bash
docker image build -t sarrarp_tk .
```

#### Sample frames
```bash
docker container run --rm \
                     -v /Users/cyli/Desktop/MICCAI-docker/datasets/:/data/ \
                     sarrarp_tk \
                     unpack /data/ -j4 -r                      
```

#### Generating mock predictions
```bash
docker container run --rm \
                     -v /Users/cyli/Desktop/MICCAI-docker/datasets/:/data/ \
                     sarrarp_tk \
                     generate /data/test/ /data/mock_predictins
```

#### Evaluation
```bash
docker container run --rm \
                     -v /Users/cyli/Desktop/MICCAI-docker/datasets/:/data/ \
                     sarrarp_tk \
                     evaluate /data/gt/ /data/mock_predictins
```



### Dataset EDA

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