FROM nvcr.io/nvidia/pytorch:22.07-py3

COPY requirements.txt /tmp/pip-tmp/
# OpenCV Fix error ref: https://itsmycode.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/

RUN apt-get update

# ref: https://serverfault.com/a/992421
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip3  --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

RUN useradd -ms /bin/bash sar-rarp50
RUN mkdir workspace && chown sar-rarp50 /workspace
USER sar-rarp50
WORKDIR ./workspace

COPY . .


# ENTRYPOINT [ "python", "-m", "scripts.sarrarp50"]
# ENTRYPOINT ["ls"]
CMD ["python", "AIA-Noobs/inference.py", \
"--seg_config", "ckpt_cfg/20220820-1844.yaml", \
"--seg_ckpt", "ckpt_cfg/20220820-1844-best.pth", \
"--act_config", "ckpt_cfg/20220824-0538.yaml", \
"--act_ckpt", "ckpt_cfg/20220824-0538.pth", \
"--data_dir", "../../data/test", \
"--output_dir", "../../data/pred", \
#"--gt_dir", "../../data/gt", \
"--sample_video", \
"--pred_seg", \
"--pred_act", \
"--tta"]
#"--eval"]

#CMD ["python", "AIA-Noobs/inference.py", \
#"--seg_config", "ckpt_cfg/20220819-0124.yaml", \
#"--seg_ckpt", "ckpt_cfg/20220819-0124-best.pth", \
#"--data_dir", "../data/val", \
#"--output_dir", "../data/pred", \
#"--gt_dir", "../data/gt", \
#"--sample_video"]