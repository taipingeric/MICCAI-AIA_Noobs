FROM python:3.10.5-bullseye

COPY requirements.txt /tmp/pip-tmp/
# OpenCV Fix error ref: https://itsmycode.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/
RUN apt-get update && apt-get install -y python3-opencv
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
"--seg_config", "ckpt_cfg/20220819-0124.yaml", \
"--seg_ckpt", "ckpt_cfg/20220819-0124-best.pth", \
"--act_config", "ckpt_cfg/20220824-0538.yaml", \
"--act_ckpt", "ckpt_cfg/20220824-0538.pth", \
"--data_dir", "../data/val", \
"--output_dir", "../data/pred", \
"--gt_dir", "../data/gt", \
"--pred_act", \
"--eval", \
"--pred_seg", \
"--sample_video"]

#CMD ["python", "AIA-Noobs/inference.py", \
#"--seg_config", "ckpt_cfg/20220819-0124.yaml", \
#"--seg_ckpt", "ckpt_cfg/20220819-0124-best.pth", \
#"--data_dir", "../data/val", \
#"--output_dir", "../data/pred", \
#"--gt_dir", "../data/gt", \
#"--sample_video"]