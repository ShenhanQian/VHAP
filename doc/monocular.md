## For Monocular Videos

<div align="center"> 
  <img src="../asset/monocular_obama.gif" width=100%>
</div>

### 1. Preprocess

This step extracts frames from video(s), then run foreground matting for each frame, which requires GPU.

```shell
SEQUENCE="obama.mp4"

python vhap/preprocess_video.py \
--input data/monocular/${SEQUENCE} \
--matting_method robust_video_matting
```

- `--matting_method robust_video_matting`: Use RobustVideoMatting due to lack of a background image.
- (Optional) `--downsample_scales 2`: Generate downsampled versions of images in a scale such as 2. (Image size lower than 1024 is preferred for efficiency.)

### 2. Align and track faces

This step automatically detects facial landmarks if absent, then begin FLAME tracking. We initialize shape and appearance parameters on the first frame, then do a sequential tracking of following frames. After the sequence tracking, we conduct 30 epochs of global tracking, which optimize all the parameters on a random frame in each iteration.

```shell
SEQUENCE="obama"
TRACK_OUTPUT_FOLDER="output/monocular/${SEQUENCE}_whiteBg_staticOffset"

python vhap/track.py --data.root_folder "data/monocular" \
--exp.output_folder $TRACK_OUTPUT_FOLDER \
--data.sequence $SEQUENCE \
# --data.n_downsample_rgb 2  # Only specify this if you have generate downsampled images during preprocessing.
```

Optional arguments

- `--model.no_use_static_offset`: disable static offset for FLAME (very stable, but less aligned facial geometry)

  > Disabling static offset will automatically triggers `--model.occluded hair`, which is crucial to prevent the head from growing too larger to align with the top of hair.

- `--exp.no_photometric`: track only with landmark (very fast, but coarse)

### 3. Export tracking results into a NeRF-style dataset

Given the tracked FLAME parameters from the above step, you can export the results to form a NeRF/3DGS style sequence, consisting of image folders and a `transforms.json`.

```shell
SEQUENCE="obama"
TRACK_OUTPUT_FOLDER="output/monocular/${SEQUENCE}_whiteBg_staticOffset"
EXPORT_OUTPUT_FOLDER="export/monocular/${SEQUENCE}_whiteBg_staticOffset_maskBelowLine"

python vhap/export_as_nerf_dataset.py \
--src_folder ${TRACK_OUTPUT_FOLDER} \
--tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white
```
