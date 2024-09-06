#!/bin/bash

# Define a list of sequences

# SEQUENCES=("obama" "biden" "justin" "nf_01" "nf_03" "person_0004" "malte_1" "bala" "wojtek_1" "marcel")  # 10
# SEQUENCES=("obama")

DATA_FOLDER="data/monocular"

# Loop through sequences and modify the command accordingly
for SEQUENCE in "${SEQUENCES[@]}"; do

    JOB_NAME="vhap_${SEQUENCE}"

    #======= Preprocess =======#
    RAW_VIDEO_PATH="${DATA_FOLDER}/${SEQUENCE}.mp4"
    PREPROCESS_COMMAND="
    python vhap/preprocess_video.py --input ${RAW_VIDEO_PATH} \
    --matting_method robust_video_matting
    "

    #======= Track =======#
    TRACK_OUTPUT_FOLDER="output/monocular/${SEQUENCE}_whiteBg_staticOffset"
    TRACK_COMMAND="
    python vhap/track.py --data.root_folder ${DATA_FOLDER} \
    --exp.output_folder $TRACK_OUTPUT_FOLDER \
    --data.sequence $SEQUENCE \

    "

    #======= Export =======#
    EXPORT_OUTPUT_FOLDER="export/monocular/${SEQUENCE}_whiteBg_staticOffset_maskBelowLine"
    EXPORT_COMMAND="python vhap/export_as_nerf_dataset.py \
    --src_folder ${TRACK_OUTPUT_FOLDER} \
    --tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white \
    
    "

    #======= Run =======#
    # Execute the command (remove the echo if you want to actually run the command)

    #------- check completeness -------#
    # last_folder=$(find "$TRACK_OUTPUT_FOLDER" -maxdepth 1 -type d | sort | tail -n 1)
    # # if [ ! -d $last_folder/eval_30 ]; then
    # if [ ! -e $last_folder/tracked_flame_params_30.npz ]; then
    #     echo $last_folder
    # fi

    #------- create video -------#
    # last_folder=$(find "$TRACK_OUTPUT_FOLDER" -maxdepth 1 -type d | sort | tail -n 1)
    # video_folder=$last_folder/eval_30
    # ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i "${video_folder}/image_grid/*.jpg" -pix_fmt yuv420p $video_folder/image_grid.mp4
    
    #------- rename -------#
    # mv $TRACK_OUTPUT_FOLDER $TRACK_OUTPUT_FOLDER

    #------- only preprocess -------#
    # COMMAND="$HOME/local/usr/bin/isbatch.sh $JOB_NAME $PREPROCESS_COMMAND"
    # COMMAND="$PREPROCESS_COMMAND"

    #------- only track -------#
    # COMMAND="$HOME/local/usr/bin/isbatch.sh $JOB_NAME $TRACK_COMMAND"
    # COMMAND="$TRACK_COMMAND"
    
    #------- only export -------#
    # COMMAND="$HOME/local/usr/bin/isbatch.sh $JOB_NAME $EXPORT_COMMAND"
    # COMMAND="$EXPORT_COMMAND"

    #------- track and export -------#
    # COMMAND="$HOME/local/usr/bin/isbatch.sh $JOB_NAME $TRACK_COMMAND && $EXPORT_COMMAND"
    # COMMAND="$TRACK_COMMAND && $EXPORT_COMMAND"

    
    # echo $COMMAND
    $COMMAND
    sleep 1
done
