#!/bin/bash

# Define a list of subjects and sequences

# SUBJECTS=("074" "104" "140" "165" "175" "210" "218" "238" "253" "264" "302" "304" "306" "460")  # 14
# SUBJECTS=("306")

# SEQUENCES=("EMO-1" "EMO-2" "EMO-3" "EMO-4" "EXP-1" "EXP-2" "EXP-3" "EXP-4" "EXP-5" "EXP-8" "EXP-9" "FREE")  # 11
# SEQUENCES=("SEN-01" "SEN-02" "SEN-03" "SEN-04" "SEN-05" "SEN-06" "SEN-07" "SEN-08" "SEN-09" "SEN-10")  # 10
# SEQUENCES=("EMO-1")


DATA_FOLDER="data/nersemble"

# Loop through subjects and sequences and modify the command accordingly
for SUBJECT in "${SUBJECTS[@]}"; do
    for SEQUENCE in "${SEQUENCES[@]}"; do

        JOB_NAME="vhap_${SUBJECT}_${SEQUENCE}"

        #======= Preprocess =======#
        VIDEO_FOLDER_PATH="${DATA_FOLDER}/${SUBJECT}/${SEQUENCE}"
        PREPROCESS_COMMAND="
        python vhap/preprocess_video.py --input $VIDEO_FOLDER_PATH \
        --downsample_scales 2 4 \
        --matting_method background_matting_v2
        "

        #======= Track =======#
        TRACK_OUTPUT_FOLDER="output/nersemble/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
        TRACK_COMMAND="
        python vhap/track_nersemble.py --data.root_folder $DATA_FOLDER \
        --exp.output_folder $TRACK_OUTPUT_FOLDER \
        --data.subject $SUBJECT --data.sequence $SEQUENCE \
        --data.n_downsample_rgb 4

        "
        #======= Export =======#
        EXPORT_OUTPUT_FOLDER="export/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"
        EXPORT_COMMAND="
        python vhap/export_as_nerf_dataset.py \
        --src_folder ${TRACK_OUTPUT_FOLDER} \
        --tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white
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
done
