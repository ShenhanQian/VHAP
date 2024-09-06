#!/bin/bash


# Define a list of subjects and sequences
# SUBJECTS=("074" "104" "140" "165" "210" "218" "238" "253" "264" "302" "304" "306")  # 12
# SUBJECTS=("074" "104" "140" "165" "175" "210" "218" "238" "253" "264" "302" "304" "306" "460")  # 14
# SUBJECTS=("306")  # tmp

# Loop through subjects and sequences and modify the command accordingly
for SUBJECT in "${SUBJECTS[@]}"; do
    export_dir=export/nersemble

    #------- combine 10 -------#
    COMMAND="python vhap/combine_nerf_datasets.py \
    --src_folders \
    $export_dir/${SUBJECT}_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EMO-2_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EMO-3_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EMO-4_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-2_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-3_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-4_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-5_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-8_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    $export_dir/${SUBJECT}_EXP-9_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    --tgt_folder \
    $export_dir/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS4_whiteBg_staticOffset_maskBelowLine \

    "

    #------- combine 20 -------#
    # COMMAND="python vhap/combine_nerf_datasets.py \
    # --src_folders \
    # $export_dir/${SUBJECT}_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EMO-2_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EMO-3_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EMO-4_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-2_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-3_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-4_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-5_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-8_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-9_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-01_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-02_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-03_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-04_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-05_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-06_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-07_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-08_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-09_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-10_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # --tgt_folder \
    # $export_dir/UNION20_${SUBJECT}_EMO1234EXP234589SEN_v16_DS4_whiteBg_staticOffset_maskBelowLine \

    # "

    #------- combine short -------#
    # COMMAND="python vhap/combine_nerf_datasets.py \
    # --src_folders \
    # $export_dir/${SUBJECT}_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_EXP-2_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_SEN-10_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # $export_dir/${SUBJECT}_FREE_v16_DS4_whiteBg_staticOffset_maskBelowLine \
    # --tgt_folder \
    # $export_dir/UNIONshort_${SUBJECT}_EMO1EXP12SEN10FREE_v16_DS4_whiteBg_staticOffset_maskBelowLine \

    # "

    #------- zip -------#
    # COMMAND="zip -r $export_dir/$SUBJECT.zip $export_dir/$SUBJECT* $export_dir/UNION_$SUBJECT*"

    #======= Run =======#
    # Execute the command (remove the echo if you want to actually run the command)

    $COMMAND
    # echo $COMMAND

done
