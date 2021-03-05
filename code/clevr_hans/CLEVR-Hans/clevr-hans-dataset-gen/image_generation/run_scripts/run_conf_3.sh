#!/bin/bash

blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=3000
NUM_VAL_SAMPLES=750
NUM_TEST_SAMPLES=750

NUM_PARALLEL_THREADS=5
NUM_THREADS=4
MIN_OBJECTS=3
MAX_OBJECTS=10
MAX_RETRIES=30

#----------------------------------------------------------#

# generate training images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/conf_3/train/images/ --output_scene_dir ../output/conf_3/train/scenes/ --output_scene_file ../output/conf_3/train/CLEVR_HANS_scenes_train.json --filename_prefix CLEVR_Hans --max_retries $MAX_RETRIES --num_images $NUM_TRAIN_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 480 --height 320 --properties_json data/properties.json --conf_class_combos_json data/Clevr_Hans_ConfClasses_3.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/conf_3/train/

#----------------------------------------------------------#

# generate test images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/conf_3/test/images/ --output_scene_dir ../output/conf_3/test/scenes/ --output_scene_file ../output/conf_3/test/CLEVR_HANS_scenes_test.json --filename_prefix CLEVR_Hans --num_images $NUM_TEST_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 480 --height 320 --properties_json data/properties.json --conf_class_combos_json data/Clevr_Hans_GTClasses_3.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/conf_3/test/

#----------------------------------------------------------#

# generate confounded val images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/conf_3/val/images/ --output_scene_dir ../output/conf_3/val/scenes/ --output_scene_file ../output/conf_3/val/CLEVR_HANS_scenes_val.json --filename_prefix CLEVR_Hans --num_images $NUM_VAL_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 480 --height 320 --properties_json data/properties.json --conf_class_combos_json data/Clevr_Hans_ConfClasses_3.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/conf_3/val/
