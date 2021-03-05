### Instructions for generating CLEVR-Hans

This is a very brief list of instructions on how to generate a CLEVR-Hans dataset using the blender framework and the 
generation script that is based on the original CLEVR script.

For using this script you will require the blender 2.78c version

For easier handling:
create a screen: screen -S clevr_hans

Then:
1. cd to clevr-hans-dataset-gen/docker folder
2. docker build -t blender_python_clevr -f Dockerfile .
3. docker run -it -v /pathto/clevr-hans-dataset-gen:/home/workspace/clevr-hans-dataset-gen --name clevr_hans --entrypoint='/bin/bash' blender_python_clevr
4. cd ../home/workspace/clevr-hans-dataset-gen/image_generation/
5. ./run_scripts/run_conf_3.sh
