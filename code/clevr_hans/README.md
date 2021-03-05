### Instructions for generating CLEVR-Hans

1. cd into docker/ folder
2. docker build -t xil_clevr_hans -f Dockerfile .



3. docker run -it -v /pathto/clevr-hans-dataset-gen:/home/workspace/clevr-hans-dataset-gen --name clevr_hans --entrypoint='/bin/bash' blender_python_clevr
4. cd ../home/workspace/clevr-hans-dataset-gen/image_generation/
5. ./run_scripts/run_conf_3.sh
