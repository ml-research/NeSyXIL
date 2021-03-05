#
# @author: Wolfgang Stammer
#

FROM python:3.5.2-slim

#get latest python & blender related dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils git python3-virtualenv \
python3-dev python3-pip libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 unzip vim tmux wget bzip2 libgtk2.0-dev gcc\
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install joblib
RUN pip install opencv-python
RUN pip install cython
RUN pip install pycocotools

#get the dependencies for the script
RUN mkdir -p /local/
RUN pip install requests pathlib

#get the blender 2.78c and setup the paths
RUN cd /tmp && wget -q https://mirror.clarkson.edu/blender/release/Blender2.78/blender-2.78c-linux-glibc219-x86_64.tar.bz2 \
&& tar xf /tmp/blender-2.78c-linux-glibc219-x86_64.tar.bz2 -C /usr/bin/ && rm -r /tmp/blender-2.78c-linux-glibc219-x86_64.tar.bz2

#copy the shared lib for blender
RUN cp /usr/bin/blender-2.78c-linux-glibc219-x86_64/lib/lib* /usr/local/lib/ && ldconfig

# create blender alias
RUN echo 'alias blender="/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender"' >> ~/.bashrc

# You may need to add a .pth file to the site-packages of Blender's bundled python
RUN echo $PWD >> /usr/bin/blender-2.78c-linux-glibc219-x86_64/2.78/python/lib/python3.5/site-packages/clevr.pth

# create a workspace dir
RUN mkdir -p home/workspace/clevr-hans-dataset-gen
RUN chmod 777 home/workspace/clevr-hans-dataset-gen/

ENV PYTHONPATH "${PYTHONPATH}:./"

WORKDIR "/root"
CMD ["/bin/bash"]
