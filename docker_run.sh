xhost +
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/projects/automatic_swept_path_analysis:/app pygame_tf