# automatic_swept_path_analysis

### 1. Docker build with shell script file 

```
docker_build.sh
```

### 2. Docker run with shell script file

```
docker_run.sh
```

### 3. Start Automatic Labelling

```
cd train
python automatic_labelling.py
```

# How to use Manual Labelling

### 1. Start Manual Labelling

```
cd train
python automatic_labelling.py
```

### 2. Manual

<strong>Q, W</strong>: Rotation(CCW, CW)

<strong>Arrows</strong>: Move up, down, left, right

<strong>T</strong>: Save as True

<strong>F</strong>: Save as False

<strong>R</strong>: Retry

<strong>C</strong>: Capture Current Pygame Image


### Caution: Don't press the down arrow first