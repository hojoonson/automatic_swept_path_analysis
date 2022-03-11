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

### 4. Start Manual Labelling

'''
cd train
python automatic_labelling.py

## Manual
### Q, E: rotaition(CCW, CW)
### Arrows: move up, down, left, right
### T: True
### F: False
### Caution: Don't press the down arrow first
'''
