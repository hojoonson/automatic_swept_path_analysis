# Auto-Simulation for Swept Path Analysis

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

# How to generate roads

### 1. Create json file that contains road shape parameters.

```
python draw/generate_roads_v2.py
```

### 2. Load roads from json file.
```
cd train
python load_roads.py
```

You can load original image and marked image.

<img src="./images/original.png" width=300 height=300></img>
<img src="./images/marked.png" width=300 height=300></img>


# How to use Manual Labelling

### 1. Execute python code.

```
cd train
python automatic_labelling.py
```

### 2. Key Funcions

<strong>Q, W</strong>: Rotation(CCW, CW)

<strong>Arrows</strong>: Move up, down, left, right

<strong>T</strong>: Save as True

<strong>F</strong>: Save as False

<strong>R</strong>: Retry

<strong>C</strong>: Capture Current Pygame Image


### Caution: Don't press the down arrow first
