This demo is built by Lei Xu using the Gstreamer library. 
This is a living multiple RTSP cameras demo based on Jetson hardware platform and DeepStream SDK. 
The code is written by Lei Xu based on DeepStream SDK documents and Git repository as: 
[jetson-multicamera](https://github.com/NVIDIA-AI-IOT/jetson-multicamera-pipelines) and
[DeepStream](https://developer.nvidia.com/deepstream-sdk) and [Tutorial](https://developer.nvidia.com/blog/intelligent-video-analytics-deepstream-sdk-3-0/).


## TODO


- [training own models]
- [deploying on cloud platform]

## Hardware requirements
 1. Jetson Nano  
 2. several rtsp cameras (I used two phones with the app IP Camera)

# Dependencies:

```bash
1. sudo apt-get install python3-dev python-gi-dev libgstreamer1.0-dev -y
2. sudo apt-get install deepstream-5.1 -y
```

# Download models
```bash
python3 ./models/main.py
```

# Run code
```bash 
1. source scripts/env_vars.sh
2. python3 main.py
```
