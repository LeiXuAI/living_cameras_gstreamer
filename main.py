import sys
import os
from utils import gst, camera
from rtspcam import rtspCameraPipeline


if __name__ == "__main__":

    # you main put rtsp cameras streams as a list as the inputs
    rtsp_cameras_list = [ ]
    frame_shape = 1920, 1080, 3
    model_list = './models/peoplenet/peoplenet_gpu.txt'
    nvinfer_interval = 1

    # pipeline
    pipeline = rtspCameraPipeline(
        rtsp_cameras_list,
        frame_shape, 
        model_list,
        nvinfer_interval,
        save_video=False,
        save_video_folder=" ",
        display=True
    )

    pipeline.start()
    try:
        pipeline._mainloop.run()
    except BaseException:
        pass
    pipeline.__del__()


