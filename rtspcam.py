# This is for multi rtsp cameras

# Gstreamer imports
import gi
import numpy as np
import time

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from utils.gst import _err_if_none, _make_element_safe, _sanitize, bus_call
from utils.camera import make_rtsp_cam_bin, baseCamera, make_nvenc_bin

class rtspCameraPipeline(baseCamera):
    def __init__(self, rtsp_uri, frame_shape, model, nvinfer_interval, *args, **kwargs):
        """
        rtsp_uri: list of rtsp_uri
        """
        self._rtsp_uri = rtsp_uri
        self._frame_shape = frame_shape
        self._model = model
        self._nvinfer_interval = nvinfer_interval

        # runtime parameters
        self.num_cams = len(self._rtsp_uri)
        self.images = [np.empty(self._frame_shape) for _ in range(0, self.num_cams)]
        self.detections = [[] for _ in range(0, self.num_cams)]  # dets for each camera
        self.frame_n = [-1 for _ in range(0, self.num_cams)]
        self.det_n = [-1 for _ in range(0, self.num_cams)]

        super().__init__(**kwargs)
    
    def _create_pipeline(self, **kwargs) -> Gst.Pipeline:
        p = self._create_rtsp_pipeline(self._rtsp_uri, self._model, **kwargs)
        return p

    def _create_rtsp_pipeline(
        self,
        rtsp_uri,
        model,
        save_video=False,
        save_video_folder="",
        display=True) -> Gst.Pipeline:
        # gst pipeline object
        pipeline = Gst.Pipeline()
        _err_if_none(pipeline)
        sources = self._make_sources(rtsp_uri)

        # Create muxer
        mux = _make_element_safe("nvstreammux")
        mux.set_property("live-source", True)
        mux.set_property("width", self._frame_shape[0])
        mux.set_property("height", self._frame_shape[1])
        mux.set_property("batch-size", self.num_cams)
        mux.set_property("batched-push-timeout", 4000000)

        # Create nvinfer 
        nvinfer = _make_element_safe("nvinfer")
        nvinfer.set_property("config-file-path", self._model)
        nvinfer.set_property("batch-size", self.num_cams)
        nvinfer.set_property("interval", self._nvinfer_interval)  # to infer every n batches

        # nvvideoconvert -> nvdsosd -> nvegltransform -> sink
        nvvidconv = _make_element_safe("nvvideoconvert", "convertor") 
        nvosd = _make_element_safe("nvdsosd")
        tiler = _make_element_safe("nvmultistreamtiler")
    
        # max 3 cameras in a row.
        n_cols = min(self.num_cams, 3)
        tiler.set_property("rows", self.num_cams // n_cols)
        tiler.set_property("columns", n_cols)
        tiler.set_property("width", self._frame_shape[0])
        tiler.set_property("height", self._frame_shape[1])

        # Render with EGL GLE sink
        transform = _make_element_safe("nvegltransform")
        renderer = _make_element_safe("nveglglessink")

        tee = _make_element_safe("tee")

        sinks = []
        if save_video:
            ts = time.strftime("%Y-%m-%dT%H-%M-%S%z")
            encodebin = make_nvenc_bin(
                filepath=save_video_folder + f"/rtspcam{ts}.mkv"
            )
            sinks.append(encodebin)

        if display:
            renderer.set_property("sync", 0)
            renderer.set_property("qos", 0)
            sinks.append(renderer)

        if len(sinks) == 0:
            # If no other sinks are added we terminate with fakesink
            fakesink = _make_element_safe("fakesink")
            sinks.append(fakesink)

        # Add all elements to the pipeline
        elements = [*sources, mux, nvinfer, tiler, nvvidconv, nvosd, tee, transform, *sinks]
        for el in elements:
            pipeline.add(el)

        for (idx, source) in enumerate(sources):
            srcpad_or_none = source.get_static_pad(f"src")
            sinkpad_or_none = mux.get_request_pad(f"sink_{idx}")
            srcpad = _sanitize(srcpad_or_none)
            sinkpad = _sanitize(sinkpad_or_none)
            srcpad.link(sinkpad)

        mux.link(nvinfer)
        nvinfer.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(tee)

        # Link tees to sinks
        for idx, sink in enumerate(sinks):
            # Use queues for each sink. This ensures the sinks can execute in separate threads
            queue = _make_element_safe("queue")
            pipeline.add(queue)
            # tee.src_%d -> queue
            srcpad_or_none = tee.get_request_pad(f"src_{idx}")
            sinkpad_or_none = queue.get_static_pad("sink")
            srcpad = _sanitize(srcpad_or_none)
            sinkpad = _sanitize(sinkpad_or_none)
            srcpad.link(sinkpad)
            # queue -> sink
            queue.link(transform)
            transform.link(sink)
        return pipeline

    @staticmethod
    def _make_sources(rtsp_uri: list) -> list:
        # Create pre-configured sources with appropriate type: argus or v4l
        sources = []
        for c in rtsp_uri:
            source = make_rtsp_cam_bin(c)
            sources.append(source)
        return sources




