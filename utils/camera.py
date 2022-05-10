'''
This is a realtime multi-camera pipeline based on NVIDIA JETSON GPU platform and gstreamer library. 
The part code is referred by Lei Xu from 
https://github.com/NVIDIA-AI-IOT/jetson-multicamera-pipelines/blob/main/jetmulticam/pipelines/basepipeline.py
https://github.com/NVIDIA-AI-IOT/jetson-multicamera-pipelines/blob/main/jetmulticam/bins/encoder.py
The part code is written by Lei Xu based on gstreamer library.
'''
import gi
import sys

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst

import time
import numpy as np
from threading import Thread
from .gst import _make_element_safe, _sanitize, bus_call, _err_if_none

# Added by Lei using rtsp cam
def make_rtsp_camera_configured(location) -> Gst.Element:
    """
    This function is used to configure rtsp camera source.
    """
    cam = _make_element_safe("rtspsrc")
    # location="rtsp://88.113.26.101:8554/pistream"
    cam.set_property("location", location)
    cam.set_property("latency", 10)

    return cam

def cb_newpad(decodebin, decoder_src_pad, data):

    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

# This function is used to create rtsp camera
def make_rtsp_cam_bin(uri) -> Gst.Bin:
    print("Creating RTSP camera source bin...")
    bin = Gst.Bin()
       
    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = _make_element_safe("uridecodebin", "uri-decode-bin")
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, bin)
    uri_decode_bin.connect("child-added", decodebin_child_added, bin)
        # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    bin.add(uri_decode_bin)
    bin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    print("Returning RTSP camera source bin...")

    return bin

# you could write other types of cameras based on the gstreamer library
def make_argus_camera_configured(sensor_id, bufapi_version=1) -> Gst.Element:
    """
    Make pre-configured camera source, so we have consistent setting across sensors
    Switch off defaults which are not helpful for machine vision like edge-enhancement
    """
    cam = _make_element_safe("nvarguscamerasrc")
    cam.set_property("sensor-id", sensor_id)
    cam.set_property("bufapi-version", bufapi_version)
    cam.set_property("wbmode", 1)  # 1=auto, 0=off,
    cam.set_property("aeantibanding", 3)  # 3=60Hz, 2=50Hz, 1=auto, 0=off
    cam.set_property("tnr-mode", 0)
    cam.set_property("ee-mode", 0)
    cam.set_property("silent", True)

    return cam

def make_nvenc_bin(filepath, bitrate=int(20e6)) -> Gst.Bin:
    h264sink = Gst.Bin()

    # Create video converter
    conv = _make_element_safe("nvvideoconvert")

    # H264 encoder
    enc = _make_element_safe("nvv4l2h264enc")
    enc.set_property("bitrate", bitrate)
    enc.set_property("bufapi-version", 1)
    enc.set_property("maxperf-enable", True)

    # parser, mux
    parser = _make_element_safe("h264parse")
    mux = _make_element_safe("matroskamux")

    # filesink
    filesink = _make_element_safe("filesink")
    filesink.set_property("sync", 0)
    filesink.set_property("location", filepath)

    # Add elements to bin before linking
    for el in [conv, enc, parser, mux, filesink]:
        h264sink.add(el)

    # Link bin elements
    conv.link(enc)
    enc.link(parser)
    parser.link(mux)
    mux.link(filesink)

    enter_pad = _sanitize(conv.get_static_pad("sink"))
    gp = Gst.GhostPad.new(name="sink", target=enter_pad)
    h264sink.add_pad(gp)

    return h264sink

class baseCamera:
    def __init__(self, **kwargs):
        print("...")
        # Gstreamer init
        GObject.threads_init()
        Gst.init(None)

        # create an event loop and feed gstreamer bus messages to it
        self._mainloop = GObject.MainLoop()

        self._p = self._create_pipeline(**kwargs)
        #self._log = logging.getLogger()
        self._bus = self._p.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", bus_call, self._mainloop)

    def __del__(self):
        self.stop()
    
    def start(self):
        self._p.set_state(Gst.State.PLAYING)
        print("pipline is set to start...")
        self.wait_ready()
        self._start_ts = time.perf_counter()

    def stop(self):
        self._p.send_event(Gst.Event.new_eos())
        self._p.set_state(Gst.State.PAUSED)
        self._p.set_state(Gst.State.NULL)

    def running(self):
        _, state, _ = self._p.get_state(1)
        return True if state == Gst.State.PLAYING else False

    def wait_ready(self):
        while not self.running():
            print("pipline is not ready...")
            time.sleep(0.1)
