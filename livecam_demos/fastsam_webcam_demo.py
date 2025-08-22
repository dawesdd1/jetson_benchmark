"""
python /home/copter/jetson_benchmark/livecam_demos/fastsam_webcam_demo.py
conda activate nanosam_arm64

"""

import os
import sys
import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from random import randint
from typing import List, Tuple, Any
import time
import tensorrt