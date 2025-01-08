# model/networks/__init.py

from .encoder import ResNet
from .decoder import Decoder

__all__ = ["ResNet", "Decoder"]