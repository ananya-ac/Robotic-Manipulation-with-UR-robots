"""JPEG 2000 numcodecs codec for image arrays, config-compatible with diffusion_policy.

diffusion_policy reads image zarrs with its vendored codec
(diffusion_policy/codecs/imagecodecs_numcodecs.py, codec id 'imagecodecs_jpeg2k', for
imagecodecs 2022.9.26). Current imagecodecs releases ship a codec with the same id but more
config keys (mct, planar, skipres, squeeze), which that vendored class rejects when it rebuilds
the codec from a zarr's metadata ("unexpected keyword argument 'mct'"). This codec stores exactly
the vendored class's config keys and calls imagecodecs' jpeg2k_encode / jpeg2k_decode functions
(stable across both versions), so datasets written in the ROS venv open unchanged in the robodiff
env after its register_codecs().
"""

import numpy as np
from numcodecs import register_codec
from numcodecs.abc import Codec


class Jpeg2k(Codec):
    codec_id = "imagecodecs_jpeg2k"

    def __init__(self, level=None, codecformat=None, colorspace=None, tile=None, reversible=None,
                 bitspersample=None, resolutions=None, numthreads=None, verbose=0):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        import imagecodecs
        img = np.asarray(buf)
        while img.ndim > 3 and img.shape[0] == 1:     # a (1, H, W, C) chunk -> (H, W, C) image
            img = img[0]
        return imagecodecs.jpeg2k_encode(img, level=self.level, codecformat=self.codecformat,
                                         colorspace=self.colorspace, tile=self.tile,
                                         reversible=self.reversible, bitspersample=self.bitspersample,
                                         resolutions=self.resolutions, numthreads=self.numthreads,
                                         verbose=self.verbose)

    def decode(self, buf, out=None):
        import imagecodecs
        img = imagecodecs.jpeg2k_decode(buf, verbose=self.verbose, numthreads=self.numthreads)
        if out is None:
            return img
        out = np.asarray(out)
        out.reshape(img.shape)[...] = img
        return out


def register_image_codecs():
    """Register Jpeg2k with numcodecs (replacing any other 'imagecodecs_jpeg2k' registration)."""
    register_codec(Jpeg2k)
