# Standard Library
import os
import struct

# Third Party Library
import numpy as np
import numpy.typing as npt


def load_mnist(path: str, kind: str = "train") -> (npt.NDArray, npt.NDArray):
    labels_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        path,
        "%s-labels-idx1-ubyte" % kind,
    )
    images_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        path,
        "%s-images-idx3-ubyte" % kind,
    )

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.0) - 0.5) * 2  # ピクセル値の正規化

    return images, labels
