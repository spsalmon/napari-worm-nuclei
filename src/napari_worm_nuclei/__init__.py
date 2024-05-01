__version__ = "0.0.1"

from ._reader import napari_get_reader
from ._widget import DataReader
from ._writer import write_multiple, write_single_image

__all__ = (
    "DataReader",
)
