from typing import TYPE_CHECKING

from magicgui.widgets import Container, create_widget
from qtpy.QtWidgets import QVBoxLayout, QFileDialog, QWidget
import os

if TYPE_CHECKING:
    import napari


# if we want even more control over our widget, we can use
# magicgui `Container`
# class DataReader(Container):
#     def __init__(self, viewer: "napari.viewer.Viewer"):
#         super().__init__()
#         self._viewer = viewer

#         # create a widget allowing the user to select a folder containing images
#         self.img_dir = create_widget(annotation=os.PathLike, label="Image directory")
#         self.img_dir.changed.connect(self._on_img_dir_change)

#         self.mask_dir = create_widget(annotation=os.PathLike, label="Mask directory")
#         self.mask_dir.changed.connect(self._on_mask_dir_change)

#     def _on_img_dir_change(self, event):
#         print(self.img_dir.value)

#     def _on_mask_dir_change(self, event):
#         print(self.mask_dir.value)

class DataReader(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.img_dir = QFileDialog()
        self.mask_dir = QFileDialog()

        self.img_dir.setFileMode(QFileDialog.Directory)
        self.mask_dir.setFileMode(QFileDialog.Directory)

        self.img_dir.fileSelected.connect(self._on_img_dir_change)
        self.mask_dir.fileSelected.connect(self._on_mask_dir_change)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.img_dir)
        self.layout().addWidget(self.mask_dir)

    def _on_img_dir_change(self, event):
        print(self.img_dir.directory())

    def _on_mask_dir_change(self, event):
        print(self.mask_dir.directory())

# class ExampleQWidget(QWidget):
#     # your QWidget.__init__ can optionally request the napari viewer instance
#     # use a type annotation of 'napari.viewer.Viewer' for any parameter
#     def __init__(self, viewer: "napari.viewer.Viewer"):
#         super().__init__()
#         self.viewer = viewer

#         btn = QPushButton("Click me!")
#         btn.clicked.connect(self._on_click)

#         self.setLayout(QHBoxLayout())
#         self.layout().addWidget(btn)

#     def _on_click(self):
#         print("napari has", len(self.viewer.layers), "layers")
