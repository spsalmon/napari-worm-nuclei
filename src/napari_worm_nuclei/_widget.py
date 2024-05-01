from typing import TYPE_CHECKING

from magicgui.widgets import Container, create_widget
from qtpy.QtWidgets import QVBoxLayout, QFileDialog, QWidget, QLineEdit, QPushButton
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

        # Set up the layout
        layout = QVBoxLayout()

        self.img_dir_path = ""
        self.mask_dir_path = ""

        # Create widgets for image directory selection
        self.img_dir_edit = QLineEdit()  # To display the path
        self.img_dir_button = QPushButton("Select Image Directory")
        self.img_dir_button.clicked.connect(self.select_img_dir)
        layout.addWidget(self.img_dir_edit)
        layout.addWidget(self.img_dir_button)

        # Create widgets for mask directory selection
        self.mask_dir_edit = QLineEdit()  # To display the path
        self.mask_dir_button = QPushButton("Select Mask Directory")
        self.mask_dir_button.clicked.connect(self.select_mask_dir)
        layout.addWidget(self.mask_dir_edit)
        layout.addWidget(self.mask_dir_button)

        # Set the layout to the QWidget
        self.setLayout(layout)

    def select_img_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.img_dir_edit.setText(dir_path)
            self.img_dir_path = dir_path

    def select_mask_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Mask Directory")
        if dir_path:
            self.mask_dir_edit.setText(dir_path)
            self.mask_dir_path = dir_path




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
