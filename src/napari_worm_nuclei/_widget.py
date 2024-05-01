from typing import TYPE_CHECKING

from magicgui.widgets import Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QFileDialog, QWidget, QLineEdit, QPushButton, QComboBox
from towbintools.foundation import file_handling
import os
import numpy as np
import pandas as pd
import tifffile

if TYPE_CHECKING:
    import napari

def add_dir_to_experiment_filemap(experiment_filemap, dir_path, subdir_name):
    subdir_filemap = file_handling.get_dir_filemap(dir_path)
    subdir_filemap.rename(columns={"ImagePath": subdir_name}, inplace=True)
    # check if column already exists
    if subdir_name in experiment_filemap.columns:
        experiment_filemap.drop(columns=[subdir_name], inplace=True)
    experiment_filemap = experiment_filemap.merge(
        subdir_filemap, on=["Time", "Point"], how="left"
    )
    experiment_filemap = experiment_filemap.replace(np.nan, "", regex=True)
    return experiment_filemap


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
        self.filemap = pd.DataFrame()

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

        # Add a button to get the files from the selected directories
        self.get_files_button = QPushButton("Get Files")
        layout.addWidget(self.get_files_button)
        self.get_files_button.clicked.connect(self.get_files)

        # Add a time and point selection widget
        self.previous_time_button = QPushButton("Previous Time")
        self.time_combo = QComboBox()
        self.next_time_button = QPushButton("Next Time")

        self.previous_time_button.clicked.connect(self.previous_time)
        self.next_time_button.clicked.connect(self.next_time)

        self.previous_point_button = QPushButton("Previous Point")
        self.point_combo = QComboBox()
        self.next_point_button = QPushButton("Next Point")

        self.previous_point_button.clicked.connect(self.previous_point)
        self.next_point_button.clicked.connect(self.next_point)

        # Put the time widgets in a horizontal layout
        time_layout = QHBoxLayout()
        time_layout.addWidget(self.previous_time_button)
        time_layout.addWidget(self.time_combo)
        time_layout.addWidget(self.next_time_button)

        # Put the point widgets in a horizontal layout
        point_layout = QHBoxLayout()
        point_layout.addWidget(self.previous_point_button)
        point_layout.addWidget(self.point_combo)
        point_layout.addWidget(self.next_point_button)

        # Add the time and point layouts to the main layout
        layout.addLayout(time_layout)
        layout.addLayout(point_layout)

        # Add a button to load the selected image and mask
        self.load_button = QPushButton("Load Images")
        layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_images)

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

    def get_files(self):
        print("Getting files from the selected directories")
        print("Image directory:", self.img_dir_path)
        print("Mask directory:", self.mask_dir_path)

        filemap = file_handling.get_dir_filemap(self.img_dir_path)
        filemap = add_dir_to_experiment_filemap(filemap, self.mask_dir_path, "MaskPath")

        # Remove all rows with NaN values
        filemap = filemap.dropna()

        # Remove all rows with empty strings
        filemap = filemap[(filemap != "").all(axis=1)]

        self.filemap = filemap

        # Reset the combo boxes
        self.time_combo.clear()
        self.point_combo.clear()

        # Populate the time and point combo boxes
        time_values = [str(time) for time in filemap["Time"].unique()]
        self.time_combo.addItems(time_values)

        point_values = [str(point) for point in filemap["Point"].unique()]
        self.point_combo.addItems(point_values)

    def previous_time(self):
        # select the previous element in the time combo box
        current_index = self.time_combo.currentIndex()
        if current_index > 0:
            self.time_combo.setCurrentIndex(current_index - 1)
    
    def next_time(self):
        # select the next element in the time combo box
        current_index = self.time_combo.currentIndex()
        if current_index < self.time_combo.count() - 1:
            self.time_combo.setCurrentIndex(current_index + 1)
    
    def previous_point(self):
        # select the previous element in the point combo box
        current_index = self.point_combo.currentIndex()
        if current_index > 0:
            self.point_combo.setCurrentIndex(current_index - 1)
    
    def next_point(self):
        # select the next element in the point combo box
        current_index = self.point_combo.currentIndex()
        if current_index < self.point_combo.count() - 1:
            self.point_combo.setCurrentIndex(current_index + 1)

def load_images(self):
        time = int(self.time_combo.currentText())
        point = int(self.point_combo.currentText())
        print("Loading images for time:", time, "and point:", point)
        
        row = self.filemap[(self.filemap["Time"] == time) & (self.filemap["Point"] == point)]
        img_path = row["ImagePath"].values[0]
        mask_path = row["MaskPath"].values[0]

        # Read images using tifffile
        img_data = tifffile.imread(img_path)
        mask_data = tifffile.imread(mask_path)

        # Add images to Napari viewer
        self.viewer.add_image(img_data, name=f"Image at time {time} point {point}")
        self.viewer.add_labels(mask_data, name=f"Mask at time {time} point {point}")









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
