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
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.setup_directory_selection()
        self.setup_file_controls()
        self.setup_navigation_controls()
        self.setup_load_button()
        self.setLayout(self.layout)

    def setup_directory_selection(self):
        self.img_dir_path, self.mask_dir_path = "", ""
        self.img_dir_edit, self.img_dir_button = self.create_dir_selector("Select Image Directory")
        self.mask_dir_edit, self.mask_dir_button = self.create_dir_selector("Select Mask Directory")

    def create_dir_selector(self, button_label):
        dir_edit = QLineEdit()
        dir_button = QPushButton(button_label)
        dir_button.clicked.connect(lambda: self.select_directory(dir_edit))
        self.layout.addWidget(dir_edit)
        self.layout.addWidget(dir_button)
        return dir_edit, dir_button

    def select_directory(self, dir_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            dir_edit.setText(dir_path)
            return dir_path
        else:
            print("Directory selection cancelled.")
            return ""

    def setup_file_controls(self):
        self.get_files_button = QPushButton("Get Files")
        self.get_files_button.clicked.connect(self.get_files)
        self.layout.addWidget(self.get_files_button)

    def setup_navigation_controls(self):
        self.setup_time_navigation()
        self.setup_point_navigation()

    def setup_time_navigation(self):
        self.previous_time_button, self.time_combo, self.next_time_button = self.create_navigation_controls("Time")
        self.layout.addLayout(self.create_horizontal_layout(self.previous_time_button, self.time_combo, self.next_time_button))

    def setup_point_navigation(self):
        self.previous_point_button, self.point_combo, self.next_point_button = self.create_navigation_controls("Point")
        self.layout.addLayout(self.create_horizontal_layout(self.previous_point_button, self.point_combo, self.next_point_button))

    def create_navigation_controls(self, label):
        prev_button = QPushButton(f"Previous {label}")
        next_button = QPushButton(f"Next {label}")
        combo = QComboBox()
        prev_button.clicked.connect(lambda: self.navigate(combo, -1))
        next_button.clicked.connect(lambda: self.navigate(combo, 1))
        return prev_button, combo, next_button

    def create_horizontal_layout(self, *widgets):
        layout = QHBoxLayout()
        for widget in widgets:
            layout.addWidget(widget)
        return layout

    def navigate(self, combo, direction):
        new_index = combo.currentIndex() + direction
        if 0 <= new_index < combo.count():
            combo.setCurrentIndex(new_index)

    def setup_load_button(self):
        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self.load_images)
        self.layout.addWidget(self.load_button)

    def get_files(self):
        if self.img_dir_edit.text() and self.mask_dir_edit.text():
            self.load_file_map(self.img_dir_edit.text(), self.mask_dir_edit.text())
            self.populate_combos()
        else:
            print("Please select both image and mask directories.")

    # def previous_time(self):
    #     # select the previous element in the time combo box
    #     current_index = self.time_combo.currentIndex()
    #     if current_index > 0:
    #         self.time_combo.setCurrentIndex(current_index - 1)
    
    # def next_time(self):
    #     # select the next element in the time combo box
    #     current_index = self.time_combo.currentIndex()
    #     if current_index < self.time_combo.count() - 1:
    #         self.time_combo.setCurrentIndex(current_index + 1)
    
    # def previous_point(self):
    #     # select the previous element in the point combo box
    #     current_index = self.point_combo.currentIndex()
    #     if current_index > 0:
    #         self.point_combo.setCurrentIndex(current_index - 1)
    
    # def next_point(self):
    #     # select the next element in the point combo box
    #     current_index = self.point_combo.currentIndex()
    #     if current_index < self.point_combo.count() - 1:
    #         self.point_combo.setCurrentIndex(current_index + 1)

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
        # add a dimension to the mask data if img_data has 4 dimensions
        if img_data.ndim == 4:
            mask_data = np.expand_dims(mask_data, axis=0)

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
