from typing import TYPE_CHECKING

from magicgui.widgets import Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QFileDialog, QWidget, QLineEdit, QPushButton, QComboBox, QRadioButton, QButtonGroup
from qtpy.QtGui import QColor
from towbintools.foundation import file_handling
import os
import numpy as np
import pandas as pd
import tifffile
import napari
from scipy.ndimage import find_objects
import xgboost as xgb
from skimage.measure import regionprops_table

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
        self.layout.setSpacing(10)
        # Decrease the margin to make the widget more compact
        self.layout.setContentsMargins(0, 0, 0, 0)
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

    def load_file_map(self, img_dir, mask_dir):
        filemap = file_handling.get_dir_filemap(img_dir)
        filemap = add_dir_to_experiment_filemap(filemap, mask_dir, "MaskPath")
        # Remove all rows with NaN values
        filemap = filemap.dropna()
        # Remove all rows with empty strings
        filemap = filemap[(filemap != "").all(axis=1)]
        self.filemap = filemap

    def populate_combos(self):
        self.time_combo.clear()
        self.time_combo.addItems(self.filemap["Time"].unique().astype(str))
        self.point_combo.clear()
        self.point_combo.addItems(self.filemap["Point"].unique().astype(str))

    def load_images(self):
        time = int(self.time_combo.currentText())
        point = int(self.point_combo.currentText())

        row = self.filemap[(self.filemap["Time"] == time) & (self.filemap["Point"] == point)]
        img_path = row["ImagePath"].values[0]
        mask_path = row["MaskPath"].values[0]

        # Read images using tifffile
        img_data = tifffile.imread(img_path)
        mask_data = tifffile.imread(mask_path)

        # If the image is 4D, swap the first and second axes
        if img_data.ndim == 4:
            img_data = np.swapaxes(img_data, 0, 1)

        # Add images to Napari viewer
        self.viewer.add_image(img_data, name=os.path.basename(img_path))
        self.viewer.add_labels(mask_data, name=os.path.basename(mask_path))

        # Set the label layer opacity to 0.5
        self.viewer.layers[-1].opacity = 0.5
        # Auto adjust the contrast limits of the image layer
        self.viewer.layers[-2].contrast_limits = (img_data.min(), img_data.max())

class AnnotationTool(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.selected_class = 'epidermis'  # Default class
        self.class_colors = {
            'epidermis': (1, 0, 0, 1),  # Red
            'intestine': (0, 0, 1, 1),  # Blue
            'other': (1, 1, 0, 1),  # Yellow
            'error': (1, 0.5, 0.5, 1)  # Coral Pink
        }
        self.class_values = {
            'epidermis': 1,
            'intestine': 2,
            'other': 3,
            'error': 4
        }

        # Map colors to class values using tuples as keys
        self.color_to_class = {self.class_colors[cls]: self.class_values[cls] for cls in self.class_colors}

        self.setup_ui()

    def create_dir_selector(self, button_label):
        dir_edit = QLineEdit()
        dir_button = QPushButton(button_label)
        dir_button.clicked.connect(lambda: self.select_directory(dir_edit))
        self.layout.addWidget(dir_edit)
        self.layout.addWidget(dir_button)
        return dir_edit, dir_button

    def create_file_selector(self, button_label):
        file_edit = QLineEdit()
        file_button = QPushButton(button_label)
        file_button.clicked.connect(lambda: self.select_file(file_edit))
        self.layout.addWidget(file_edit)
        self.layout.addWidget(file_button)
        return file_edit, file_button

    def select_directory(self, dir_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            dir_edit.setText(dir_path)
            return dir_path
        else:
            print("Directory selection cancelled.")
            return ""

    def select_file(self, file_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            file_edit.setText(file_path)
            return file_path
        else:
            print("File selection cancelled.")
            return ""

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        # Decrease the margin to make the widget more compact
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.start_annotating_button = QPushButton("Start Annotating")
        self.start_annotating_button.clicked.connect(self.prepare_annotation_layer)
        self.layout.addWidget(self.start_annotating_button)

        # Setup radio buttons for class selection
        self.class_buttons = QButtonGroup(self)  # Using a button group to manage radio buttons
        class_layout = QHBoxLayout()
        for cls in ['epidermis', 'intestine', 'other', 'error']:
            btn = QRadioButton(cls)
            btn.toggled.connect(self.on_class_selected)
            self.class_buttons.addButton(btn)
            class_layout.addWidget(btn)
            if cls == 'epidermis':
                btn.setChecked(True)  # Set default selected class

        self.layout.addLayout(class_layout)

        # self.convert_labels_button = QPushButton("Convert Labels")
        # self.convert_labels_button.clicked.connect(self.convert_labels)
        # self.layout.addWidget(self.convert_labels_button)

        self.save_dir_edit, self.save_dir_button = self.create_dir_selector("Select Save Directory")

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_annotations)
        self.layout.addWidget(self.save_button)

        self.model_file_edit, self.model_file_button = self.create_file_selector("Select XGB Model")

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        self.setLayout(self.layout)

    def on_class_selected(self, checked):
        radio_button = self.sender()
        if checked:
            self.selected_class = radio_button.text()
            self.update_point_tool_color()


    def prepare_annotation_layer(self):
        # Check for the last label layer and its dimensions
        label_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]
        if label_layers:
            last_label_layer = label_layers[-1]
            z_dim = last_label_layer.data.shape[0] if last_label_layer.ndim == 3 else None
        else:
            z_dim = None
            print("No label layers found, defaulting to standard dimensions.")

        # Creates a new point layer or retrieves an existing one
        if 'Annotations' not in [layer.name for layer in self.viewer.layers]:
            initial_data = np.zeros((0, 3)) if z_dim else np.zeros((0, 2))  # Use ternary operator to set initial_data size
            self.points_layer = self.viewer.add_points(initial_data, name='Annotations',
                                                    face_color=np.array(self.class_colors[self.selected_class]),
                                                    ndim=3 if z_dim else 2)
            print(f"Annotation layer added with {'3D' if z_dim else '2D'} capabilities.")
        else:
            self.points_layer = self.viewer.layers['Annotations']
            print("Using existing annotation layer.")

        self.update_point_tool_color()

    def update_point_tool_color(self):
        if hasattr(self, 'points_layer'):
            # Deselect all points
            self.points_layer.selected_data = []
            # Set the current face color to the selected class color
            self.points_layer.current_face_color = self.class_colors[self.selected_class]
            print(f"Ready to add points with color {self.class_colors[self.selected_class]} for class {self.selected_class}.")

    def save_annotations(self):
        """Save the annotations as a CSV file."""
        if not hasattr(self, 'points_layer'):
            print("No annotations to save.")
            return

        save_dir = self.save_dir_edit.text()
        if not save_dir:
            print("Please select a save directory.")
            return

        # Get the name of the image layer
        save_name = self.viewer.layers[0].name
        # replace the file extension with .csv
        save_name = os.path.splitext(save_name)[0] + '.csv'
        save_path = os.path.join(save_dir, save_name)
        annotation_dataframe = pd.DataFrame(columns=['Plane', 'Label', 'Class'])

        label_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]
        if not label_layers:
            print("No label layer found.")
            return
        base_label_layer = label_layers[-1]
        label_data = base_label_layer.data

        for point, color in zip(self.points_layer.data, self.points_layer.face_color):
            # For each point, get the value of the label layer at that point
            # convert the point coordinates to integers
            point = tuple(int(p) for p in point)
            label_value = label_data[tuple(point)]
            if label_value == 0:
                continue
            # Map the color to a class value
            class_value = self.color_to_class[tuple(color)]
            # Convert annotation_dataframe to DataFrame and append the new row
            annotation_dataframe = pd.concat([annotation_dataframe, pd.DataFrame([[point[0], label_value, class_value]], columns=['Plane', 'Label', 'Class'])])

        annotation_dataframe.to_csv(save_path, index=False)

    def predict(self):
        # Load the XGB model
        model_path = self.model_file_edit.text()
        if not model_path:
            print("Please select a model file.")
            return
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)

        label_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]
        if not label_layers:
            print("No label layer found.")
            return
        base_label_layer = label_layers[-1]
        label_data = base_label_layer.data

        img_layer = self.viewer.layers[0]
        img_data = img_layer.data[0]

        print(img_data.shape)

        # If the image is 3D, iterate over each plane
        if img_data.ndim == 3:
            for i, plane_img in enumerate(img_data):
                # Get the label data for the current plane
                plane_labels = label_data[i].astype(np.uint8)
                feature_of_all_labels = regionprops_table(plane_labels, intensity_image= plane_img, properties=('area', 'area_convex', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'extent', 'feret_diameter_max', 'solidity', 'perimeter', 'intensity_max', 'intensity_mean', 'intensity_min', 'weighted_moments_hu'))
                mean_features_plane = []
                for key in feature_of_all_labels:
                    mean_features_plane.append(np.mean(feature_of_all_labels[key]))

                # Predict the class of each label
                for label in np.unique(plane_labels):
                    if label == 0:
                        continue
                    label_mask = plane_labels == label
                    features_of_label = regionprops_table(label_mask, intensity_image= plane_img, properties=('area', 'area_convex', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'extent', 'feret_diameter_max', 'solidity', 'perimeter', 'intensity_max', 'intensity_mean', 'intensity_min', 'weighted_moments_hu'))
                    feature = []
                    for key in features_of_label:
                        feature.extend(features_of_label[key])

                    # concatenate the features of the label with the mean features of all labels
                    feature_vector = np.concatenate((feature, mean_features_plane))
                    feature_vector = feature_vector.reshape(1, -1)
                    print(feature_vector.shape)
                    prediction = clf.predict(feature_vector)
                    print(f'Prediction for label {label}: {prediction}')



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
