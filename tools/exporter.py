import glob
import json
import os
from abc import abstractmethod

import cv2
import numpy as np
from tqdm import tqdm

from tools.utils import create_image_info, create_annotation_infos


class BaseExporter:

    def __init__(self, img_path, ann_path, cat_file_path, output_ann_path, split, mask_channel, ext_ann=".png",
                 ext_img=".jpg", palette=None, supercategory="common-object"):
        self.img_path = img_path
        self.ann_path = ann_path
        self.cat_path = cat_file_path
        self.split = split
        self.channel = mask_channel
        self.ext_ann = ext_ann
        self.ext_img = ext_img
        self.palette = palette
        self.supercategory = supercategory
        self.categories = None

        # Create annotations directory if it does not exists
        self.output_ann_path = output_ann_path
        self.output_ann_path = os.path.join(self.output_ann_path, "annotations")
        os.makedirs(self.output_ann_path, exist_ok=True)

        if mask_channel == -1 and palette is None:
            raise ValueError("if mask_channel is -1 you need to provide palette as a list of RGB tuples"
                             "where the index of the color tuple in the list, corresponds with the class id")

        # Set licenses and info in coco output
        self.coco_output = {
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                }
            ],
            "info": {"description": "dataset exported in COCO Format"}
        }

    @abstractmethod
    def _get_classes_names_ids(self):
        pass

    def export(self, filter_area=4):
        # Fill categories
        self.categories = self._build_categories()
        self.coco_output["categories"] = self.categories

        # Fill images and annotations
        images, annotations = self._build_images_annotations(filter_area)
        self.coco_output["images"] = images
        self.coco_output["annotations"] = annotations

        return self.coco_output

    def save(self):
        self.output_ann_path = os.path.join(self.output_ann_path, "instances_{}2017.json".format(
            "val" if self.split == "test" else self.split)
                                            )
        with open(self.output_ann_path, "w") as output_json_file:
            json.dump(self.coco_output, output_json_file)

    def _build_palette(self, class_ids):
        # Set color palette (get original from mmdet, if available)
        if self.palette is None:
            try:
                import mmdet

                self.palette = mmdet.datasets.coco.CocoDataset.PALETTE
                nr_coco_classes = len(self.palette)
                nr_curr_classes = len(class_ids)
                if nr_curr_classes > nr_coco_classes:
                    for i in range(nr_coco_classes, nr_curr_classes + 1):
                        self.palette.append(self.palette[i % len(self.palette)])
                elif nr_coco_classes > nr_curr_classes:
                    self.palette = self.palette[:nr_curr_classes]
                print("Note: Took color palette from mmdet and build it circularly if more than 80 classes")
            except ModuleNotFoundError:
                # Generate list of new random colors
                colors = []
                # TODO: the colors can be improved
                while len(colors) < len(class_ids):
                    color = np.random.choice(range(256), size=3)
                    color = tuple([int(c) for c in color])
                    if color not in colors:
                        colors.append(color)
                self.palette = colors
                print("Note: Random color palette was generated")

    def _build_categories(self):
        categories = []
        class_ids, class_names = self._get_classes_names_ids()

        self._build_palette(class_ids)

        for i, cls_id in enumerate(class_ids):
            data = {
                "id": int(cls_id),
                "name": str(class_names[i]),
                "supercategory": self.supercategory,
                "color": list(self.palette[i])
            }
            categories.append(data)

        return categories

    def _build_images_annotations(self, filter_area=4):
        # Initialize
        images = []
        annotations = []
        image_id = 1
        segmentation_id = 1

        # Find all images and masks
        image_files = glob.glob(self.img_path + f"*{self.ext_img}")
        label_files = glob.glob(self.ann_path + f"*{self.ext_ann}")
        label_base_files = [os.path.basename(filename) for filename in label_files]

        # Go through each image
        for image_file in tqdm(image_files):
            image = cv2.imread(image_file)

            # Skip the image without label file
            base_name = str(os.path.basename(image_file).split('.')[0]) + self.ext_ann
            if base_name not in label_base_files:
                continue

            label_file = os.path.join(self.ann_path, os.path.splitext(base_name)[0] + self.ext_ann)
            image_info = create_image_info(
                image_id, os.path.basename(image_file), image.shape
            )
            images.append(image_info)

            if self.channel == -1:
                # Mask resides in 3 channels (read as BGR !!!)
                orig_mask = cv2.imread(label_file)
            else:
                # Mask resides in 1 channel
                orig_mask = cv2.imread(label_file)[..., self.channel]

            # Go through each existing category
            for category_dict in self.categories:
                color = category_dict["color"]
                class_id = category_dict["id"]
                category_info = {
                    "id": class_id,
                    "is_crowd": 0,
                }  # does not support the crowded type

                if self.channel == -1:
                    # Match the whole mask (RGB) color
                    binary_mask = np.all(orig_mask == np.transpose(color, (2, 1, 0)), axis=-1).astype("uint8")
                else:
                    # Match the class id in a categorical one channel mask
                    binary_mask = np.array(orig_mask == int(class_id)).astype("uint8")

                # Create annotation info
                annotation_info, annotation_id = create_annotation_infos(
                    segmentation_id,
                    image_id,
                    category_info,
                    binary_mask,
                    filter_area=filter_area
                )
                # Update annotations and instance is
                annotations.extend(annotation_info)
                segmentation_id = annotation_id

            image_id = image_id + 1

        return images, annotations
