import glob
import json
import os
from abc import abstractmethod
from typing import Dict, Tuple, List, Optional, Any, Union
from numpy.typing import NDArray
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from tools.utils import create_image_info, create_annotation_infos, generate_color


class BaseExporter:

    def __init__(self, img_path: str, ann_path: str, cat_file_path: str, output_ann_path: str, split: str,
                 mask_channel: int, ext_ann: str = ".png", ext_img: str = ".jpg",
                 palette: Optional[List[Tuple[int, int, int]]] = None, supercategory: str = "common-object"):

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
            raise ValueError("if mask_channel is -1 you need to provide palette as a list of RGB tuples "
                             "where the index of the color tuple in the list corresponds with the class id")

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
    def _get_classes_names_ids(self) -> Tuple[NDArray[str], NDArray[str]]:
        """
        Build classes ids and names into 2 numpy arrays, from a local file/source.
        A class and a name from the same index correspond to each other.

        Returns: class_ids, class_names

        """
        pass

    def export(self, filter_area: int = 4, polygon_only: bool = False) -> Dict[str, Union[List, Dict[str, Any]]]:
        """
        Exports annotations to COCO JSON format
        Args:
            filter_area: area under which objects are discarded (considered to be too small)
            polygon_only: whether to export to polygon format only

        Returns: Dictionary formatted in COCO style

        """
        # Fill categories
        self.categories = self._build_categories()
        self.coco_output["categories"] = self.categories

        # Fill images and annotations
        images, annotations = self._build_images_annotations(filter_area, polygon_only)
        self.coco_output["images"] = images
        self.coco_output["annotations"] = annotations

        return self.coco_output

    def save(self):
        """
        Saves COCO-like JSON formatted annotations to file

        """
        self.output_ann_path = os.path.join(self.output_ann_path, "instances_{}2017.json".format(
            "val" if self.split == "test" else self.split)
                                            )
        with open(self.output_ann_path, "w") as output_json_file:
            json.dump(self.coco_output, output_json_file)

    def _build_palette(self, class_ids: NDArray[str]):
        """
        Sets color palette for classes, if not already set.
        If mmdet is installed in your environment it gets the palette defined in that project for COCO (80 classes).
        It adds or select a specific number of random colors, depending on the number of classes in your dataset.
        If mmdet is not installed the entire palette woll be build from random distinctive colors.
        Args:
            class_ids: numpy array with classes ids

        """
        # Set color palette (get original from mmdet, if available)
        if self.palette is None:
            try:
                from mmdet import datasets

                self.palette = datasets.coco.CocoDataset.PALETTE
                nr_coco_classes = len(self.palette)
                nr_curr_classes = len(class_ids)
                if nr_curr_classes > nr_coco_classes:
                    for i in range(nr_coco_classes, nr_curr_classes):
                        color = generate_color(self.palette)
                        self.palette.append(color)
                    print("Note: Took color palette from mmdet (80 classes) and built upon it")

                elif nr_coco_classes > nr_curr_classes:
                    self.palette = self.palette[:nr_curr_classes]
                    print("Note: Took color palette from mmdet (80 classes) and selected from it")

            except ModuleNotFoundError:
                # Generate list of new random colors
                colors = []
                while len(colors) < len(class_ids):
                    color = generate_color(colors)
                    colors.append(color)
                self.palette = colors
                print("Note: Random color palette was generated")

    def _build_categories(self) -> List[Dict[str, Union[int, str, List]]]:
        """
        Builds content for 'categories' key in the final COCO formatted JSON.

        Returns: List with dictionaries for categories available in your dataset

        """
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

    def _build_images_annotations(self, filter_area: int = 4, polygon_only: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build 'images' and 'annotations' dictionary entries in the COCO JSON format
        Args:
            filter_area: area under which objects are discarded (considered to be too small)
            polygon_only: whether to export to polygon format only

        Returns: 2 lists of dictionaries, mainly: images, annotations

        """
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
                print(f"File {base_name} does not have a corresponding label")
                continue

            label_file = os.path.join(self.ann_path, os.path.splitext(base_name)[0] + self.ext_ann)
            image_info = create_image_info(
                image_id, os.path.basename(image_file), image.shape
            )
            images.append(image_info)

            if self.channel == -1:
                # Mask resides in 3 channels (reads as BGR)
                orig_mask = cv2.imread(label_file)
                img = Image.fromarray(orig_mask)
                print(img.mode)
            else:
                # Mask resides in 1 channel (imread reads in 3 channels-BGR, even if original was 'greyscale')
                orig_mask = cv2.imread(label_file)
                orig_mask = orig_mask[..., self.channel]

            # Go through each existing category
            for category_dict in self.categories:
                color = category_dict["color"]
                class_id = category_dict["id"]
                category_info = {
                    "id": class_id,
                    "is_crowd": 0,
                }  # does not support crowded type

                if self.channel == -1:
                    # Match the whole mask to color
                    binary_mask = np.all(orig_mask == np.array(color), axis=-1).astype("uint8")
                else:
                    # Match the class id in a categorical one channel mask
                    binary_mask = np.array(orig_mask == int(class_id)).astype("uint8")

                # Create annotation info
                annotation_info, annotation_id = create_annotation_infos(
                    segmentation_id,
                    image_id,
                    category_info,
                    binary_mask,
                    filter_area=filter_area,
                    polygon_only=polygon_only
                )
                # Update annotations and instance is
                annotations.extend(annotation_info)
                segmentation_id = annotation_id

            image_id = image_id + 1

        return images, annotations
