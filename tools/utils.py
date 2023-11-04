import datetime
import os

import cv2
from typing import Tuple, Dict, Union, List, Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from skimage import io


def create_image_info(
        image_id: int,
        file_name: str,
        image_size: Tuple[int, int],
        date_captured: str = datetime.datetime.utcnow().isoformat(" "),
        license_id: int = 1,
        coco_url: str = "",
        flickr_url: str = "",
) -> Dict[str, Union[int, str]]:
    """
    Build a dictionary with information for one image

    """
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }

    return image_info


def create_annotation_infos(
        annotation_id: int,
        image_id: int,
        category_info: Dict[str, int],
        binary_mask: NDArray[np.uint8],
        filter_area: int = 4
) -> Tuple[Any, int]:
    """
     Builds list of dictionaries for 'annotations' field in COCO format

    Args:
        annotation_id: current annotation id (among all instances)
        image_id: id of the image
        category_info: dictionary with category information
        binary_mask: mask corresponding to instances from a specific class in an image
        filter_area: area under which objects are discarded (considered to be too small)

    Returns: tuple of annotations from a specific class from an image amd the annotations counter

    """
    annotation_infos = []
    as_polygon = True
    # Pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = (
            np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0) * 255
    ).astype("uint8")

    # Find all contours in padded image
    contours, hierarchy = cv2.findContours(
        padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    # Hierarchy for a contour(c) is represented by: [Next_c_idx, Previous_c_idx, First_Child_c_idx, Parent_c_idx]
    if hierarchy is not None:
        # Search for parent contours where are more than 1 contour and set polygon annotation as false
        if len(hierarchy[0]) > 1 and any([c[3] != -1 for c in hierarchy[0]]):
            as_polygon = False

    for i, contour in enumerate(contours):

        # Filter unenclosed objects
        if len(contour) < 3:
            continue

        # Subtract 1 because of previous padding
        contour = np.subtract(contour, 1)

        # Make sure that the contour does not include values < 0 (possible -1, after find contours + subtraction above)
        contour[contour < 0] = 0

        # Find original mask of instance defined by contours
        img = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.uint8)
        cv2.fillPoly(img, pts=[contour], color=255)
        instance_binary_mask = np.where(img == 255, 1, 0)
        instance_binary_mask = instance_binary_mask & binary_mask

        # Create uncompressed RLE format from binary mask
        instance_binary_mask_fr = np.asfortranarray(instance_binary_mask)
        rle = binary_mask_to_unc_rle(instance_binary_mask_fr)

        # Create compressed RLE format from uncompressed RLE and find area and bbox
        compressed_rle = unc_rle_to_comp_rle(rle)
        seg_area_rle = int(mask.area(compressed_rle))
        bounding_box_rle = mask.toBbox(compressed_rle).tolist()

        # Filter small objects, or points
        bbox_area = bounding_box_rle[2] * bounding_box_rle[3]
        if bbox_area < filter_area:
            continue

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": category_info["is_crowd"],
            # we keep area and bbox in rle format as they are more accurate than polygon
            "bbox": bounding_box_rle,
            "area": seg_area_rle,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        # Store annotation in corresponding format
        if as_polygon:
            segmentation = contour.ravel().tolist()
            annotation_info["segmentation"] = [segmentation]
        else:
            annotation_info["segmentation"] = rle

        # Increase instance id
        annotation_id += 1

        annotation_infos.append(annotation_info)

    return annotation_infos, annotation_id


def binary_mask_to_unc_rle(binary_mask: NDArray[np.intc]) -> Dict[str, List[int]]:
    """
    Convert a binary mask into uncompressed RLE format
    Args:
        binary_mask: mask corresponding to instances from a specific class in an image

    Returns: mask in RLE uncompressed format (a dictionary with 2 keys, 'counts' and 'size')

    """
    # Define COCO format and initialize
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    last_elem = 0
    running_length = 0

    # Encode to RLE uncompressed format
    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def unc_rle_to_comp_rle(unc_rle: Dict[str, List[int]]) -> Dict[str, Union[List[int], bytes]]:
    """
    Convert RLE uncompressed format in RLE compressed format
    Args:
        unc_rle: mask in RLE uncompressed format
    """
    return mask.frPyObjects(unc_rle, unc_rle.get("size")[0], unc_rle.get("size")[1])


def check_export_results(img_path: str, json_file_path: str, output_dir: str, cats: List[int]):
    """
    Plots images and masks from the exported-to-COCO annotation files
    Args:
        img_path: path to images directory
        json_file_path: path to COCO JSON file
        output_dir: path to directory where to save images with masks
        cats: images containing categories/class ids from thi list will be saved for visualization

    """
    # Create directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations in coco format
    coco = COCO(json_file_path)
    img_ids = coco.getImgIds(catIds=cats)

    # Display and save all images with the requested categories
    for i in range(len(img_ids)):
        img = coco.loadImgs(img_ids[i])[0]
        image = io.imread(img_path + img["file_name"])
        plt.axis("off")
        plt.imshow(image)
        ann_ids = coco.getAnnIds(imgIds=img["id"], catIds=cats, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)
        coco.showAnns(annotations)
        plt.savefig(os.path.join(output_dir, f"{img['file_name'].split('.')[0]}.png"))
        plt.close()


def generate_color(existing_colors: List[Tuple[int, int, int]]):
    """
    Generate a new random color that does not exist in the palette
    Args:
        existing_colors: List of existing colors in the palette

    Returns: a new color

    """
    found = False
    color = (0, 0, 0)
    while not found:
        # TODO: the colors can be improved
        color = np.random.choice(range(256), size=3)
        color = tuple([int(c) for c in color])
        if color not in existing_colors:
            found = True
    return color
