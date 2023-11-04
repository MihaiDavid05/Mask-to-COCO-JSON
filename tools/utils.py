import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from skimage import io


def create_image_info(
        image_id,
        file_name,
        image_size,
        date_captured=datetime.datetime.utcnow().isoformat(" "),
        license_id=1,
        coco_url="",
        flickr_url="",
):
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
        annotation_id,
        image_id,
        category_info,
        binary_mask,
        filter_area=4
):
    annotation_infos = []

    # Pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = (
            np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0) * 255
    ).astype("uint8")

    # Find only most external contours in padded image
    contours, _ = cv2.findContours(
        padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    for i, contour in enumerate(contours):

        # Filter unenclosed objects
        if len(contour) < 3:
            continue

        # Subtract 1 because of previous padding
        contour = np.subtract(contour, 1)

        # Make sure that the contour does not include values < 0 (possible -1, after subtraction above)
        contour[contour < 0] = 0
        segmentation = contour.ravel().tolist()

        # Find original mask of instance defined by contours
        img = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.uint8)
        cv2.fillPoly(img, pts=[contour], color=255)
        instance_binary_mask = np.where(img == 255, 1, 0)
        instance_binary_mask = instance_binary_mask & binary_mask

        # It means the contour belonged to a hole
        if instance_binary_mask.sum() < len(contour) / 2:
            continue

        # Create uncompressed RLE format from binary mask
        instance_binary_mask_fr = np.asfortranarray(instance_binary_mask)
        rle = binary_mask_to_rle(instance_binary_mask_fr)

        # Create compressed RLE format from uncompressed RLE and find area and bbox
        compressed_rle = mask.frPyObjects(rle, rle.get("size")[0], rle.get("size")[1])
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
            "bbox": bounding_box_rle,
            "area": seg_area_rle,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        # If only 1 contour in image, we can define it as a polygon (we are sure no holes are present)
        # Otherwise we will proceed to encode mask in RLE format (thst takes holes into consideration)
        if len(contours) <= 1:
            annotation_info["segmentation"] = [segmentation]
        else:
            annotation_info["segmentation"] = rle

        annotation_id += 1

        annotation_infos.append(annotation_info)

    return annotation_infos, annotation_id


def binary_mask_to_rle(binary_mask):
    # Define COCO format
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


def check_export_results(img_path, output_ann_path, output_dir, cats):
    # Create directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations in coco format
    coco = COCO(output_ann_path)
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
