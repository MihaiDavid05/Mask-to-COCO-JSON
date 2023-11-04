import argparse

from tools.datasets import UECFoodPixCompleteExporter
from tools.utils import check_export_results


def parse_args():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument(
        "img_path",
        help="path to directory with images",
        type=str,
        default="",
    )
    parser.add_argument(
        "ann_path",
        help="path to directory with masks",
        type=str,
        default="",
    )
    parser.add_argument(
        "cat_file_path",
        help="path to file with mapping between class indexes and names (if exists)",
        type=str,
        default="",
    )
    parser.add_argument(
        "output_ann_path",
        help="path to directory for storing annotations",
        type=str,
        default="",
    )
    parser.add_argument(
        "split",
        help="train or test ('test' will be converted into 'val' when saving the COCO JSON files format)",
        type=str,
        default="train",
    )
    parser.add_argument(
        "mask_channel",
        help=" Which channels does the mask resides on. ATTENTION: 0, 1 or 2 correspond to B, G or R channels."
             " Usually the mask is a an RGB but with the same 'greyscale' map with the class indexes repeated 3 times."
             " In this case the channel could be either 0, 1 or 2, it does not matter."
             " Otherwise, it is represented on a exact, single, channel (for example R in UECFOODPIXCOMPLETE). "
             " Value -1 means that the mask is on 3 channels (RGB), in this case the mask colors"
             " correspond with the color palette of the classes and you should create the palette from ids-to-colors"
             " mapping present somewhere in your files",
        choices=[-1, 0, 1, 2],
        type=int,
        default=0,
    )
    parser.add_argument(
        "--filter_area",
        help="area under which the objects are filtered out (for a VERY low resolution this should be a bit higher)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--masks_output_path",
        help="in case you want to check the resulting exported masks, set the path to output directory",
        type=str,
        default="output/",
    )
    parser.add_argument(
        "--cats",
        metavar='c',
        type=int,
        nargs='*',
        help="In case you already set the path to output directory for mask, "
             "you can constrain the resulting masks by providing a list of categories which appear (TOGETHER) in "
             "each images and you'll see these categories in each image. If the list is empty all categories are shown",
        default=[]
    )
    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    result_masks_output_dir = args.masks_output_path

    # NOTE: You should generate the palette HERE if mask_channel is -1, else palette will be automatically generated
    color_palette = None

    # Instantiate your exporter
    exporter = UECFoodPixCompleteExporter(img_path=args.img_path,
                                          ann_path=args.ann_path,
                                          cat_file_path=args.cat_file_path,
                                          output_ann_path=args.output_ann_path,
                                          split=args.split,
                                          mask_channel=args.mask_channel,
                                          palette=color_palette)
    # Export to coco format
    _ = exporter.export(args.filter_area)

    # Save to file
    exporter.save()

    # Check results for images that contain specific categories
    if result_masks_output_dir != "":
        check_export_results(exporter.img_path, exporter.output_ann_path,
                             output_dir=result_masks_output_dir, cats=args.cats)
