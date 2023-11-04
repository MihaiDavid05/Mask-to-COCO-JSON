import numpy as np
from tools.exporter import BaseExporter


class UECFoodPixCompleteExporter(BaseExporter):

    def __init__(self, img_path, ann_path, cat_file_path, output_ann_path, split, mask_channel, ext_ann=".png",
                 ext_img=".jpg", palette=None):
        super().__init__(img_path, ann_path, cat_file_path, output_ann_path, split, mask_channel, ext_ann,
                         ext_img, palette)

    def _get_classes_names_ids(self):
        # Read file
        with open(self.cat_path, "r") as f:
            data = f.readlines()

        # Lines with ids and names separated by tab
        data = np.array([line[:-1].rstrip().split("\t") for line in data])

        # Ignore header from text file (here, needed).
        # You may also skip some classes here, for example background if you don't need it in annotations.
        class_ids, class_names = data[1:, 0], data[1:, 1]

        return class_ids, class_names
