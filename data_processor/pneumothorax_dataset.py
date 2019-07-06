from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
import pydicom


class PneumothoraxDataset(Dataset):
    def __init__(self, dcm_images_dir, mask_csv_path):
        self.dcm_images_dir = dcm_images_dir
        self.mask_csv_path = mask_csv_path

        self.mask_df = pd.read_csv(self.mask_csv_path)
        self.masks = {row["ImageId"]: row["EncodedPixels"] for row in self.mask_df.to_dict('records')}

        self.dcm_images_names = os.listdir(self.dcm_images_dir)

    def __len__(self):
        return len(self.dcm_images_names)

    def __getitem__(self, idx):

        assert idx < len(self), f"Dataset does not contain {idx} index!!"

        img_name = self.dcm_images_names[idx]
        dcm_img_path = os.path.join(self.dcm_images_dir, img_name)

        data = pydicom.dcmread(dcm_img_path)

        shape_1 = data.Rows
        shape_2 = data.Columns

        result = {"img": data.pixel_array, "patient_id": data.PatientID, "age": data.PatientAge, "sex": data.PatientSex,
                  "modality": data.Modality, "body_part": data.BodyPartExamined, "position": data.ViewPosition,
                  "mask": self.rle2mask(self.masks[img_name], shape_1, shape_2)}

        return result

    @staticmethod
    def rle2mask(rle, width, height):
        if rle == '-1':
            return np.zeros((width, height))
        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 255
            current_position += lengths[index]

        return mask.reshape(width, height)






