from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
import pydicom
import PIL


class PneumothoraxDataset(Dataset):
    def __init__(self, dcm_images_dir, mask_csv_path, reshape=128):
        self.dcm_images_dir = dcm_images_dir
        self.mask_csv_path = mask_csv_path

        self.mask_df = pd.read_csv(self.mask_csv_path)
        self.masks = {row["ImageId"]: row[" EncodedPixels"] for row in self.mask_df.to_dict('records')}

        self.dcm_images_names = os.listdir(self.dcm_images_dir)

        self.reshape = reshape

    def __len__(self):
        return len(self.dcm_images_names)

    def __getitem__(self, idx):

        assert idx < len(self), f"Dataset does not contain {idx} index!!"

        img_name = self.dcm_images_names[idx]
        dcm_img_path = os.path.join(self.dcm_images_dir, img_name)

        data = pydicom.dcmread(dcm_img_path)

        shape_1 = data.Rows
        shape_2 = data.Columns
        img = PIL.Image.fromarray(data.pixel_array).resize((self.reshape, self.reshape))
        img = np.array(img)
        if img_name.replace(".dcm", "") in self.masks:
            mask = PIL.Image.fromarray(self.rle2mask(self.masks[img_name.replace(".dcm", "")], shape_1,
                                                     shape_2)).resize((self.reshape, self.reshape))
        else:
            mask = np.zeros((self.reshape, self.reshape))

        mask = np.array(mask)

        result = {"img": img, "patient_id": data.PatientID, "age": data.PatientAge, "sex": data.PatientSex,
                  "modality": data.Modality, "body_part": data.BodyPartExamined, "position": data.ViewPosition,
                  "mask": mask}

        return result

    @staticmethod
    def rle2mask(rle, width, height):
        if rle.strip() == '-1':
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

    @staticmethod
    def mask2rle(img, width, height):
        rle = []
        lastColor = 0
        currentPixel = 0
        runStart = -1
        runLength = 0

        for x in range(width):
            for y in range(height):
                currentColor = img[x][y]
                if currentColor != lastColor:
                    if currentColor == 255:
                        runStart = currentPixel
                        runLength = 1
                    else:
                        rle.append(str(runStart))
                        rle.append(str(runLength))
                        runStart = -1
                        runLength = 0
                        currentPixel = 0
                elif runStart > -1:
                    runLength += 1
                lastColor = currentColor
                currentPixel += 1

        return " ".join(rle)







