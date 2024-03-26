import numpy as np

import app.optical_flow as oflow

## Test de get_crop

mag_means = np.zeros((100))
mag_means[30:60] = 12
mag_means[15:18] = 10
mag_means[5] = 6
mag_means[80:85] = 10
mag_means[92] = 6

for patience in [0, 1, 5]:
    crop = oflow.get_crop(mag_means, threshold=5, padding_in=0, padding_out=0, patience=patience)
    print(f'Cropping found for patience={patience} and threshold=5: {crop}')
crop_other = oflow.get_crop(mag_means, threshold=11, padding_in=0, padding_out=0, patience=0)
print(f'Cropping found for patience=0 and threshold=11: {crop_other}')
