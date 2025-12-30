import os
import numpy as np
import cv2

classes = ["AnnualCrop", "Forest", "Residential", "River", "Industrial"]
splits = ["train", "test"]

base_dir = "satellite_dataset"
img_size = 64

for split in splits:
    for cls in classes:
        path = os.path.join(base_dir, split, cls)
        os.makedirs(path, exist_ok=True)

        for i in range(100 if split == "train" else 30):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            if cls == "Forest":
                img[:] = (0, np.random.randint(100, 255), 0)
            elif cls == "River":
                img[:] = (np.random.randint(0, 50), np.random.randint(0, 50), 255)
            elif cls == "Residential":
                img[:] = np.random.randint(120, 200)
            elif cls == "Industrial":
                img[:] = (150, 150, 150)
            else:  # AnnualCrop
                img[:] = (0, 255, 255)

            cv2.imwrite(f"{path}/{cls}_{i}.jpg", img)

print("Dataset Generated Successfully!")
