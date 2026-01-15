import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import time
from Datasets.datasets import VIFData
import torch.nn as nn
import transforms as T
import os
from models.model import Net


def main():
    infrared_dir = "/home/caizn/VIF/MSRS/test/ir"
    visible_dir = "/home/caizn/VIF/MSRS/test/vi"
    save_dir = "/home/caizn/LUT-Fuse/resultstest"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Net()
    # get_context = Generator_for_info()
    model.load_state_dict(torch.load("./checkpoint/MSRS.pth"))
    model = model.to(device)
    model.eval()

    data_transform = {
        "train": T.Compose([T.ToTensor()]),
        "val": T.Compose([T.ToTensor()])}

    val_dataset = SimpleDataSet(visible_path=visible_dir,
                                infrared_path=infrared_dir,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)

    infrared_files = sorted(os.listdir(infrared_dir))
    visible_files = sorted(os.listdir(visible_dir))

    assert len(infrared_files) == len(visible_files), "The number of images in the infrared and visible folders do not match!"
    times = []

    for step, data in enumerate(val_loader):
        I_A, I_B, task = data

        if torch.cuda.is_available():
            I_A = I_A.to("cuda")
            I_B = I_B.to("cuda")

        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(I_A, I_B)
        end_time = time.time()
        torch.cuda.synchronize()
        # end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        if not os.path.splitext(task[0])[1]:
            task_with_extension = task[0] + ".png"
        else:
            task_with_extension = task[0]
        save_path = os.path.join(save_dir, task_with_extension)
        fusion_result = outputs.squeeze(0).clamp(0, 1).cpu()
        fusion_result_image = ToPILImage()(fusion_result)
        fusion_result_image.save(save_path)


    warmup_skip = 25
    if len(times) > warmup_skip:
        times_after_warmup = times[warmup_skip:]
        avg_time = np.mean(times_after_warmup)
        std_time = np.std(times_after_warmup)
        print(f"Processing completed! after skipping the first {warmup_skip} images，avg_time: {avg_time:.4f} seconds，std_time: {std_time:.4f} seconds")
    else:
        print(f"Not enough images to skip the first {warmup_skip} ！Total images: {len(times)}")

if __name__ == "__main__":
    main()

