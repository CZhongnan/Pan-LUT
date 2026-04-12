import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

# 根据你的项目结构修改下面的导入路径
from models.model import *


def parse_args():
    parser = argparse.ArgumentParser(description="Image fusion demo")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["vif", "mif", "pan", "mef", "mff"],
        help="fusion task type",
    )
    parser.add_argument("--image_A", type=str, required=True, help="path of image A")
    parser.add_argument("--image_B", type=str, required=True, help="path of image B")
    parser.add_argument("--out", type=str, required=True, help="output image path")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu, default auto")
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensor_vif(image_a_path, image_b_path, device):
    to_tensor = ToTensor()

    image_a = Image.open(image_a_path).convert(mode="RGB")
    image_b = Image.open(image_b_path).convert(mode="L")

    i_a = to_tensor(image_a).unsqueeze(0).to(device)
    i_b = to_tensor(image_b).unsqueeze(0).to(device)
    return i_a, i_b


def load_tensor_mef_mff(image_a_path, image_b_path, device):
    to_tensor = ToTensor()

    image_a = Image.open(image_a_path).convert(mode="RGB")
    image_a = image_a.convert("YCbCr")

    image_b = Image.open(image_b_path).convert(mode="RGB")
    image_b = image_b.convert("YCbCr")

    i_a = to_tensor(image_a).unsqueeze(0).to(device)
    i_b = to_tensor(image_b).unsqueeze(0).to(device)
    return i_a, i_b


def load_model(task, device):
    if task == "vif":
        model = Net_VIF()
        ckpt_path = "./checkpoint/vif.pth"
    elif task == "mef":
        model = Net_MEF()
        ckpt_path = "./checkpoint/mef.pth"
    elif task == "mff":
        model = Net_MEF()
        ckpt_path = "./checkpoint/mff.pth"
    elif task == "mif":
        ckpt_path = "./checkpoint/mif.pth"
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        raise RuntimeError(f"Found checkpoint but no model loading logic for task '{task}'.")
    elif task == "pan":
        ckpt_path = "./checkpoint/pan.pth"
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        raise RuntimeError(f"Found checkpoint but no model loading logic for task '{task}'.")
    else:
        raise ValueError(f"Unsupported task: {task}")

    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def infer_vif(model, i_a, i_b):
    with torch.no_grad():
        out = model(i_a, i_b)
    return out


def infer_mef_mff(model, i_a, i_b):
    i_ay = i_a[:, 0:1, :, :]
    i_acb = i_a[:, 1:2, :, :]
    i_acr = i_a[:, 2:3, :, :]

    i_by = i_b[:, 0:1, :, :]
    i_bcb = i_b[:, 1:2, :, :]
    i_bcr = i_b[:, 2:3, :, :]

    cr = torch.cat((i_acr, i_bcr), dim=1)
    cb = torch.cat((i_acb, i_bcb), dim=1)

    with torch.no_grad():
        if lut_module is not None:
            out = lut_module(i_ay, i_by, cb, cr)
        else:
            out = model(i_ay, i_by, cb, cr)
    return out


def tensor_to_pil_image(out_tensor):
    if isinstance(out_tensor, (list, tuple)):
        out_tensor = out_tensor[0]

    if not isinstance(out_tensor, torch.Tensor):
        raise TypeError("模型输出不是 torch.Tensor，无法保存图像。")

    out_tensor = out_tensor.detach().float().cpu()

    if out_tensor.dim() == 4:
        out_tensor = out_tensor[0]

    if out_tensor.dim() != 3:
        raise ValueError(f"不支持的输出维度: {tuple(out_tensor.shape)}")

    out_tensor = out_tensor.clamp(0.0, 1.0)

    channels = out_tensor.shape[0]
    if channels == 1:
        return to_pil_image(out_tensor)
    if channels == 3:
        return to_pil_image(out_tensor)

    raise ValueError(f"不支持的输出通道数: {channels}，期望 1 或 3 通道。")


def save_output(out_tensor, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_pil_image(out_tensor)
    image.save(str(out_path))


def main():
    args = parse_args()
    device = get_device(args.device)

    model = load_model(args.task, device)

    if args.task == "vif":
        i_a, i_b = load_tensor_vif(args.image_A, args.image_B, device)
        out = infer_vif(model, i_a, i_b)
    elif args.task in ["mef", "mff"]:
        i_a, i_b = load_tensor_mef_mff(args.image_A, args.image_B, device)
        out = infer_mef_mff(model, i_a, i_b)
    else:
        raise RuntimeError(f"Found checkpoint for task '{args.task}', but no inference logic is defined.")

    save_output(out, args.out)
    print(f"Done. Output saved to: {args.out}")


if __name__ == "__main__":
    main()
