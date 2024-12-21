# 2024.10.16
创建代码，ram_group.py,可以批量输入数据，同时可以输出每个ram的json文件，方便查看每个ram的运行情况
```
import argparse
import os
import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything.segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram_plus
from ram import inference_ram
import torchvision.transforms as TS
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def process_images(image_dir, output_dir, config_file, ram_checkpoint, grounded_checkpoint, sam_checkpoint,
                   box_threshold, text_threshold, device):
    """
    Process a directory of images, run recognition models, and save outputs as JSON files.

    Args:
        image_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory to save the output JSON files.
        config_file (str): Path to the configuration file for the grounding model.
        ram_checkpoint (str): Path to the checkpoint file for the RAM model.
        grounded_checkpoint (str): Path to the checkpoint file for the grounding model.
        sam_checkpoint (str): Path to the checkpoint file for the SAM model.
        box_threshold (float): Threshold for filtering boxes.
        text_threshold (float): Threshold for filtering text.
        device (str): Device to use ('cpu' or 'cuda').
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    grounding_model = load_model(config_file, grounded_checkpoint, device=device)
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
    ram_model = ram_plus(pretrained=ram_checkpoint, image_size=384, vit='swin_l').to(device)
    ram_model.eval()

    # Process each image in the directory
    print(f"Processing images in directory: {image_dir}")
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_dir, filename)

            # Load and preprocess image
            image_pil, image = load_image(image_path)
            raw_image = image_pil.resize((384, 384))
            raw_image = transform(raw_image).unsqueeze(0).to(device)

            # Recognize tags using RAM model
            ram_results = inference_ram(raw_image, ram_model)
            tags = ram_results[0].replace(' |', ',')
            tags_chinese = ram_results[1].replace(' |', ',')

            # Get grounding model output
            boxes_filt, scores, pred_phrases = get_grounding_output(
                grounding_model, image, tags, box_threshold, text_threshold, device=device
            )

            # Save results to JSON
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_labels.json")
            json_data = {
                "filename": filename,
                "tags": tags,
                "tags_chinese": tags_chinese,
                "predictions": [
                    {
                        "phrase": phrase,
                        "score": float(score),
                        "box": box.numpy().tolist()
                    }
                    for phrase, score, box in zip(pred_phrases, scores, boxes_filt)
                ]
            }
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=4)

            print(f"Processed and saved: {filename}")

# Example usage
if __name__ == "__main__":
    process_images(
        image_dir="datasets/msrs",  # Path to input image directory
        output_dir="outputs/group",  # Path to output JSON directory
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",  # Path to GroundingDINO config
        ram_checkpoint="/home/lpw/fastssd/lpw/Grounded-Segment-Anything/pretrain/ram_plus_swin_large_14m.pth",  # Path to RAM model checkpoint
        grounded_checkpoint="/home/lpw/fastssd/lpw/Grounded-Segment-Anything/pretrain/groundingdino_swint_ogc.pth",  # Path to GroundingDINO checkpoint
        sam_checkpoint="/home/lpw/fastssd/lpw/Grounded-Segment-Anything/pretrain/sam_vit_h_4b8939.pth",  # Path to SAM model checkpoint
        box_threshold=0.25,
        text_threshold=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
```