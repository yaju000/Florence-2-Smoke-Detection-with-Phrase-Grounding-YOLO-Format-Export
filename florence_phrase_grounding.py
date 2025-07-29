import os
import cv2
import torch
import time
import numpy as np
import ultralytics
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors

ultralytics.checks()

model_id = "microsoft/Florence-2-large"

# Ensure the runtime is set to GPU in Colab.
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def inference(image, task_prompt, text_input=None):
    """
    Performs inference using the given image and task prompt.

    Args:
        image (PIL.Image or tensor): The input image for processing.
        task_prompt (str): The prompt specifying the task for the model.
        text_input (str, optional): Additional text input to refine the prompt.

    Returns:
        dict: The model's processed response after inference.
    """
    # Combine task prompt with additional text input if provided
    prompt = task_prompt if text_input is None else task_prompt + text_input

    # Generate input data for model processing from the given prompt and image
    inputs = processor(
        text=prompt,  # Text input for the model
        images=image,  # Image input for the model
        return_tensors="pt",  # Return PyTorch tensors
    ).to("cuda", torch.float16)  # Move inputs to GPU with float16 precision

    # Generate model predictions (token IDs)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),  # text input IDs to CUDA
        pixel_values=inputs["pixel_values"].cuda(),  # pixel values to CUDA
        max_new_tokens=1024,  # Set maximum number of tokens to generate
        early_stopping=False,  # Disable early stopping
        do_sample=False,  # Use deterministic inference
        num_beams=3,  # Set beam search width for better predictions
    )

    # Decode generated token IDs into text
    generated_text = processor.batch_decode(
        generated_ids,  # Generated token IDs
        skip_special_tokens=False,  # Retain special tokens in output
    )[0]  # Extract first result from batch

    # Post-process the generated text into a structured response
    parsed_answer = processor.post_process_generation(
        generated_text,  # Raw generated text
        task=task_prompt,  # Task type for post-processing
        image_size=(image.width, image.height),  # scaling output
    )

    return parsed_answer  # Return the final processed output

def read_image(filename=None):
    if filename is not None:
        image_name = filename
    else:
        image_name = "bus.jpg"  # or "zidane.jpg"

    assets = "https://github.com/ultralytics/notebooks/releases/download/v0.0.0"

    safe_download(f"{assets}/{image_name}")  # Download the image

    # Read the image using OpenCV and convert it into the PIL format
    return Image.fromarray(cv2.cvtColor(cv2.imread(f"/content/{image_name}"), cv2.COLOR_BGR2RGB))

def convert_to_yolo_format(W, H, x1, y1, x2, y2, label):
    """ Converts bounding box coordinates to YOLO format.
    Args:
        W : Width of the image.
        H : Height of the image.
        x1 : Top-left x-coordinate of the bounding box.
        y1 : Top-left y-coordinate of the bounding box.
        x2 : Bottom-right x-coordinate of the bounding box.
        y2 : Bottom-right y-coordinate of the bounding box.
        label : Class label for the bounding box.
    Returns:
        str: Formatted string in YOLO format.
    """
    # 計算中心點與寬高
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # 轉成相對座標（0~1）
    x_center_norm = x_center / W
    y_center_norm = y_center / H
    width_norm = width / W
    height_norm = height / H

    output_format = f"{label} {x_center_norm:.4f} {y_center_norm:.4f} {width_norm:.4f} {height_norm:.4f}"

    return output_format

def write_yolo_txt(filename, yolo_line, output_dir="labels"):
    """ Writes a line in YOLO format to a text file.
    Args:
        filename (str): Name of the file to write.
        yolo_line (str): The YOLO formatted string to write.
        output_dir (str): Directory where the file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, filename)

    with open(txt_path, "w") as f:
        f.write(yolo_line + "\n")

    print(f"Saved to: {txt_path}")

"""
# Result format 😀
{
    "<OD>": {
        "bboxes": [[x1, y1, x2, y2], ...],
        "labels": ["label1", "label2", ...]
    }
}
"""
# ------
# Main
# ------

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" #"<OD>" <OPEN_VOCABULARY_DETECTION>
video_path = "/workspace/video/video_output.mp4"
text_input = "smoke"  # Example text input for the task
# "Strictly detect smoke only if it clearly appears. Return nothing if no smoke is visible."
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# video_name = video_path.split("/")[-1].split(".")[0]
video_name = "test_video"
model_name = model_id.split("/")[-1]
output_video = f'/workspace/project/{model_name}_{video_name}_output.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 設定標註資料
output_images_path = "/workspace/project/images"
output_labels_path = "/workspace/project/labels"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # ------
    # Input
    # ------
    cv2.imwrite(f"{output_images_path}/{video_name}_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", frame)
    print(f"Saved to: {output_images_path}")

    start_time = time.time()
    image_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = inference(image_input, task_prompt, text_input)["<CAPTION_TO_PHRASE_GROUNDING>"]
    print("Spend time : ", time.time()-start_time)

    # Plot the results on an image
    annotator = Annotator(image_input)  # initialize Ultralytics annotator

    yolo_lines = []
    for idx, (box, label) in enumerate(zip(results["bboxes"], results["labels"])):
        annotator.box_label(box, label=label, color=colors(idx, True))
        label_index = results["labels"].index(label)
        output_format = convert_to_yolo_format(
                                    W=width, H=height,
                                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                                    label=label_index
                                )
        yolo_lines.append(output_format)
        
    write_yolo_txt(f"{video_name}_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.txt", 
                   "\n".join(yolo_lines), 
                   output_labels_path)    
    
    output = Image.fromarray(annotator.result())  # display the output
    
    output_frame = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    out.write(output_frame)  # Write the output frame to the video file

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
print("輸出完成：", output_video)
cv2.destroyAllWindows()
