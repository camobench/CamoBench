import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForImageTextRetrieval
)
import pyiqa
import t2v_metrics
import csv
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(DEVICE)

def get_clip_score(image_path: str, text: str) -> float:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        cos_sim = torch.sum(image_embeds * text_embeds, dim=-1).item()
        clip_score = max(100.0 * cos_sim, 0.0)
    return clip_score

# BLIP
blip_model_id = "Salesforce/blip-itm-base-coco"
blip_processor = BlipProcessor.from_pretrained(blip_model_id)
blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_id).to(DEVICE)

def get_blip_score(image_path: str, text: str) -> float:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = blip_model(**inputs)
        itm_scores = outputs.itm_score 
        score = torch.nn.functional.softmax(itm_scores, dim=1)[:, 1].item()
    return score

# VQAScore
vqa_metric = t2v_metrics.VQAScore(model='clip-flant5-xl', device=DEVICE)

def get_vqascore(image_path: str, text: str) -> float:
    score = vqa_metric(images=[image_path], texts=[text]).item()
    return score 

# NIQE MUSIQ LAION-Aes
niqe_metric = pyiqa.create_metric('niqe', device=DEVICE, as_loss=False)
musiq_metric = pyiqa.create_metric('musiq', device=DEVICE, as_loss=False)
laion_metric = pyiqa.create_metric('laion_aes', device=DEVICE, as_loss=False)

def get_niqe_score(image_path: str) -> float:
    return niqe_metric(image_path).item()

def get_musiq_score(image_path: str) -> float:
    return musiq_metric(image_path).item()

def get_laion_aesthetics_score(image_path: str) -> float:
    return laion_metric(image_path).item()

def run_evaluation_pipeline(csv_file="iaa_dataset.csv", image_dir="images", output_jsonl="evaluation_results.jsonl"):
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(csv_file, mode="r", encoding="utf-8-sig") as f_csv, \
         open(output_jsonl, mode="w", encoding="utf-8") as f_jsonl:
        
        reader = csv.DictReader(f_csv)
        
        for row_idx, row in enumerate(reader):
            img_id = row.get("id", "")
            bg_desc = row.get("Background Description", "")
            cam_desc = row.get("Camouflaged Object Description", "")
            
            img_path = os.path.join(image_dir, f"iaa_{img_id}.png")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found {img_path}, skipping this row.")
                continue
                
            print(f"Processing ID: {img_id} ({row_idx + 1})")
            
            try:
                # 1-3: Scene Consistency (Background)
                scene_clip = get_clip_score(img_path, bg_desc)
                scene_blip = get_blip_score(img_path, bg_desc)
                scene_vqa = get_vqascore(img_path, bg_desc)
                
                # 4-6: Illusion Shape Consistency (Camouflaged Object)
                shape_clip = get_clip_score(img_path, cam_desc)
                shape_blip = get_blip_score(img_path, cam_desc)
                shape_vqa_original = get_vqascore(img_path, cam_desc)
                
                # 7-9: Overall Visual Quality
                overall_niqe = get_niqe_score(img_path)
                overall_musiq = get_musiq_score(img_path)
                overall_laion = get_laion_aesthetics_score(img_path)
                
                # 10: Illusion Fidelity - DeltaVQAScore
                temp_img_path = os.path.join(temp_dir, f"temp_{img_id}_64.png")
                with Image.open(img_path) as img:
                    img_64 = img.convert("RGB").resize((64, 64))
                    img_64.save(temp_img_path)
                
                shape_vqa_64 = get_vqascore(temp_img_path, cam_desc)
                delta_vqa = shape_vqa_64 - shape_vqa_original
                
                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
                # Build result dictionary
                result_dict = {
                    "id": str(img_id),
                    "Semantic Consistency – Scene Consistency - CLIP": str(scene_clip),
                    "Semantic Consistency – Scene Consistency - BLIP": str(scene_blip),
                    "Semantic Consistency – Scene Consistency - VQAScore": str(scene_vqa),
                    "Semantic Consistency – Illusion Shape Consistency - CLIP": str(shape_clip),
                    "Semantic Consistency – Illusion Shape Consistency - BLIP": str(shape_blip),
                    "Semantic Consistency – Illusion Shape Consistency - VQAScore": str(shape_vqa_original),
                    "Overall Visual Quality - NIQE": str(overall_niqe),
                    "Overall Visual Quality - MUSIQ": str(overall_musiq),
                    "Overall Visual Quality - LAION-Aes": str(overall_laion),
                    "Illusion Fidelity - DeltaVQAScore": str(delta_vqa)
                }
                
                # Write to jsonl
                f_jsonl.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"Error processing ID {img_id}: {e}")

    print(f"All processing complete! Results saved to: {output_jsonl}")

if __name__ == "__main__":
    run_evaluation_pipeline(
        csv_file="./data/iaa/iaa_dataset.csv", 
        image_dir="./data/iaa/images", 
        output_jsonl="./data/existing_automation_metrics/score.jsonl"
    )