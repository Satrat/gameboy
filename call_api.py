import replicate
import os
from PIL import Image
import requests
from io import BytesIO

def run_inference(
    image_path, 
    prompt="high quality colorized photograph, natural colors, detailed",
    negative_prompt="blurry, low quality",
    num_inference_steps=20,
    cfg_scale=7.5,
):
    with open(image_path, "rb") as f:
        input_image = f

        deployment = replicate.deployments.get("satrat/gameboy-upsample")
        
        prediction = deployment.predictions.create(
            input={
                "image": input_image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": 42 # for reproducability, usually set to None
            }
        )
    
        prediction.wait()
        return prediction.output

def download_result(output_url, save_path="output.png"):
    response = requests.get(output_url)
    img = Image.open(BytesIO(response.content))
    img.save(save_path)
    print(f"Image saved to {save_path}")
    return img

if __name__ == "__main__":
    result = run_inference(image_path="test_png/test_sara1.png")
    print(f"Generated image URL: {result}")
    
    if result:
        download_result(result, "generated_output.png")
