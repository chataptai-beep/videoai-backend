import os
import requests
from openai import OpenAI
from elevenlabs import generate, save, set_api_key
import replicate
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

def generate_script(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional marketing script writer. Write a 60-second script based on the prompt. Return JSON with 'script_text' and 'image_prompt' for the avatar."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    import json
    return json.loads(response.choices[0].message.content)

def generate_voice(text, output_path):
    audio = generate(
        text=text,
        voice="Bella",
        model="eleven_monolingual_v1"
    )
    save(audio, output_path)
    return output_path

def generate_avatar(image_prompt, output_path):
    output = replicate.run(
        "black-forest-labs/flux-pro",
        input={"prompt": image_prompt}
    )
    # Output is a list of URLs
    img_url = output[0]
    response = requests.get(img_url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

import time

def generate_lipsync(audio_url, video_url):
    api_key = os.getenv("SYNCLABS_API_KEY")
    if not api_key:
        print("SYNCLABS_API_KEY missing, skipping real lip-sync...")
        return None

    print(f"Calling SyncLabs with {audio_url} and {video_url}...")
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "audioUrl": audio_url,
        "videoUrl": video_url,
        "synergize": True,
        "model": "lipsync-2"
    }
    
    try:
        response = requests.post("https://api.synclabs.so/v2/generate", json=payload, headers=headers)
        response.raise_for_status()
        job = response.json()
        return job["id"] # Return the job ID for the frontend to poll
            
    except Exception as e:
        print(f"SyncLabs request failed: {e}")
        return None

def get_lipsync_status(job_id):
    api_key = os.getenv("SYNCLABS_API_KEY")
    headers = {"x-api-key": api_key}
    
    try:
        status_res = requests.get(f"https://api.synclabs.so/v2/jobs/{job_id}", headers=headers)
        status_res.raise_for_status()
        return status_res.json()
    except Exception as e:
        print(f"SyncLabs status check failed: {e}")
        return {"status": "FAILED", "error": str(e)}

def get_broll(keywords):
    # Placeholder for stock video API (e.g. Pexels)
    # Returns a list of stock video URLs based on keywords
    print(f"Searching B-roll for: {keywords}")
    return [
        "https://cdn.pixabay.com/vimeo/328941242/coffee-23067.mp4?width=1280&hash=...", # Example
    ]
