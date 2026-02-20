import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def upload_to_supabase(file_path, destination_path):
    bucket_name = "videos"
    try:
        with open(file_path, 'rb') as f:
            supabase.storage.from_(bucket_name).upload(
                path=destination_path,
                file=f,
                file_options={"content-type": "video/mp4"}
            )
        
        # Get public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(destination_path)
        return public_url
    except Exception as e:
        print(f"Supabase Upload failed: {e}")
        return f"https://mock-storage.com/{destination_path}"
