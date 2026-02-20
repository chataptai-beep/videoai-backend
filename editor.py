from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ImageClip
import os

def assemble_video(avatar_path, audio_path, output_path, aspect_ratio="9:16"):
    # Load audio to get duration
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    # Create video from image if it's just a static avatar for now
    clip = ImageClip(avatar_path).set_duration(duration)
    
    # Set audio
    clip = clip.set_audio(audio)

    # Handle aspect ratio (9:16)
    if aspect_ratio == "9:16":
        # Target: 1080x1920
        w, h = clip.size
        target_ratio = 9/16
        current_ratio = w/h
        
        if current_ratio > target_ratio:
            # Too wide, crop sides
            new_w = h * target_ratio
            clip = clip.crop(x_center=w/2, width=new_w)
        else:
            # Too tall, crop top/bottom
            new_h = w / target_ratio
            clip = clip.crop(y_center=h/2, height=new_h)
        
        clip = clip.resize(height=1920)

    # Burn in captions (Simplified for MVP)
    # Note: TextClip might require ImageMagick installed on the system
    try:
        txt_clip = TextClip("Videeo.ai MVP", fontsize=70, color='yellow', font='Arial-Bold')
        txt_clip = txt_clip.set_pos('bottom').set_duration(duration)
        video = CompositeVideoClip([clip, txt_clip])
    except Exception as e:
        print(f"TextClip failed (check ImageMagick): {e}")
        video = clip

    video.write_videofile(output_path, fps=24, codec="libx264")
    return output_path
