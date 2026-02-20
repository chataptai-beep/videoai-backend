"""
Image Generator using kie.ai API with Nano Banana model.
Generates reference character images for video generation.
"""

import asyncio
import json
import logging
from typing import Optional

import httpx

from config import get_settings, sanitize_header_token

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Generates reference images using kie.ai's Nano Banana model.
    Used for character consistency across video scenes.
    """
    
    def __init__(self):
        self.api_key = get_settings().kie_api_key
        self.base_url = "https://api.kie.ai/api/v1"  # Correct kie.ai API base
        self.poll_interval = get_settings().api_poll_interval_seconds
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "text, watermark, logo, blurry, distorted, multiple people",
        output_format: str = "png"
    ) -> str:
        """
        Generate an image using kie.ai's Nano Banana model.
        
        Args:
            prompt: Image generation prompt
            negative_prompt: What to avoid in the image
            output_format: Output format (png/jpg)
        
        Returns:
            URL to the generated image
        
        Raises:
            Exception: If generation fails
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Image prompt is required for reference image generation")
        logger.info(f"Generating image with prompt: {prompt[:50]}...")
        
        # Step 1: Create the generation task
        task_id = await self._create_task(prompt, negative_prompt, output_format)
        
        # Step 2: Poll for completion
        image_url = await self._poll_for_result(task_id)
        
        logger.info(f"Image generated: {image_url}")
        return image_url

    def _auth_header(self) -> str:
        """Bearer token for API; raises clear error if key is missing (avoids 'Illegal header value b\"Bearer \"')."""
        # Read from config at call time so restart-reloaded keys are used
        key = sanitize_header_token((get_settings().kie_api_key or ""))
        if not key:
            raise ValueError(
                "KIE_API_KEY is not set or invalid. Add KIE_API_KEY=your_key to the .env file in the project root and restart the server."
            )
        return "Bearer " + key

    async def _create_task(
        self,
        prompt: str,
        negative_prompt: str,
        output_format: str
    ) -> str:
        """Create an image generation task and return the task ID."""
        
        # Enhance the prompt for better results
        enhanced_prompt = (
            f"{prompt}. "
            "Hyper-realistic, 4K resolution, detailed textures, "
            "professional studio lighting, cinematic quality."
        )
        
        try:
            # Use longer timeouts to avoid 522 errors
            timeout = httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                # Based on n8n workflow, kie.ai uses this format:
                # {"model": "google/nano-banana", "input": {"prompt": "...", "output_format": "png"}}
                response = await client.post(
                    f"{self.base_url}/jobs/createTask",
                    headers={
                        "Authorization": self._auth_header(),
                        "Content-Type": "application/json",
                        "User-Agent": "Videeo-Pipeline/1.0"
                    },
                    json={
                        "model": "google/nano-banana",
                        "input": {
                            "prompt": enhanced_prompt,
                            "output_format": output_format
                        }
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                logger.debug(f"createTask response: {data}")
                
                # kie.ai returns taskId in data.taskId
                task_id = (
                    data.get("data", {}).get("taskId") or
                    data.get("taskId") or
                    data.get("task_id") or
                    data.get("id")
                )
                
                if not task_id:
                    raise Exception(f"No task ID in response: {data}")
                
                logger.info(f"Image task created: {task_id}")
                return task_id
                
        except httpx.HTTPStatusError as e:
            logger.error(f"kie.ai API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Image generation failed: {e.response.status_code} - {e.response.text}")
    
    async def _poll_for_result(
        self,
        task_id: str,
        max_attempts: int = 60  # Increased for longer generation times
    ) -> str:
        """Poll for task completion and return the image URL."""
        
        for attempt in range(max_attempts):
            try:
                # Use longer timeouts to avoid 522 errors
                timeout = httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=30.0)
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    # kie.ai uses recordInfo endpoint with query param
                    response = await client.get(
                        f"{self.base_url}/jobs/recordInfo",
                        params={"taskId": task_id},
                        headers={
                            "Authorization": self._auth_header(),
                            "User-Agent": "Videeo-Pipeline/1.0"
                        }
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    logger.debug(f"recordInfo response (attempt {attempt + 1}): state={data.get('data', {}).get('state')}")
                    
                    # CRITICAL: Check top-level API error first (code != 200)
                    if isinstance(data, dict):
                        api_code = data.get("code")
                        api_msg = data.get("msg", "")
                        if api_code is not None and api_code != 200:
                            error_message = f"{api_code} - {api_msg}"
                            logger.error(f"Image generation failed: {error_message}")
                            raise Exception(f"Image generation failed: {error_message}")
                    
                    # kie.ai uses data.state for status
                    state = (
                        data.get("data", {}).get("state", "").lower() or
                        data.get("state", "").lower() or
                        data.get("status", "").lower()
                    )
                    
                    if state == "success":
                        # KEY FINDING: Nano Banana returns data.resultJson as a JSON string
                        # Need to parse it and read resultUrls[0]
                        result_json_str = data.get("data", {}).get("resultJson")
                        
                        if result_json_str:
                            try:
                                result = json.loads(result_json_str)
                                urls = result.get("resultUrls", [])
                                if urls and len(urls) > 0:
                                    return urls[0]
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse resultJson: {e}, raw: {result_json_str[:200]}")
                        
                        # Fallback: try other common response formats
                        output = data.get("data", {}).get("output") or data.get("output")
                        
                        image_url = None
                        
                        if isinstance(output, str) and output.startswith("http"):
                            image_url = output
                        elif isinstance(output, dict):
                            image_url = output.get("url") or output.get("image") or output.get("image_url")
                        elif isinstance(output, list) and len(output) > 0:
                            first = output[0]
                            if isinstance(first, str) and first.startswith("http"):
                                image_url = first
                            elif isinstance(first, dict):
                                image_url = first.get("url") or first.get("image")
                        
                        # Also check for direct URL fields
                        if not image_url:
                            image_url = (
                                data.get("data", {}).get("imageUrl") or
                                data.get("data", {}).get("image_url") or
                                data.get("data", {}).get("url") or
                                data.get("imageUrl") or
                                data.get("image_url")
                            )
                        
                        if image_url:
                            return image_url
                        else:
                            raise Exception(f"No image URL in completed response: {data}")
                    
                    elif state in ["failed", "error"]:
                        error = (
                            data.get("data", {}).get("error") or
                            data.get("error") or
                            data.get("message") or
                            "Unknown error"
                        )
                        raise Exception(f"Image generation failed: {error}")
                    
                    elif state in ["pending", "processing", "running", "queued", "waiting", ""]:
                        # Still processing, wait and retry
                        logger.debug(f"Image task {task_id} state: {state}, attempt {attempt + 1}/{max_attempts}")
                        await asyncio.sleep(self.poll_interval)
                    else:
                        logger.warning(f"Unknown state '{state}', continuing to poll...")
                        await asyncio.sleep(self.poll_interval)
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Task not found, might still be initializing
                    logger.debug(f"Task {task_id} not found yet, retrying...")
                    await asyncio.sleep(self.poll_interval)
                else:
                    logger.error(f"HTTP error polling task: {e}")
                    await asyncio.sleep(self.poll_interval)
                
            except httpx.RequestError as e:
                logger.warning(f"Network error polling task {task_id}: {e}. Retrying...")
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Unexpected error polling task {task_id}: {e}")
                await asyncio.sleep(self.poll_interval)
        
        raise Exception(f"Image generation timed out after {max_attempts * self.poll_interval} seconds")
