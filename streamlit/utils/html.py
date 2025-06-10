def video_tag(base64_video: str, video_id: str) -> str:
    """Generates an HTML video tag from a base64-encoded string.

    Args:
        base64_video (str): The base64-encoded MP4 video string.
        video_id (str): The HTML ID attribute for the video tag.

    Returns:
        str: HTML snippet containing a single video element.
    """
    return f"""
    <div style="flex: 1; padding: 0 10px;">
    <video id="{video_id}" style="width: 100%; max-width: 100%;" controls>
        <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    </div>
    """


def video_html(b64_original: str, b64_predicted: str) -> str:
    """Generates HTML to display two side-by-side videos and a play button.

    Args:
        b64_original (str): Base64-encoded original video.
        b64_predicted (str): Base64-encoded predicted video.

    Returns:
        str: HTML snippet containing two videos and a play button.
    """
    return f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: center; width: 100%;">
    {video_tag(b64_original, "vid1")}
    {video_tag(b64_predicted, "vid2")}
    </div>

    <div style="margin-top: 20px; text-align: center;">
    <button style="padding: 10px 20px; font-size: 16px;"
        onclick="document.getElementById('vid1').play(); document.getElementById('vid2').play();">
        ▶️ Play Both
    </button>
    </div>
    """
