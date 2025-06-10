def video_tag(base64_video: str, video_id: str):
    return f"""
    <div style="flex: 1; padding: 0 10px;">
    <video id="{video_id}" style="width: 100%; max-width: 100%;" controls>
        <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    </div>
    """


def video_html(b64_original: str, b64_predicted: str):
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
