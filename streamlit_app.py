import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import IPython.display as display
from IPython.display import HTML
from google.colab import files

# Path to the uploaded video file
video_path = "/content/sample-5s.mp4"  # Change this to your video filename
output_video_path = "/content/output_video_with_common_caption.mp4"  # Path to save the new video

# Load the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the video frame width, height, and FPS (frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the video writer to write the new video with captions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the first frame (or any frame you want) to generate one caption
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
    else:
        # Convert the frame to a PIL image for captioning
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Generate the caption for this frame (the same caption will be used for all frames)
        inputs = processor(pil_image, return_tensors="pt")
        out_text = model.generate(**inputs)
        common_caption = processor.decode(out_text[0], skip_special_tokens=True)

        # Go back to the start of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read frames and overlay the common caption
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            # Overlay the common caption on the frame
            cv2.putText(frame, common_caption, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write the frame with the common caption to the output video
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print("Video Processing Completed!")

# Function to display the processed video in Colab
def display_video(video_path):
    # Display the processed video using HTML5 video tag
    display.display(HTML(f'<video width="640" height="480" controls><source src="{video_path}" type="video/mp4"></video>'))

# Display the processed video with the common caption throughout
display_video(output_video_path)

# Allow the user to download the processed video
files.download(output_video_path)
