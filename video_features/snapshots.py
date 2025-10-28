import cv2
import os
from PIL import Image, ImageDraw, ImageFont

def create_video_frames_grid(video_directory="videos", output_frames_directory="output_frames", output_grid_image="video_grid.jpeg"):
    """
    Reads videos from a directory, extracts a frame at 60 seconds from each,
    and combines them into a single grid image with 10 columns.

    Each image in the grid will have a label with the video name.
    The final grid image and the individual video frames are saved as JPEGs.
    """
    if not os.path.exists(output_frames_directory):
        os.makedirs(output_frames_directory)

    frame_files = []
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            cap = cv2.VideoCapture(video_path)

            # Set the time to 60 seconds (in milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, 60000)

            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_frames_directory, f"{os.path.splitext(filename)[0]}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_files.append((frame_filename, filename))
            cap.release()

    if not frame_files:
        print("No video frames were extracted. Please check the video directory and files.")
        return

    # Create the image grid
    images = [Image.open(f[0]) for f in frame_files]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)

    num_images = len(images)
    cols = 10
    rows = (num_images + cols - 1) // cols

    grid_width = cols * max_width
    grid_height = rows * (max_height + 40)  # Add space for labels

    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for i, (image, label) in enumerate(zip(images, [f[1] for f in frame_files])):
        row = i // cols
        col = i % cols
        x_offset = col * max_width
        y_offset = row * (max_height + 40)
        grid_image.paste(image, (x_offset, y_offset))
        draw.text((x_offset + 10, y_offset + max_height + 10), label, font=font, fill='black')

    grid_image.save(output_grid_image)
    print(f"Grid image saved to {output_grid_image}")
    print(f"Individual frames saved in the '{output_frames_directory}' directory.")

if __name__ == "__main__":
    create_video_frames_grid('/media/nik/1E9AAD029AACD793/psyco/video/')