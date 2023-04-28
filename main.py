import tkinter as tk
from tkinter import filedialog
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageTk
from kalman import MotionDetector

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VideoPlayer(tk.Frame):
    def __init__(self, master, object_tracker, **kwargs):
        super().__init__(master, **kwargs)

        self.master = master
        self.tracker = object_tracker

        self.vid = None
        self.current_frame = 0
        self.create_widgets()

    def load_video(self):
        # Opens file dialog to select a video file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load the video using imageio
            self.vid = imageio.get_reader(file_path, format='FFMPEG', mode='I')
            self.update_frame()

    def update_frame(self):
        if self.vid is None:
            return

        # Get the current frame from the video
        frame = self.vid.get_data(self.current_frame)
        if frame is not None:
            # Update the object tracker with the current frame
            self.tracker.update_tracking(frame)
            # Draw the tracked objects on the frame
            tracked_frame = self.tracker.draw_tracked_objects(Image.fromarray(frame, mode='RGB'))

            # Display the tracked frame using matplotlib
            self.ax.clear()
            self.ax.imshow(tracked_frame)
            self.canvas.draw()

        # Move to the next frame
        self.current_frame += 1
        if self.current_frame >= self.vid.count_frames():
            self.current_frame = 0

    def jump_frames(self, num_frames):
        if self.vid is None:
            return

        # Jump to the specified number of frames
        self.current_frame += num_frames
        self.current_frame = max(0, min(self.current_frame, self.vid.count_frames() - 1))
        self.update_frame()

    def create_widgets(self):
        # Create the matplotlib figure and axis
        fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack()

        # Create the Load Video button
        self.load_button = tk.Button(self, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Create the Previous 60 Frames button
        self.prev_button = tk.Button(self, text="Previous 60 Frames", command=lambda: self.jump_frames(-60))
        self.prev_button.pack()

        # Create the Next 60 Frames button
        self.next_button = tk.Button(self, text="Next 60 Frames", command=lambda: self.jump_frames(60))
        self.next_button.pack()

def main():
    root = tk.Tk()

    # Set motion detector parameters
    motion_detector_params = (8, 40, 80, 0, 15, 50)
    # Create placeholder frames
    frame1, frame2, frame3 = np.zeros((480, 640)), np.zeros((480, 640)), np.zeros((480, 640))
    motion_detector = MotionDetector(*motion_detector_params)

    # Create the VideoPlayer app
    app = VideoPlayer(root, motion_detector)
    app.pack()

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
