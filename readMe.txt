Name: Obada Hamdan
Assignment 3 - CSE 4310

To run the code simply run main file which is my driver code. 


This code combines a video player application and a motion detection and object tracking algorithm. The video player uses tkinter and matplotlib to display the video frames with the detected and tracked objects using a MotionDetector class that applies various computer vision techniques to detect and track moving objects in the video stream. The video player provides a graphical user interface for selecting and displaying the video and allows the user to jump to the next or previous frames. The motion detection and object tracking algorithm detects moving objects by applying background subtraction, thresholding, dilation, and contour detection techniques and tracks them by maintaining a list of their centers and the number of consecutive frames they have been tracked. A circle is drawn around the tracked objects if they have been tracked for a sufficient number of frames to highlight them in the video stream.
