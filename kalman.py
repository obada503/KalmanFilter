import cv2
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

class MotionDetector:
    def __init__(self, hysteresis, motion_threshold, distance_threshold, skip_frames, max_objects, min_area):
        self.hysteresis = hysteresis
        self.motion_threshold = motion_threshold
        self.distance_threshold = distance_threshold
        self.skip_frames = skip_frames
        self.max_objects = max_objects
        self.frame_count = 0
        self.motion_frame = None
        self.tracked_objects = []
        self.min_area = min_area
        # Create background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)

    def euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        if point1 is None or point2 is None:
            return float('inf')
        else:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def update_tracking(self, frame):
        # Update tracking information for the frame
        if self.frame_count % (self.skip_frames + 1) == 0:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            # Threshold the mask
            _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
            # Dilate the mask
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the centers of the contours
            centers = []
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue

                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue

                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                centers.append((center_x, center_y))

            # Update tracked objects
            if self.tracked_objects:
                new_tracked_objects = []
                for center in centers:
                    min_distance = float('inf')
                    closest_object = None
                    for obj in self.tracked_objects:
                        distance = self.euclidean_distance(obj[1], center)
                        if distance < min_distance and distance < self.distance_threshold:
                            min_distance = distance
                            closest_object = obj

                    if closest_object:
                        closest_object[2] += 1
                        if closest_object[2] >= self.hysteresis:
                            closest_object[1] = center
                        new_tracked_objects.append(closest_object)
                    else:
                        if len(new_tracked_objects) < self.max_objects:
                            new_tracked_objects.append([frame, center, 1])

                self.tracked_objects = new_tracked_objects
            else:
                self.tracked_objects = [[frame, center, 1] for center in centers]

        self.frame_count += 1

    def draw_tracked_objects(self, frame):
        # Draw tracked objects on the frame
        img = np.array(frame)

        for obj in self.tracked_objects:
            if obj[2] >= self.hysteresis:
                center_x, center_y = obj[1]
                # Draw a circle around the tracked object
                cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), 2)

        # Return the frame with the tracked objects drawn
        return Image.fromarray(img)
