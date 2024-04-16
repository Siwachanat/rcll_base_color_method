import pyrealsense2 as rs
import numpy as np
import cv2
import time

point = (320, 240)
def initialize_camera(serial):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # This will enable the device with the given serial number
    config.enable_device(serial)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline

def get_frames(pipeline):
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    except Exception as e:
        print(e)
        return None, None


def visualize_data(depth_image, color_image):
    # Display depth and color images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    cv2.circle(color_image, point, 4, (0, 0, 255))
    distance = depth_image[point[1], point[0]]
    cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    images = np.hstack((color_image, depth_colormap))
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

def print_data(depth_image, color_image, x, y):
    # Display depth data
    print('x:y@d', x, y, '@', depth_image[y][x], 'c',color_image[y][x])

def printHsv(depth_data,frame):
    # Red color detection range
    red_mask = detect_color(frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
    find_objects_of_color(depth_data, frame, red_mask, (0, 0, 255))  # BGR for red
    # Yellow color detection range
    yellow_mask = detect_color(frame, np.array([20, 100, 100]), np.array([30, 255, 255]))
    find_objects_of_color(depth_data, frame, yellow_mask, (0, 255, 255))  # BGR for yellow
    # Green color detection range
    green_mask = detect_color(frame, np.array([40, 40, 40]), np.array([90, 255, 255]))
    find_objects_of_color(depth_data, frame, green_mask, (0, 255, 0))  # BGR for
    # Black color detection
    black_mask = detect_color(frame, np.array([0, 0, 0]), np.array([360, 255, 50]))
    find_objects_of_color(depth_data, frame, black_mask, (0, 0, 0))


def detect_color(img, lower_color, upper_color):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Erode and dilate to remove accidental, small detections
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def find_objects_of_color(depth_data,frame, color_mask, output_color):
    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop over the contours
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small objects
            # Calculate the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), output_color, 2)

            # Calculate the centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, output_color, -1)
                cv2.imshow('color', frame)
                if output_color == (0, 0, 255):
                    print('Found Red Obj at ',cX,' ',cY ,' depth =', depth_data[cY][cX])
                elif output_color == (0, 255, 255):
                    print('Found Yellow Obj at ',cX,' ',cY ,' depth =', depth_data[cY][cX])
                elif output_color == (0, 255, 0):
                    print('Found Green Obj at ', cX, ' ', cY ,' depth =', depth_data[cY][cX])

                elif output_color == (0, 0, 0):
                    print('Found black Obj at ', cX, ' ', cY, ' depth =', depth_data[cY][cX])

def show_distance(event, x, y, args, params):
    global point
    point = (x%640, y)

def image_process(depth_image):
    print("start processing")

if __name__ == "__main__":
    # Provide the serial number of your RealSense camera
    serial_number = '929522060114'  # Update this to your camera's serial number
    pipeline = initialize_camera(serial_number)

    # Create mouse event
    cv2.namedWindow("RealSense")
    cv2.setMouseCallback("RealSense", show_distance)
    try:
        while True:
            start_time = time.time()  # Capture the start time

            depth_data, color_data = get_frames(pipeline)
            if depth_data is not None and color_data is not None:
                #Processing part
                #show depth and color frame
                visualize_data(depth_data, color_data)

                #print depth at pos X Y
                #print_data(depth_data, color_data, 320,240 )
                print_data(depth_data, color_data, point[0],point[1])

                #detect color
                printHsv(depth_data, color_data)

                #image processing here
                #image_process(depth_data)

            # Calculate the time taken to process and display the frame
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.5:
                time.sleep(0.5 - elapsed_time)  # Delay to achieve 2 fps

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
