# imports 
import pyrealsense2 as rs
import cv2 as cv 
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
#import AiPhile 
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow.lite as tflite
from PIL import Image
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640

#Draw Color Constant
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = [147,20,255]    
INDIGO=[75,0,130]   
VIOLET=[238,130,238]   
GRAY=[127,127,127]  

point = (320, 240)

cwd = os.getcwd()

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
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

    cv.circle(color_image, point, 4, (0, 0, 255))
    distance = depth_image[point[1], point[0]]
    cv.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    images = np.hstack((color_image, depth_colormap))
    cv.imshow('RealSense', images)
    cv.waitKey(1)

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
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Create a mask for the color
    mask = cv.inRange(hsv, lower_color, upper_color)
    # Erode and dilate to remove accidental, small detections
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    return mask

def find_objects_of_color(depth_data,frame, color_mask, output_color):
    # Find contours in the mask
    contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Loop over the contours
    for contour in contours:
        if cv.contourArea(contour) > 100:  # Filter out small objects
            # Calculate the bounding box
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), output_color, 2)

            # Calculate the centroid
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the centroid on the frame
                cv.circle(frame, (cX, cY), 5, output_color, -1)
                cv.imshow('color', frame)
                if output_color == (0, 0, 255):
                    print('Found Red Obj at ',cX,' ',cY ,' depth =', depth_data[cY][cX])
                elif output_color == (0, 255, 255):
                    print('Found Yellow Obj at ',cX,' ',cY ,' depth =', depth_data[cY][cX])
                elif output_color == (0, 255, 0):
                    print('Found Green Obj at ', cX, ' ', cY ,' depth =', depth_data[cY][cX])

                elif output_color == (0, 0, 0):
                    print('Found black Obj at ', cX, ' ', cY, ' depth =', depth_data[cY][cX])

def find_base_by_color(depth_data,frame, color_mask, output_color,dxmin,dxmax,dymin,dymax):
    # Find contours in the mask
    contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Loop over the contours
    for contour in contours:
        if cv.contourArea(contour) > 100:  # Filter out small objects
            # Calculate the bounding box
            x, y, w, h = cv.boundingRect(contour)           
            
            # Calculate the centroid
            M = cv.moments(contour)
            #cv.rectangle(frame, (x, y), (x+w, y+h), output_color, 2)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if( cX > dxmin and cX < dxmax ) and ( cY > dymin and cY < dymax ):
                    # Draw the centroid on the frame
                    cv.rectangle(frame, (x, y), (x+w, y+h), output_color, 2)
                    cv.circle(frame, (cX, cY), 5, output_color, -1)
                    #Draw Label
                    label = '%s: %d mm' % ("RED BASE", depth_data[cY][cX])
                    labelSize, baseLine = cv.getTextSize(
                        label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(y, labelSize[1] + 10)
                    cv.rectangle(frame, (x, label_ymin - labelSize[1] - 10), (
                        x + labelSize[0], label_ymin + baseLine - 10), output_color, cv.FILLED)
                    cv.putText(frame, label, (x, label_ymin - 7),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    #cv.imshow('color', frame)


def show_distance(event, x, y, args, params):
    global point
    point = (x%640, y)

def image_process(depth_image):
    print("start processing")
def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels
def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def getAvailableCameraIds(max_to_test):
    available_ids = []
    for i in range(max_to_test):
        temp_camera = cv.VideoCapture(i)
        if temp_camera.isOpened():
            temp_camera.release()
            print("found camera with id {}".format(i))
            available_ids.append(i)
    return available_ids

def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode: 
        x, y, w, h =obDecoded.rect
        cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = obDecoded.polygon
        hull = points
  
        return hull



model_path = os.path.join(cwd, 'Conveyor32-model.lite')
label_path = os.path.join(cwd, 'label.txt')

serial_number = '040322071476' #D435i
#serial_number = '929522060114'  # Update this to your camera's serial number
pipeline = initialize_camera(serial_number)
#getAvailableCameraIds(10)

#cap = cv.VideoCapture(0,cv.CAP_DSHOW) # Windows Direct Show

#cap = cv.VideoCapture(2)
#cap.set(cv.CAP_PROP_FOURCC,  1196444237)
#cap.set(3,640)
#cap.set(4,640)
#cap.set(cv.CAP_PROP_FPS, 30)
#ret, frame = cap.read()
#image_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
#image_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
image_width = 640
image_height = 480


#print(ret)
#print(frame)
frame_counter =0
starting_time =time.time()
#load Model file and label object
interpreter = load_model(model_path)
labels = load_labels(label_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

# Get input index
input_index = input_details[0]['index']

while True:
    #ret, frame = cap.read()
    depth_data, color_data = get_frames(pipeline)
    frame = color_data
    #frame = color_data[0:480,0:480] #CROP
    #frame_rgb=frame
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_resized = cv.resize(frame_rgb, (width, height))
    #If Model use Float32
    frame_resized = frame_resized.astype(np.float32)
    frame_resized /= 255.
    #If Model use INT8
    #frame_resized = frame_resized.astype(np.int8)

    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)

    # set frame as input tensors
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # perform inference
    interpreter.invoke()
    # Get output tensor
    output_details = interpreter.get_output_details()
    #print(output_details) #TF V2
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    #positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    #classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    #scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    #print("-->",scores )
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * image_height)))
            xmin = int(max(1, (boxes[i][1] * image_width)))
            ymax = int(min(image_height, (boxes[i][2] * image_height)))
            xmax = int(min(image_width, (boxes[i][3] * image_width)))

            cv.rectangle(frame, (xmin, ymin),
                          (xmax, ymax), (10, 255, 0), 4)
            x_centroid = (xmin+xmax)//2
            y_centroid = (ymin+ymax)//2 
            #print((xmin+xmax)//2 , (ymin+ymax)//2) 
            obj_centroid = '%d ,%d' % (x_centroid, y_centroid)
            cv.putText(frame, obj_centroid, (x_centroid-50, y_centroid - 7),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)            
            # Draw label
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            cv.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (
                xmin + labelSize[0], label_ymin + baseLine - 10), GREEN, cv.FILLED)
            cv.putText(frame, label, (xmin, label_ymin - 7),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            #FIND BASE IN CONVEYOR
            red_mask = detect_color(frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
            find_base_by_color(depth_data,frame,red_mask,RED,xmin,xmax,ymin,ymax)





    #if time.time() - start >= 1:
    #    print('fps:', frame_counter)
    ##    frame_counter = 0
    #    start = time.time()

    ##cv.imshow('Object detector', frame)

    Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode:
        x, y, w, h = obDecoded.rect
        cv.rectangle(frame, (x, y), (x + w, y + h), ORANGE, 4)

    #fps = frame_counter/(time.time()-starting_time)
    #AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)

    # Press 'q' to quit
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()