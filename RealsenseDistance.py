import cv2
import pyrealsense2
import statistics
import matplotlib.pyplot as plt
from realsense_depth import *

point = (400, 300)

# variable
frame_h = 480
frame_w = 640
avg_depth = [600]*640
max_vision = 600
min_vision = 100
depth_margin = 30
left_pos = 200
right_pos = 440
min_depth = max_vision
object_depth = max_vision


def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

# depth processing
def depth_process():
    for x in range(frame_w):
        avg_depth[x] = 0
        for y in range(frame_h):
            depth = depth_frame[y][x]
            if (depth <= max_vision and depth > min_vision):
                avg_depth[x] += depth
            else:
                avg_depth[x] += max_vision
        avg_depth[x] = int(avg_depth[x]/frame_h)

    # Step 1: Find Max value
    min_value = min(avg_depth)

    # Step 2: Find all positions close to maximum value (margin 30 mm)
    min_positions = [i for i, value in enumerate(avg_depth) if (abs(value - min_value) <= depth_margin)]
    for i in range(10):  # May be a spike recalculate to filter it out 8 times
        if (len(min_positions) < 30):
            for i in min_positions:
                avg_depth[i] = 0
            min_value = max(avg_depth)
            min_positions = [i for i, value in enumerate(avg_depth) if (abs(value - min_value) <= depth_margin)]

    # Step 3: find middle index
    middle_index = min_positions[len(min_positions) // 2]
    # Extracting the maximum values using their positions
    min_values = [avg_depth[i] for i in min_positions]
    # Calculating the average of these maximum values
    avg_min_values = int(statistics.mean(min_values))
    object_depth = avg_min_values
    real_depth = int((depth_frame[230][middle_index] + depth_frame[235][middle_index] + depth_frame[240][middle_index] + depth_frame[245][middle_index] + depth_frame[250][middle_index]+depth_frame[240][middle_index - 10] + depth_frame[240][middle_index - 5] + depth_frame[240][middle_index +5] + depth_frame[240][middle_index +10] )/9)

    # print values we can choose some values to publish
    print(avg_depth)
    print("Min Value= ", min_value)
    print("Min Average= ", avg_min_values)
    print("Middle Position = ", middle_index)
    print("Object depth = ", object_depth)
    print("Real depth = ", real_depth)
    if (middle_index < left_pos):
        print("Left")
    elif (middle_index > right_pos):
        print("Right")
    else:
        print("Center")

    if (real_depth < 100):
        print("Close")
    elif (real_depth > 500):
        print("Far")
    else:
        print("Perfect")

    # Plotting the avg_ivt_depth
    plt.figure(figsize=(10, 6))
    plt.plot(avg_depth, 'bo')
    plt.title('avg_ivt_depth with Middle Index of Max Values Marked')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # Marking the middle_index
    plt.scatter(middle_index, avg_depth[int(middle_index)], color='red', label='Middle Index', zorder=5)
    # Adding text to show the middle_index
    plt.text(middle_index, avg_depth[int(middle_index)],
             f' Middle Index: {middle_index} Real Depth: {real_depth} mm.', color='red', ha='right')
    plt.legend()
    plt.show()


while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]

    cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 13:
        depth_process()


