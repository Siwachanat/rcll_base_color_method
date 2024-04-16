import pyrealsense2 as rs

def list_realsense_devices():
    # Create a context object. This object owns the handles to all connected realsense devices
    context = rs.context()
    if len(context.devices) > 0:
        for d in context.devices:
            print(f"Device: {d.get_info(rs.camera_info.name)}")
            print(f"  Serial Number: {d.get_info(rs.camera_info.serial_number)}")
    else:
        print("No Intel RealSense devices were found.")

if __name__ == "__main__":
    list_realsense_devices()
