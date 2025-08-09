Vehicle Traffic Counter using YOLOv5 & OpenCV
This project is a real-time vehicle detection, tracking, and counting system that uses YOLOv5 for object detection, OpenCV for video processing, and MySQL for storing traffic statistics.
It detects different vehicle types, tracks their movement direction (incoming/outgoing), and logs data periodically to a database.

📌 Features
Real-time Detection – Uses YOLOv5s for detecting cars, motorcycles, buses, and trucks.

Vehicle Tracking – Maintains unique IDs for each detected vehicle using IoU and Hungarian algorithm matching.

Direction Detection – Classifies vehicles as incoming or outgoing based on crossing a reference line.

Traffic Statistics – Counts incoming, outgoing, and on-road vehicles per type.

Database Logging – Saves traffic data into a MySQL database at regular intervals.

Visualization – Displays bounding boxes, counts, and FPS on video frames.

📂 Project Structure
📁 Vehicle-Traffic-Counter
│-- main.py             # Main application (detection, tracking, counting)
│-- connectsql.py       # Example script for inserting static data into MySQL
│-- coco.names          # COCO dataset class labels
│-- yolov5s.pt          # YOLOv5 small model weights
│-- traffic_video.mp4   # (Optional) Sample video for testing
│-- README.md           # Project documentation
⚙️ Requirements
Install dependencies:

pip install opencv-python torch torchvision ultralytics numpy mysql-connector-python scipy
🗄️ Database Setup
Create a MySQL database:

CREATE DATABASE traffic;
Create the traffic_data table:


CREATE TABLE traffic_data (
    date DATE,
    hour INT,
    small_in INT,
    big_in INT,
    small_out INT,
    big_out INT
);
Update MySQL connection details in:

main.py (inside db() method)

connectsql.py (for standalone test)

▶️ Usage
1. Run with a video file
python main.py
Make sure "traffic_video.mp4" exists in the same folder, or update the video path in:


tracker = EfficientVehicleTracker("traffic_video.mp4")
2. Run with webcam
tracker = EfficientVehicleTracker(0)
📊 Output
On-screen Display

Bounding boxes with vehicle type and ID

Incoming & outgoing counts

Vehicles currently on the road

FPS counter

Database Logging
Saves per-minute traffic statistics:

sql
date | hour | small_in | big_in | small_out | big_out

🚀 Future Improvements
Add speed estimation per vehicle.
