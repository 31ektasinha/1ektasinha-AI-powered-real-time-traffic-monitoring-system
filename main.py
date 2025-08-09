import numpy as np
import mysql.connector as mc
from collections import defaultdict
import time
import cv2
import torch
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class EfficientVehicleTracker:
    def __init__(self, video_source=0, detection_frequency=3, confidence_threshold=0.5):
        """
        Initialize the vehicle tracker with optimized settings
        
        Args:
            video_source: Camera index or video file path
            detection_frequency: How often to run detection (every n frames)
            confidence_threshold: Confidence threshold for detections
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {video_source}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Define parameters
        self.detection_frequency = detection_frequency
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        
        # Load a more efficient pre-trained model (YOLOv5s)
        from ultralytics import YOLO
        self.model = YOLO("yolov5s.pt")

        
        # Set model to evaluation mode and move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        
        # Only detect vehicle classes
        self.vehicle_classes = [2, 3, 5, 7]  # COCO indices for car, motorcycle, bus, truck
        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Counter variables
        self.vehicles_on_road = defaultdict(int)
        self.incoming_counter = defaultdict(int)
        self.outgoing_counter = defaultdict(int)
        
        # Define crossing lines (adjust these for your specific setup)
        self.line_y = int(self.frame_height * 0.5)  # Middle of the frame by default
        self.direction_regions = {
            'incoming': (0, self.line_y),
            'outgoing': (self.line_y, self.frame_height)
        }
        
        # Vehicle tracking
        self.next_id = 0
        self.tracked_vehicles = {}  # {id: {type, box, centroid, frames_since_seen, crossed, direction}}
        self.max_disappeared = 15  # Maximum frames a vehicle can disappear before removing it
        
    def detect_vehicles(self, frame):
        """Run YOLOv5 object detection on the frame"""
        # Convert OpenCV BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(frame_rgb, verbose=False)  # returns list of Results
        detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # Filter detections to only include vehicles with confidence above threshold
        vehicle_detections = []
        for x1, y1, x2, y2, conf, cls in detections:
            if int(cls) in self.vehicle_classes and conf > self.confidence_threshold:
                vehicle_type = self.class_names[int(cls)]
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                vehicle_detections.append({
                    'type': vehicle_type,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'centroid': (int(centroid[0]), int(centroid[1])),
                    'confidence': float(conf)
                })
                
        return vehicle_detections
    
    def update_tracks(self, detections):
        """Update tracked vehicles with new detections using IoU matching"""
        # If there are no tracked vehicles yet, initialize all detections as new vehicles
        if not self.tracked_vehicles:
            for detection in detections:
                self.tracked_vehicles[self.next_id] = {
                    'type': detection['type'],
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'frames_since_seen': 0,
                    'crossed': False,
                    'direction': self.determine_region(detection['centroid'][1]),
                    'trajectory': [detection['centroid']]
                }
                self.vehicles_on_road[detection['type']] += 1
                self.next_id += 1
            return
        
        # Create matrices for IoU matching
        if detections and self.tracked_vehicles:
            # Compute IoU between each detection and tracked vehicle
            detection_boxes = np.array([d['bbox'] for d in detections])
            tracked_boxes = np.array([list(v['bbox']) for v in self.tracked_vehicles.values()])
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(detections), len(self.tracked_vehicles)))
            for i, det_box in enumerate(detection_boxes):
                for j, trk_box in enumerate(tracked_boxes):
                    iou_matrix[i, j] = self.calculate_iou(det_box, trk_box)
            
            # Find assignments using Hungarian algorithm
            from scipy.optimize import linear_sum_assignment
            det_indices, trk_indices = linear_sum_assignment(-iou_matrix)
            
            # Create lists to keep track of matched/unmatched objects
            matched_detections = []
            matched_trackers = []
            
            # Filter matches with low IoU
            for det_idx, trk_idx in zip(det_indices, trk_indices):
                if iou_matrix[det_idx, trk_idx] < 0.3:  # IoU threshold
                    continue
                
                matched_detections.append(det_idx)
                matched_trackers.append(list(self.tracked_vehicles.keys())[trk_idx])
                
            # Update matched tracks
            for det_idx, trk_id in zip(matched_detections, matched_trackers):
                detection = detections[det_idx]
                self.tracked_vehicles[trk_id]['bbox'] = detection['bbox']
                self.tracked_vehicles[trk_id]['centroid'] = detection['centroid']
                self.tracked_vehicles[trk_id]['frames_since_seen'] = 0
                self.tracked_vehicles[trk_id]['trajectory'].append(detection['centroid'])
                
                # Update direction based on trajectory
                if len(self.tracked_vehicles[trk_id]['trajectory']) >= 3:
                    y_values = [p[1] for p in self.tracked_vehicles[trk_id]['trajectory'][-3:]]
                    if y_values[2] > y_values[0]:  # Moving down (outgoing)
                        self.tracked_vehicles[trk_id]['direction'] = 'outgoing'
                    elif y_values[2] < y_values[0]:  # Moving up (incoming)
                        self.tracked_vehicles[trk_id]['direction'] = 'incoming'
                
                # Check if vehicle crossed the line
                self.check_line_crossing(trk_id)
            
            # Initialize new tracks for unmatched detections
            unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
            for idx in unmatched_detections:
                detection = detections[idx]
                self.tracked_vehicles[self.next_id] = {
                    'type': detection['type'],
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'frames_since_seen': 0,
                    'crossed': False,
                    'direction': self.determine_region(detection['centroid'][1]),
                    'trajectory': [detection['centroid']]
                }
                self.vehicles_on_road[detection['type']] += 1
                self.next_id += 1
            
            # Update unmatched trackers
            unmatched_trackers = [trk_id for trk_id in self.tracked_vehicles.keys() 
                                if trk_id not in matched_trackers]
            
            # Increment disappeared counter for unmatched trackers
            expired_trackers = []
            for trk_id in unmatched_trackers:
                self.tracked_vehicles[trk_id]['frames_since_seen'] += 1
                if self.tracked_vehicles[trk_id]['frames_since_seen'] > self.max_disappeared:
                    expired_trackers.append(trk_id)
            
            # Remove expired trackers
            for trk_id in expired_trackers:
                vehicle_type = self.tracked_vehicles[trk_id]['type']
                # Only decrement if we're sure the vehicle has left
                if self.vehicles_on_road[vehicle_type] > 0:
                    self.vehicles_on_road[vehicle_type] -= 1
                del self.tracked_vehicles[trk_id]
        
    def check_line_crossing(self, vehicle_id):
        """Check if a vehicle has crossed the counting line"""
        if vehicle_id not in self.tracked_vehicles:
            return
            
        vehicle = self.tracked_vehicles[vehicle_id]
        if vehicle['crossed']:
            return
            
        # Get recent y positions from trajectory
        if len(vehicle['trajectory']) < 2:
            return
            
        prev_y = vehicle['trajectory'][-2][1]
        curr_y = vehicle['trajectory'][-1][1]
        
        # Check if crossing line from top to bottom or bottom to top
        if (prev_y < self.line_y and curr_y >= self.line_y) or (prev_y > self.line_y and curr_y <= self.line_y):
            vehicle['crossed'] = True
            direction = 'incoming' if prev_y > self.line_y else 'outgoing'
            
            # Update counters
            if direction == 'incoming':
                self.incoming_counter[vehicle['type']] += 1
            else:
                self.outgoing_counter[vehicle['type']] += 1
    
    def determine_region(self, y_coord):
        """Determine which region a y-coordinate belongs to"""
        if y_coord < self.line_y:
            return 'incoming'
        return 'outgoing'
    
    def calculate_iou(self, boxA, boxB):
        """Calculate IoU between two bounding boxes"""
        # Convert format from (x1, y1, x2, y2) to xywh if necessary
        if len(boxA) == 4 and len(boxB) == 4:
            # Calculate intersection area
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            # Intersection area
            intersection_area = max(0, xB - xA) * max(0, yB - yA)
            
            # Union Area
            boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            union_area = boxA_area + boxB_area - intersection_area
            
            # Return IoU
            return intersection_area / max(union_area, 1e-6)
        return 0
    
    def draw_visualization(self, frame):
        """Draw visualization on the frame"""
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), 
                 (0, 255, 0), 2)
        
        # Draw direction labels
        cv2.putText(frame, "Incoming", (10, self.line_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Outgoing", (10, self.line_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw vehicles
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            x1, y1, x2, y2 = vehicle_data['bbox']
            
            # Set color based on direction
            color = (0, 255, 0) if vehicle_data['direction'] == 'incoming' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and type
            label = f"{vehicle_data['type']} #{vehicle_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw centroid
            cv2.circle(frame, vehicle_data['centroid'], 4, color, -1)
        
        # Draw counters
        stats_x = 10
        stats_y = 30
        
        # Total counter
        total_incoming = sum(self.incoming_counter.values())
        total_outgoing = sum(self.outgoing_counter.values())
        total_on_road = sum(self.vehicles_on_road.values())
        
        cv2.putText(frame, f"Incoming: {total_incoming}", (stats_x, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        stats_y += 30
        
        cv2.putText(frame, f"Outgoing: {total_outgoing}", (stats_x, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stats_y += 30
        
        cv2.putText(frame, f"On Road: {total_on_road}", (stats_x, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        stats_y += 30
        
        # Vehicle type breakdown
        for vtype, count in self.vehicles_on_road.items():
            if count > 0:
                cv2.putText(frame, f"{vtype}: {count}", (stats_x, stats_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                stats_y += 25
        
        return frame
    
    def get_statistics(self):
        """Return current traffic statistics"""
        return {
            "incoming": dict(self.incoming_counter),
            "outgoing": dict(self.outgoing_counter),
            "on_road": dict(self.vehicles_on_road),
            "total_incoming": sum(self.incoming_counter.values()),
            "total_outgoing": sum(self.outgoing_counter.values()),
            "total_on_road": sum(self.vehicles_on_road.values())
        }
    
    def db(self, small_in, big_in, small_out, big_out):
        try:
            # Get the current date and time
            now = datetime.datetime.now()
            date = now.date()  # Date in YYYY-MM-DD format
            hour = now.hour  # Get current hour (24-hour format)
            minute = now.minute  # Get current minute
            connect = mc.connect(host= "localhost", user = 'root', password = '1234', port = 3307, database = 'traffic')
            # Check database connection
            if connect.is_connected():
                cursor = connect.cursor()
                print('cursor executed')

                # Prepare the SQL query with parameterized values
                query = """
                    INSERT INTO traffic_data (date, hour, small_in, big_in, small_out, big_out)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                data = (date, hour, small_in, big_in, small_out, big_out)

                # Execute the query and commit the transaction
                cursor.execute(query, data)
                connect.commit()
                connect.close()

                print(f'Query successfully executed at {now.strftime("%Y-%m-%d %H:%M:%S")}')
            else:
                print("Database connection is not established.")
        except Exception as e:
            print(f"Error occurred while inserting data into the database: {e}")

    def run(self):
        """Main processing loop"""
        start_time = time.time()
        start_time_push = time.time()
        frames_processed = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frames_processed += 1
            
            # Only run detection every n frames for efficiency
            if self.frame_count % self.detection_frequency == 0:
                detections = self.detect_vehicles(frame)
                self.update_tracks(detections)
            
            # Draw visualization
            output_frame = self.draw_visualization(frame)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frames_processed / elapsed_time
                cv2.putText(output_frame, f"FPS: {fps:.2f}", 
                            (self.frame_width - 120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Vehicle Traffic Counter", output_frame)
            
            # Print statistics periodically
            if self.frame_count % (self.detection_frequency * 10) == 0:
                stats = self.get_statistics()
                print("\nTraffic Statistics:")
                print(f"Incoming: {stats['incoming']} (Total: {stats['total_incoming']})")
                print(f"Outgoing: {stats['outgoing']} (Total: {stats['total_outgoing']})")
                print(f"Currently on road: {stats['on_road']} (Total: {stats['total_on_road']})")
                current_time = time.time()
                elapsed_time_push = current_time - start_time_push
                if elapsed_time_push >=  60:
                    print('in the db pushblock')
                    small_in = stats['incoming'].get('car', 0) + stats['incoming'].get('motorbike',0)
                    big_in = stats['incoming'].get('bus', 0) + stats['incoming'].get('truck',0)
                    small_out = stats['outgoing'].get('car', 0) + stats['outgoing'].get('motorbike',0)
                    big_out = stats['outgoing'].get('bus', 0) + stats['outgoing'].get('truck',0)
                    for key in self.incoming_counter:
                        self.incoming_counter[key] = 0
                    for key in self.outgoing_counter:
                        self.outgoing_counter[key] = 0
                        start_time_push = current_time
                    try:
                        self.db(small_in, big_in, small_out, big_out)
                        start_time_push = current_time
                    except Exception as e:
                        print('some exception occured')
                else:
                    print(elapsed_time_push)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Pass video file path or camera index
        tracker = EfficientVehicleTracker("traffic_video.mp4")
        # tracker = EfficientVehicleTracker(0)  # For webcam
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
