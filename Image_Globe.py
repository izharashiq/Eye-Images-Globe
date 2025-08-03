import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
import random

if not os.path.exists("eye_images"):
    os.makedirs("eye_images")

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

scanning = True
capturing = False
face_detected = False
captured_images = 0
last_capture_time = time.time()
eye_images = []
eye_positions = []
globe_points = []
source_eye_idx = 15
auto_rotation = 0.0

scan_y = 0
scan_speed = 3
scan_direction = 1
face_scan_complete = False
scan_phase_start = time.time()

current_rotation = {"x": 0, "y": 0}
source_eye_box = {} 

current_eye_crop = None

font = cv2.FONT_HERSHEY_SIMPLEX


def create_matrix_background(frame):
    """Return original frame without any B&W or rain."""
    return frame

def draw_scan_line(frame, y_pos, width):
    cv2.line(frame, (0, y_pos), (width, y_pos), (0, 255, 0), 3)
    
    cv2.line(frame, (0, y_pos-1), (width, y_pos-1), (0, 150, 0), 1)
    cv2.line(frame, (0, y_pos+1), (width, y_pos+1), (0, 150, 0), 1)
    cv2.line(frame, (0, y_pos-2), (width, y_pos-2), (0, 80, 0), 1)
    cv2.line(frame, (0, y_pos+2), (width, y_pos+2), (0, 80, 0), 1)
    
    cv2.circle(frame, (20, y_pos), 5, (0, 255, 0), -1)
    cv2.circle(frame, (width-20, y_pos), 5, (0, 255, 0), -1)

def draw_matrix_text(frame, text, x, y, size=0.6, thickness=2):

    cv2.putText(frame, text, (x+1, y+1), font, size, (0, 100, 0), thickness+1)

    cv2.putText(frame, text, (x, y), font, size, (0, 255, 0), thickness)

def draw_face_grid(frame, landmarks, width, height):
    if not landmarks:
        return
    
    face_points = []
    face_indices = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 125, 142, 36, 205, 206, 207, 213, 192, 147, 187, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
    
    for idx in face_indices:
        x = int(landmarks[idx].x * width)
        y = int(landmarks[idx].y * height)
        face_points.append((x, y))
    
    for i in range(len(face_points)):
        start_point = face_points[i]
        end_point = face_points[(i + 1) % len(face_points)]
        cv2.line(frame, start_point, end_point, (0, 150, 0), 1)
        
    for point in face_points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

def generate_sphere_points(num_points, radius=150):
    points = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    
    for i in range(num_points):
        y = 1 - (i / (num_points - 1)) * 2
        radius_at_y = math.sqrt(1 - y * y)
        
        theta = golden_angle * i
        
        x = math.cos(theta) * radius_at_y * radius
        z = math.sin(theta) * radius_at_y * radius
        y = y * radius
        
        points.append([x, y, z])
    return points

def get_hand_orientation(landmarks, width, height):
    try:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]
        
        wrist_pos = np.array([wrist.x * width, wrist.y * height, wrist.z * 100])
        middle_pos = np.array([middle_mcp.x * width, middle_mcp.y * height, middle_mcp.z * 100])
        index_pos = np.array([index_mcp.x * width, index_mcp.y * height, index_mcp.z * 100])
        
        palm_vector = middle_pos - wrist_pos
        side_vector = index_pos - wrist_pos
        normal = np.cross(palm_vector, side_vector)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        rotation_y = math.atan2(normal[0], normal[2])
        rotation_x = math.atan2(-normal[1], math.sqrt(normal[0]**2 + normal[2]**2))
        
        return rotation_x, rotation_y
    except:
        return 0, 0

def extract_right_eye(landmarks, frame, width, height):

    right_eye_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    coords = []
    for i in right_eye_ids:
        x = int(landmarks[i].x * width)
        y = int(landmarks[i].y * height)
        coords.append((x, y))
    
    if not coords:
        return None, (0, 0, 0, 0)
    
    x_coords, y_coords = zip(*coords)
    min_x, max_x = max(min(x_coords) - 25, 0), min(max(x_coords) + 25, width)
    min_y, max_y = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, height)
    
    if max_x - min_x < 50:
        center_x = (min_x + max_x) // 2
        min_x, max_x = max(center_x - 25, 0), min(center_x + 25, width)
    if max_y - min_y < 35:
        center_y = (min_y + max_y) // 2
        min_y, max_y = max(center_y - 17, 0), min(center_y + 18, height)
    
    eye_crop = frame[min_y:max_y, min_x:max_x]
    return eye_crop, (min_x, min_y, max_x - min_x, max_y - min_y)

def draw_matrix_box(frame, x, y, width, height, label="", is_source=False):
    if is_source:
        cv2.rectangle(frame, (x-4, y-4), (x+width+4, y+height+4), (0, 255, 0), 3)
        cv2.rectangle(frame, (x-2, y-2), (x+width+2, y+height+2), (0, 200, 0), 1)
        cv2.rectangle(frame, (x-1, y-1), (x+width+1, y+height+1), (0, 150, 0), 1)
        
        bracket_size = 15
        cv2.line(frame, (x-4, y-4), (x-4+bracket_size, y-4), (0, 255, 0), 3)
        cv2.line(frame, (x-4, y-4), (x-4, y-4+bracket_size), (0, 255, 0), 3)
        cv2.line(frame, (x+width+4, y-4), (x+width+4-bracket_size, y-4), (0, 255, 0), 3)
        cv2.line(frame, (x+width+4, y-4), (x+width+4, y-4+bracket_size), (0, 255, 0), 3)

        cv2.line(frame, (x-4, y+height+4), (x-4+bracket_size, y+height+4), (0, 255, 0), 3)
        cv2.line(frame, (x-4, y+height+4), (x-4, y+height+4-bracket_size), (0, 255, 0), 3)
        cv2.line(frame, (x+width+4, y+height+4), (x+width+4-bracket_size, y+height+4), (0, 255, 0), 3)
        cv2.line(frame, (x+width+4, y+height+4), (x+width+4, y+height+4-bracket_size), (0, 255, 0), 3)
        
        if label:
            label_width = len(label) * 10 + 10
            cv2.rectangle(frame, (x+width+8, y), (x+width+8+label_width, y+25), (0, 255, 0), -1)
            cv2.rectangle(frame, (x+width+8, y), (x+width+8+label_width, y+25), (0, 150, 0), 2)
            cv2.putText(frame, label, (x+width+12, y+18), font, 0.5, (0, 0, 0), 2)
    else:
        cv2.rectangle(frame, (x-1, y-1), (x+width+1, y+height+1), (0, 200, 0), 2)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 100, 0), 1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    frame = create_matrix_background(frame)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if scanning:
        draw_scan_line(frame, scan_y, width)
        
        scan_y += scan_speed
        if scan_y >= height - 20:
            scanning = False
            capturing = True
            face_scan_complete = True
        
        if face_results.multi_face_landmarks:
            if not face_detected:
                face_detected = True
            
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            draw_face_grid(frame, landmarks, width, height)
            
            if time.time() - scan_phase_start > 3.0:
                scanning = False
                capturing = True
                face_scan_complete = True

    elif capturing and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        eye_crop, eye_bounds = extract_right_eye(landmarks, frame, width, height)
        
        if eye_crop is not None and eye_crop.shape[0] > 0 and eye_crop.shape[1] > 0:
            current_eye_crop = eye_crop.copy()
            
            if eye_bounds is not None:
                box_x, box_y, box_w, box_h = eye_bounds
                
                cv2.rectangle(frame, (box_x, box_y),
                              (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

            eye_resized = cv2.resize(eye_crop, (box_w, box_h))
            if box_y + box_h <= frame.shape[0] and box_x + box_w <= frame.shape[1]:
                frame[box_y:box_y+box_h, box_x:box_x+box_w] = eye_resized

            draw_matrix_box(frame, box_x, box_y, box_w, box_h, "SOURCE", True)
            
            draw_matrix_text(frame, f"IMAGES CAPTURED: {captured_images}/30", 20, 30)
            
            bar_width = 300
            bar_x = (width - bar_width) // 2
            bar_y = height - 50
            progress = captured_images / 30.0
            
            cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x+bar_width+2, bar_y+22), (0, 255, 0), 2)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_width*progress), bar_y+20), (0, 200, 0), -1)
            
            if time.time() - last_capture_time > 0.4:
                if captured_images < 30:
                    eye_sample = cv2.resize(eye_crop, (70, 45))
                    eye_images.append(eye_sample)
                    
                    pos_x = random.randint(200, width - 150)
                    pos_y = random.randint(200, height - 100)
                    eye_positions.append((pos_x, pos_y))
                    
                    filename = f"eye_images/{captured_images:03}.jpg"
                    cv2.imwrite(filename, eye_crop)
                    
                    captured_images += 1
                    last_capture_time = time.time()
                    
                else:
                    capturing = False
                    globe_points = generate_sphere_points(len(eye_images))

        for i, img in enumerate(eye_images):
            x, y = eye_positions[i]
            if img.shape[0] > 0 and img.shape[1] > 0:
                frame[y:y+45, x:x+70] = img
                draw_matrix_box(frame, x, y, 70, 45, f"{i+1:02d}")

    elif not capturing and len(globe_points) > 0:
        auto_rotation = 0
        
        hand_center_x, hand_center_y = width // 2, height // 2
        hand_rotation_x, hand_rotation_y = 0, 0

        hand_closed = False
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
        
            index_tip = lm[8]
            index_base = lm[5]
        
            distance = math.hypot(index_tip.x - index_base.x, index_tip.y - index_base.y)
            hand_closed = distance < 0.05
        
        if hand_results.multi_hand_landmarks:
            
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            key_points = [0, 4, 8, 12, 16, 20]
            for point_idx in key_points:
                landmark = hand_landmarks.landmark[point_idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 150, 0), -1)
            
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
            for point_idx in [4, 8, 12, 16, 20]:
                tip = hand_landmarks.landmark[point_idx]
                tip_x, tip_y = int(tip.x * width), int(tip.y * height)
                cv2.line(frame, (wrist_x, wrist_y), (tip_x, tip_y), (0, 150, 0), 1)
            
            hand_center_x = int(hand_landmarks.landmark[9].x * width)
            hand_center_y = int(hand_landmarks.landmark[9].y * height)
            
            hand_rotation_x, hand_rotation_y = get_hand_orientation(hand_landmarks.landmark, width, height)
            
            cv2.circle(frame, (hand_center_x, hand_center_y), 8, (0, 255, 0), 2)
            cv2.circle(frame, (hand_center_x, hand_center_y), 4, (0, 200, 0), -1)
        
        total_rotation_x = hand_rotation_x * 2.0
        total_rotation_y = hand_rotation_y * 2.0

        
        cos_y, sin_y = math.cos(total_rotation_y), math.sin(total_rotation_y)
        cos_x, sin_x = math.cos(total_rotation_x), math.sin(total_rotation_x)
        
        eye_data = []
        for i, (img, point) in enumerate(zip(eye_images, globe_points)):
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
                
            x, y, z = point
            
            x_rot = x * cos_y - z * sin_y
            z_rot = x * sin_y + z * cos_y
            
            y_rot = y * cos_x - z_rot * sin_x
            z_final = y * sin_x + z_rot * cos_x
            
            focal_length = 600
            perspective = focal_length / (focal_length + z_final + 150)
            
            screen_x = int(hand_center_x + x_rot * perspective)
            screen_y = int(hand_center_y + y_rot * perspective * 0.75)
            
            eye_data.append((z_final, screen_x, screen_y, img, i, perspective))
        
        eye_data.sort(key=lambda x: x[0])
        
        for depth, ex, ey, img, idx, scale in eye_data:
            if hand_closed:
                continue

            img_w = int(70 * scale)
            img_h = int(45 * scale)
            
            if img_w > 12 and img_h > 8:
                img_scaled = cv2.resize(img, (img_w, img_h))
                
                x1, x2 = ex - img_w//2, ex - img_w//2 + img_w
                y1, y2 = ey - img_h//2, ey - img_h//2 + img_h
                
                if (10 <= x1 and x2 <= width-10 and 10 <= y1 and y2 <= height-10):
                    alpha = min(1.0, scale * 1.4)
                    if alpha > 0.4:
                        roi = frame[y1:y2, x1:x2]
                        blended = cv2.addWeighted(roi, 1-alpha*0.9, img_scaled, alpha*0.9, 0)
                        frame[y1:y2, x1:x2] = blended
                    
                    if idx == source_eye_idx:
                        draw_matrix_box(frame, x1, y1, img_w, img_h, "SRC", True)
                    else:
                        draw_matrix_box(frame, x1, y1, img_w, img_h)
                        
                        if scale > 0.8:
                            cv2.putText(frame, f"{idx+1:02d}", 
                                      (x1-15, y1+img_h//2), font, 0.3, (0, 200, 0), 1)


    cv2.namedWindow("IMAGE GLOBE MOVE WITH HANDS", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("IMAGE GLOBE MOVE WITH HANDS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("IMAGE GLOBE MOVE WITH HANDS", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[EYE IMAGE GLOBE] Connection terminated.")