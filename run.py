import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from brain import RobotNurseBrain

# ==========================================
# 1. 2D GRID ENVIRONMENT
# ==========================================
class HospitalGridWorld:
    def __init__(self, grid_size=(6, 9), cell_pixels=65):
        self.rows, self.cols = grid_size
        self.cell_pixels = cell_pixels
        self.width = self.cols * cell_pixels
        self.height = self.rows * cell_pixels

        # Discrete key coordinates (row, col)
        self.locations = {
            "IDLE_BAY": (3, 1),
            "STAGING_AREA": (3, 4),
            "AT_NURSE": (3, 7)
        }
        
        # Continuous visual position (for smooth animation)
        self.robot_pos = list(self.locations["IDLE_BAY"])
        self.nurse_pos = (3, 8)

    def step_towards(self, target_state_name, speed=0.15):
        """Moves robot gradually towards the target waypoint on the grid."""
        target_r, target_c = self.locations[target_state_name]
        
        # Smooth interpolation
        self.robot_pos[0] += (target_r - self.robot_pos[0]) * speed
        self.robot_pos[1] += (target_c - self.robot_pos[1]) * speed

    def render(self, robot_state, belief_score, current_action):
        """Draws the 2D top-down grid representation."""
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 245

        # 1. Draw grid lines
        for r in range(self.rows + 1):
            y = r * self.cell_pixels
            cv2.line(canvas, (0, y), (self.width, y), (215, 215, 215), 1)
        for c in range(self.cols + 1):
            x = c * self.cell_pixels
            cv2.line(canvas, (x, 0), (x, self.height), (215, 215, 215), 1)

        # 2. Draw designated zones
        def draw_cell_zone(coord, color, label):
            r, c = coord
            x1, y1 = c * self.cell_pixels, r * self.cell_pixels
            x2, y2 = x1 + self.cell_pixels, y1 + self.cell_pixels
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(canvas, label, (x1 + 4, y1 + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1)

        draw_cell_zone(self.locations["IDLE_BAY"], (235, 220, 200), "Bay")
        draw_cell_zone(self.locations["STAGING_AREA"], (210, 240, 210), "Staging")
        draw_cell_zone(self.locations["AT_NURSE"], (200, 220, 255), "Handover")

        # 3. Draw Nurse & Sterile Field
        nr, nc = self.nurse_pos
        nurse_cx = int((nc + 0.5) * self.cell_pixels)
        nurse_cy = int((nr + 0.5) * self.cell_pixels)
        cv2.circle(canvas, (nurse_cx, nurse_cy), 20, (60, 60, 220), -1)
        cv2.putText(canvas, "Nurse", (nurse_cx - 18, nurse_cy + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 180), 1)

        # 4. Draw Robot
        rr, rc = self.robot_pos
        robot_cx = int((rc + 0.5) * self.cell_pixels)
        robot_cy = int((rr + 0.5) * self.cell_pixels)
        cv2.circle(canvas, (robot_cx, robot_cy), 18, (220, 140, 20), -1)
        cv2.circle(canvas, (robot_cx, robot_cy), 7, (255, 255, 255), -1)
        cv2.putText(canvas, "Robot", (robot_cx - 16, robot_cy - 24), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 100, 0), 1)

        # 5. Top Header & Status on Grid
        cv2.rectangle(canvas, (0, 0), (self.width, 36), (40, 40, 40), -1)
        cv2.putText(canvas, f"Target State: {robot_state} | Action: {current_action}", 
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        return canvas


# ==========================================
# 2. SENSING (GAZE & GESTURE)
# ==========================================
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)

def estimate_gaze_direction(landmarks, w):
    """Checks iris offset relative to eye width."""
    try:
        iris_x = landmarks.landmark[468].x * w
        inner_x = landmarks.landmark[133].x * w
        outer_x = landmarks.landmark[33].x * w
        eye_w = abs(inner_x - outer_x)
        if eye_w == 0:
            return "FOCUSED_FORWARD"
        ratio = (iris_x - min(inner_x, outer_x)) / eye_w
        # Iris centered in the camera frame => Looking straight at the robot
        if 0.40 <= ratio <= 0.60:
            return "LOOKING_ROBOT"
        # Iris deflected to the side/down => Looking away / working on the patient
        else:
            return "FOCUSED_FORWARD"
    except IndexError:
        return "FOCUSED_FORWARD"

def estimate_hand_gesture(landmarks):
    """Detects open forward reaching hand."""
    wrist = landmarks.landmark[0]
    middle_tip = landmarks.landmark[12]
    index_tip = landmarks.landmark[8]
    if middle_tip.y < landmarks.landmark[9].y and index_tip.y < landmarks.landmark[5].y:
        return "HAND_REACHING"
    return "HAND_IDLE"



# ==========================================
# 4. MAIN LOOP & VISUALIZATION (ACT LAYER)
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    grid_env = HospitalGridWorld()
    brain = RobotNurseBrain()
    step = 0

    print("Running 2D Grid Visualization. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Sense ---
        gaze = "FOCUSED_FORWARD"
        gesture = "HAND_IDLE"

        face_res = face_mesh.process(rgb)
        if face_res.multi_face_landmarks:
            gaze = estimate_gaze_direction(face_res.multi_face_landmarks[0], w)

        hand_res = hands.process(rgb)
        if hand_res.multi_hand_landmarks:
            gesture = estimate_hand_gesture(hand_res.multi_hand_landmarks[0])
            mp_drawing.draw_landmarks(frame, hand_res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # --- Think ---
        brain.update_belief(gaze, gesture)
        action = brain.decide_action()
        step += 1

        # --- Act / Animate Grid ---
        grid_env.step_towards(brain.state, speed=0.18)
        grid_canvas = grid_env.render(brain.state, brain.belief, action)

        # Logging for Gate G2
        if step % 15 == 0:
            print(f"[Step {step:04d}] Belief: {brain.belief:.2f} | Robot At: {brain.state:<12} | Action: {action}")

        # --- HUD on Webcam Feed ---
        cv2.rectangle(frame, (10, 10), (320, 110), (20, 20, 20), -1)
        cv2.putText(frame, f"Gaze: {gaze}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Gesture: {gesture}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Readiness: {brain.belief * 100:.0f}%", (18, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Match heights and stack horizontally
        target_h = 390
        frame_resized = cv2.resize(frame, (int(w * (target_h / h)), target_h))
        grid_resized = cv2.resize(grid_canvas, (grid_env.width, target_h))
        combined_view = np.hstack((frame_resized, grid_resized))

        cv2.imshow("Robotic Nurse Assistant - Live 2D Grid Simulation", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()