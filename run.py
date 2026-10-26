import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np

# Conditional MediaPipe import
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from brain import RobotNurseBrain
from simulation import run_synthetic_simulation

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

    def step_towards(self, target_state_name, speed=0.06):
        """Moves robot gradually towards the target waypoint on the grid."""
        target_r, target_c = self.locations[target_state_name]
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
if HAS_MEDIAPIPE:
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    hands = mp_hands.Hands(max_num_hands=1)

def estimate_gaze_direction(landmarks, w=None):
    try:
        r_iris = landmarks.landmark[468].x
        r_inner = landmarks.landmark[133].x
        r_outer = landmarks.landmark[33].x
        r_w = abs(r_inner - r_outer)

        l_iris = landmarks.landmark[473].x
        l_inner = landmarks.landmark[362].x
        l_outer = landmarks.landmark[263].x
        l_w = abs(l_inner - l_outer)

        if r_w == 0 or l_w == 0:
            return "FOCUSED_FORWARD"

        r_ratio = (r_iris - min(r_inner, r_outer)) / r_w
        l_ratio = (l_iris - min(l_inner, l_outer)) / l_w
        avg_ratio = (r_ratio + l_ratio) / 2.0

        if 0.38 <= avg_ratio <= 0.62:
            return "LOOKING_ROBOT"
        return "FOCUSED_FORWARD"
    except (IndexError, AttributeError):
        return "FOCUSED_FORWARD"

def is_finger_open(landmarks, tip_idx, knuckle_idx, wrist_idx=0):
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    return np.linalg.norm(tip - wrist) > (np.linalg.norm(knuckle - wrist) * 1.15)

def estimate_hand_gesture(landmarks):
    try:
        index_open = is_finger_open(landmarks, 8, 6)
        middle_open = is_finger_open(landmarks, 12, 10)
        ring_open = is_finger_open(landmarks, 16, 14)

        wrist_z = landmarks.landmark[0].z
        middle_z = landmarks.landmark[12].z
        reaching_forward = (middle_z - wrist_z) < 0.05

        if (index_open and middle_open and ring_open) and reaching_forward:
            return "HAND_REACHING"
        return "HAND_IDLE"
    except (IndexError, AttributeError):
        return "HAND_IDLE"


# ==========================================
# 3. MAIN RUNNER (ROBUST GATE G1/G2 ENTRY)
# ==========================================
def main():
    # Allow manual override via command line (e.g., python run.py --mock)
    force_mock = "--mock" in sys.argv

    cap = None
    webcam_available = False

    if HAS_MEDIAPIPE and not force_mock:
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                webcam_available = True

    # If no camera is available or --mock is specified, trigger fallback
    if not webcam_available:
        if cap is not None:
            cap.release()
        print("[INFO] Camera not detected (or --mock passed). Launching fallback simulation...")
        run_synthetic_simulation(HospitalGridWorld)
        return

    # Real-time computer vision mode
    grid_env = HospitalGridWorld()
    brain = RobotNurseBrain()
    step = 0

    print("[INFO] Webcam initialized. Running Live 2D Grid Visualization. Press 'q' to quit.")

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
        grid_env.step_towards(brain.state, speed=0.06)
        grid_canvas = grid_env.render(brain.state, brain.belief, action)

        if step % 15 == 0:
            print(f"[Step {step:04d}] Belief: {brain.belief:.2f} | Robot At: {brain.state:<12} | Action: {action}")

        # --- HUD on Webcam Feed ---
        cv2.rectangle(frame, (10, 10), (320, 110), (20, 20, 20), -1)
        cv2.putText(frame, f"Gaze: {gaze}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Gesture: {gesture}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Readiness: {brain.belief * 100:.0f}%", (18, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

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