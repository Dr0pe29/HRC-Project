import cv2
import numpy as np
import time
from brain import RobotNurseBrain

def run_synthetic_simulation(grid_env_class):
    """
    Runs an interactive synthetic fallback simulation of the Sense-Think-Act loop.
    
    Parameters:
        grid_env_class: The HospitalGridWorld class definition passed from run.py.
    """
    print("\n[INFO] Starting Synthetic / Mock Simulation Mode.")
    print("[INFO] Simulating clinical timeline across test cases. Press 'q' in the window to quit.\n")

    grid_env = grid_env_class()
    brain = RobotNurseBrain(initial_supplies=3)
    target_h = 390
    sim_w = 480

    # Scripted clinical scenario timeline:
    # (duration_seconds, gaze_state, gesture_state, scenario_phase_label)
    timeline = [
        (4.0, "FOCUSED_FORWARD", "HAND_IDLE",     "Phase 1: Nurse Focused Mid-Task on Wound Dressing"),
        (4.0, "LOOKING_ROBOT",   "HAND_IDLE",     "Phase 2: Glance at Robot (Needs Standby Supplies)"),
        (5.0, "LOOKING_ROBOT",   "HAND_REACHING", "Phase 3: Nurse Reaching (Executing Item #1 Handover)"),
        (3.0, "LOOKING_ROBOT",   "HAND_IDLE",     "Phase 4: Hand Retracted (Simulating Grasp / Placing Gauze)"),
        (5.0, "LOOKING_ROBOT",   "HAND_REACHING", "Phase 5: Re-reaching for Item #2 (Multi-Item Delivery)"),
        (2.5, "LOOKING_ROBOT",   "HAND_REACHING", "Phase 6: Reaching for Item #3..."),
        (4.0, "FOCUSED_FORWARD", "HAND_IDLE",     "Phase 7: SAFETY ABORT (Nurse Abruptly Pivots Back to Patient!)"),
        (5.0, "FOCUSED_FORWARD", "HAND_IDLE",     "Phase 8: Resumed Procedure (Robot Confirms Retreat to Bay)")
    ]

    total_sim_time = sum(d for d, _, _, _ in timeline)
    start_sim_time = time.time()
    step = 0

    while True:
        elapsed = time.time() - start_sim_time
        if elapsed >= total_sim_time:
            # Loop simulation scenario seamlessly
            start_sim_time = time.time()
            elapsed = 0.0

        # Determine current timeline phase
        t_accum = 0.0
        curr_gaze = "FOCUSED_FORWARD"
        curr_gesture = "HAND_IDLE"
        curr_phase = "Initializing"

        for dur, g, gest, phase in timeline:
            if t_accum <= elapsed < t_accum + dur:
                curr_gaze = g
                curr_gesture = gest
                curr_phase = phase
                break
            t_accum += dur

        # --- Think Layer (HMM Intent Inference & Utility Action Selection) ---
        brain.update_belief(curr_gaze, curr_gesture)
        action = brain.decide_action()
        step += 1

        # --- Act Layer (2D Grid Top-Down Animation) ---
        grid_env.step_towards(brain.state, speed=0.06)
        grid_canvas = grid_env.render(brain.state, brain.belief, action)

        if step % 20 == 0:
            print(f"[Sim Step {step:04d}] Phase: {curr_phase} | Readiness: {brain.belief*100:.0f}% | "
                  f"Pos: {brain.state:<12} | Action: {action}")

        # Render Synthetic Telemetry Dashboard (replaces camera viewport)
        hud_canvas = np.ones((target_h, sim_w, 3), dtype=np.uint8) * 30
        cv2.putText(hud_canvas, "MOCK SENSOR STREAM (NO WEBCAM)", (16, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 255), 2)
        cv2.line(hud_canvas, (16, 42), (sim_w - 16, 42), (80, 80, 80), 1)

        cv2.putText(hud_canvas, "Timeline Phase:", (16, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(hud_canvas, curr_phase, (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1)

        cv2.putText(hud_canvas, f"Synthetic Gaze:    {curr_gaze}", (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(hud_canvas, f"Synthetic Gesture: {curr_gesture}", (16, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        cv2.rectangle(hud_canvas, (16, 200), (sim_w - 16, 250), (45, 45, 45), -1)
        cv2.putText(hud_canvas, f"HMM Nurse Readiness: {brain.belief * 100:.0f}%", (26, 232), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        cv2.putText(hud_canvas, f"Robot Target:   {brain.state}", (16, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 120), 1)
        cv2.putText(hud_canvas, f"Supplies Left:  {brain.supplies_count}", (16, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1)
        cv2.putText(hud_canvas, f"Robot Status:   {brain.says[:34]}", (16, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        grid_resized = cv2.resize(grid_canvas, (grid_env.width, target_h))
        combined_view = np.hstack((hud_canvas, grid_resized))

        cv2.imshow("Robotic Nurse Assistant - 2D Grid Simulation (Fallback)", combined_view)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()