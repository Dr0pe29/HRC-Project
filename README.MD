# Proactive Robotic Nurse Assistant (HRC)

A Human-Robot Collaboration (HRC) simulation featuring real-time non-verbal sensing (gaze and hand tracking via MediaPipe) paired with an intent-aware decision pipeline (Hidden Markov Model and 1-step Expected Utility Policy). The robot proactively stages and delivers sterile wound care supplies without disrupting clinical focus.

The system supports two experimental conditions:
 
| Condition | Mode | Robot trigger | Home position |
| :--- | :--- | :--- | :--- |
| **A: Reactive** | `--reactive` | Nurse presses the **spacebar** (button). The HMM is ignored. | `IDLE_BAY` |
| **B: Proactive** | default | HMM belief over gaze and gesture (anticipates the need). | `STAGING_AREA` |

---

## 1. Setup

### Automated Setup (Windows PowerShell)
```powershell
.\setup_env.ps1
```
*(If PowerShell blocks execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Manual Setup

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Running the System

Ensure your virtual environment is active, then run:

```bash
python run.py              # Condition B: proactive (HMM-driven)
python run.py --reactive   # Condition A: reactive (button-driven)
```

* **With a Webcam:** The system launches the live dual-pane view: webcam skeletal tracking on the left and the 2D top-down hospital ward simulation on the right.
* **Without a Webcam (Automatic Fallback):** If no camera is detected, the program automatically switches to a synthetic simulation demonstrating all clinical scenarios.
* **Force Mock Mode:** You can explicitly run the camera-free synthetic timeline at any time using:
  ```bash
  python run.py --mock
  ```
* **Exit:** Press `q` while focused on the simulation window.

---

## 3. How to Interact (Live Mode)

| User Action | System Perception | Robot Behavior |
| :--- | :--- | :--- |
| **Focused Mid-Task** (Look away / down at table) | Gaze: `FOCUSED_FORWARD`, Gesture: `HAND_IDLE` | Nurse is busy (`HighFocus`). Robot remains parked in `IDLE_BAY` out of the way. |
| **Glance at Robot** (Look toward camera, hands down) | Gaze: `LOOKING_ROBOT`, Gesture: `HAND_IDLE` | Standby cue (`TaskTransition`). Robot advances to `STAGING_AREA` and holds without intruding. |
| **Reach for Handover** (Look toward camera + open hand) | Gaze: `LOOKING_ROBOT`, Gesture: `HAND_REACHING` | Delivery request (`AwaitingHandover`). Robot advances to `AT_NURSE`, executes item transfer, and steps back to staging. |
| **Safety Abort Test** (Reach, then abruptly look away) | Shift from `AwaitingHandover` to `HighFocus` | Sterile field protection. Expected utility drops immediately, forcing the robot to abort and retreat. |

---

## 4. File Structure

* `run.py` – Main execution script, webcam sensing pipeline, and dual-pane OpenCV visualization.
* `brain.py` – The Think layer: Categorical HMM for intent estimation, Expected Utility action selection, and handover/inventory tracking.
* `mock_simulation.py` – Headless/camera-free fallback runner executing scripted clinical stress test phases.
* `requirements.txt` – Pinned library dependencies (`opencv-python`, `mediapipe`, `numpy`, `hmmlearn`).
* `log.txt` – Generated runtime log tracking time-stamped beliefs, positions, and dialogue every step (satisfies Gate G2).