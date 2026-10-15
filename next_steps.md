# System Improvement Roadmap & Next Steps

---

## 1. Upgrade the "Think" Layer to Course Methods (Rubric Alignment)

### A. Formalize Intent Recognition with a Hidden Markov Model (HMM) or Discrete Bayesian Network (BN)

* **Current State:** The decision layer relies on a heuristic additive update (`min(1.0, belief + 0.25)`).
* **Rubric Requirement:** The Technical Soundness criterion (30 pts) explicitly requires: *"Intent-recognition block (e.g., HMM, BN) implemented and produces plausible outputs"*.


* **Improvement:**
* Define nurse intent as discrete hidden states:

$$\text{Hidden State } X_t \in \{\text{FocusedMidTask}, \text{NeedsSuppliesStandby}, \text{ReadyForHandover}\}$$


* Formulate state transitions using a transition matrix $P(X_t \mid X_{t-1})$.


* Condition transitions on sensor observations using emission probabilities:

$$P(O_t \mid X_t) = P(\text{Gaze}_t, \text{Gesture}_t \mid X_t)$$


* Formally model sensor jitter, tracking dropouts, and partial observability to fulfill Gate G2 and technical evaluation criteria.





### B. Implement a Markov Decision Process (MDP) or Cost-Utility Policy

* Replace fixed threshold rules with an explicit utility/reward function:
* **Positive Reward ($+R$):** Executing delivery when the nurse is in the `ReadyForHandover` state.
* **Disruption Penalty ($-C_{\text{interrupt}}$):** Advancing into the active workspace while the nurse is `FocusedMidTask`.
* **Movement/Oscillation Cost ($-C_{\text{motion}}$):** Penalize unnecessary transitions between staging and idle bay to prevent erratic robot behavior.



---

## 2. Physical Safety & Spatial Awareness (HRC Best Practices)

### A. Temporal Filtering & Dwell-Time Smoothing

* **Issue:** Gaze estimation from webcams is sensitive to blinks, brief micro-saccades, and temporary tracking dropouts.
* **Solution:** Introduce a sliding window or temporal buffer (e.g., $300\text{–}500\text{ ms}$ or $\approx 10\text{ frames}$). The robot should only transition states if the detected gaze or gesture persists across this threshold, avoiding jarring starts and stops.

### B. Explicit Spatial Zones (Hall's Proxemics in Healthcare)

Ground the simulated grid distances into human spatial interaction zones:

* **Idle Bay (Public Zone, $> 3.6\text{ m}$):** The robot remains parked completely outside the clinician's sterile field and primary visual cone.
* **Staging Area (Social / Consultative Zone, $1.2\text{–}2.0\text{ m}$):** The robot stays in peripheral view, close enough for quick delivery but safely outside arm-reach to avoid physical disruption.
* **Handover / Sterile Field (Personal Zone, $0.45\text{–}1.0\text{ m}$):** The robot only breaches this boundary when explicit handover intent is confirmed.

---

## 3. Mutual Legibility & Robot Transparency (Closing the HRI Loop)

### Bi-directional Communication (Robot Expressiveness)

* **Status Signaling (Visual):** Add an interface indicator or simulated LED ring reflecting current robot intent:
* **Green:** Standing by / Idle Bay.
* **Blue:** Moving to Staging Area.
* **Amber / Orange:** Aborting approach and backing away.


* **Auditory Cues (Feedback):** Include a non-intrusive notification chime when the robot reaches the staging position so the nurse knows supplies are ready without needing to take their eyes off the wound dressing.

---

## 4. Robust Sensor Fallbacks & Failure Modes

### Handling Landmark Occlusions

* **Issue:** If the nurse turns their back, drops their head too low, or walks out of frame, MediaPipe returns no landmarks.
* **Fail-Safe Mechanism:**
* Implement an explicit timeout for lost tracking.
* If tracking is lost while the robot is in `AT_NURSE`, the system must default to a safe state: immediately back off to `STAGING_AREA` or return to `IDLE_BAY` to prevent collisions.