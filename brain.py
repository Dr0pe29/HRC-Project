
import time
import numpy as np
from hmmlearn import hmm

MODE = "proactive"
DEBUG = False
STEP_TIME = 0.5 #CHanged to get faster updates

# ------------- THINK LAYER ----------------

# Hidden States X_t:
# 0: HighFocus (Nurse busy)
# 1: TaskTransition (Nurse glanced up, needs supplies ready on standby)
# 2: AwaitingHandover (Nurse looking and reaching for supply)
states = ["HighFocus", "TaskTransition", "AwaitingHandover"]

# Observations:
# Based on Looking=yes/no and Gesture=none/reach
# 0: Looking=no, Gesture=none
# 1: Looking=no, Gesture=reach
# 2: Looking=yes, Gesture=none
# 3: Looking=yes, Gesture=reach
def obs_index(looking, gesture):
    idx = 0
    if looking == "yes":
        idx += 2
    if gesture == "reach":
        idx += 1
    return idx

start_probs = np.array([0.7, 0.2, 0.1])

trans_mat = np.array([
    [0.85, 0.12, 0.03],  
    [0.15, 0.70, 0.15], 
    [0.10, 0.15, 0.75],  
])

emission_mat = np.array([
    [0.85, 0.05, 0.08, 0.02],  
    [0.15, 0.05, 0.70, 0.10],  
    [0.02, 0.03, 0.20, 0.75],  
])
    
# 1. Number of hidden states
n_states = 3  # [HighFocus, TaskTransition, AwaitingHandover]

# 2. Number of possible observation symbols
n_observations = 4  # (Looking, NotLooking) x (Gesture, NoGesture)

# 3. Define HMM model
model = hmm.CategoricalHMM(n_components=n_states, init_params="", params="")
model.startprob_ = start_probs       # initial distribution
model.transmat_  = trans_mat
model.emissionprob_ = emission_mat

print("start_probs:", np.sum(start_probs))
print("trans_mat rows:", np.sum(trans_mat, axis=1))
print("emission_mat rows:", np.sum(emission_mat, axis=1))

def get_belief(obs_list, window=8): #Added window to limit history inertia
    recent_obs = obs_list[-window:]
    X = np.array(recent_obs).reshape(-1, 1)
    posteriors = model.predict_proba(X)
    belief = posteriors[-1]
    if DEBUG:
        print(" [HMM] last obs:", obs_list[-1], "| belief:",belief)
    return belief

# ------------- ACT LAYER ----------------

positions = ["IDLE_BAY", "STAGING_AREA", "AT_NURSE"]
actions = ["stay", "closer", "back"]

def move(pos, action):
    if action == 1: 
        return min(2, pos + 1)
    if action == 2:
        return max(0, pos - 1)
    return pos


reward_table = np.array([
    [ 2.0, -1.0, -5.0],  # IDLE_BAY: Good when nurse is busy; bad when awaiting
    [-1.0,  3.5,  0.5],  # STAGING_AREA: Ideal for TaskTransition (proactive standby)
    [-8.0, -2.0,  5.0],  # AT_NURSE: High reward for handover, heavy penalty if busy!
])
motion_cost = 0.3 #Reduced to make it more responsive

def choose_action(belief, pos):
    best_action = 0
    best_score = -1000
    for action in range(3):
        new_pos = move(pos, action)
        score = 0
        for n in range(3):
            score = score + belief[n] * reward_table[new_pos][n]
        if new_pos != pos:
            score = score - motion_cost
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


class RobotNurseBrain:
    def __init__(self, initial_supplies=3):
        self.start_time = time.time()
        self.last_step = 0.0
        
        self.pos = 0  # 0: IDLE_BAY, 1: STAGING_AREA, 2: AT_NURSE
        self.state = positions[self.pos]
        self.belief = 0.0
        self.says = "Waiting in idle bay"
        
        # Multi-item inventory & handover tracking
        self.supplies_count = initial_supplies
        self.item_delivered = False
        self.handover_start_time = None
        self.HANDOVER_DURATION = 1.5  # Transfer duration in seconds
        self.max_supplies = initial_supplies
        self.restocking_start_time = None
        self.RESTOCK_DURATION = 3.0
        
        self.recent_reaches = []
        self.recent_gazes = []
        self.obs_list = [0]

        with open("log.txt", "w") as f:
            f.write("time,mode,looking,gesture,obs,P_HighFocus,P_TaskTransition,P_AwaitingHandover,robot_position,supplies_left,robot_says\n")

    def update_belief(self, gaze, gesture):
        now = time.time() - self.start_time

        self.recent_gazes.append(gaze == "LOOKING_ROBOT")
        self.recent_reaches.append(gesture == "HAND_REACHING")

        if now - self.last_step < STEP_TIME:
            return
        self.last_step = now

        # Smooth detection over recent frames to avoid flickering
        is_looking = sum(self.recent_gazes) > (len(self.recent_gazes) * 0.5)
        is_reaching = sum(self.recent_reaches) > (len(self.recent_reaches) * 0.5)
        self.recent_gazes = []
        self.recent_reaches = []

        looking_word = "yes" if is_looking else "no"
        gesture_word = "reach" if is_reaching else "none"

        obs = obs_index(looking_word, gesture_word)
        self.obs_list.append(obs)

        # Update HMM belief
        belief = get_belief(self.obs_list)
        self.belief = belief[2]
        nurse_state = np.argmax(belief)  # 0: HighFocus, 1: TaskTransition, 2: AwaitingHandover


        # ------------------------ HANDOVER LOGIC ------------------------
        #MAYBE THIS SHOULD BE TURNED INTO THE HMM AND MDP LOGIC, BUT FOR NOW IT'S A SIMPLE RULE-BASED SYSTEM
        
        # OUT OF SUPPLIES: Must return to Bay to restock
        if self.supplies_count == 0:
            self.pos = 0  # Force to IDLE_BAY
            self.state = positions[self.pos]
            self.handover_start_time = None

            if self.restocking_start_time is None:
                self.restocking_start_time = now
                self.says = "Tray empty: restocking supplies in Bay..."
            elif now - self.restocking_start_time >= self.RESTOCK_DURATION:
                # Restocking finished: refill tray and clear timer
                self.supplies_count = self.max_supplies
                self.restocking_start_time = None
                self.item_delivered = False
                self.says = f"Supplies restocked ({self.max_supplies} items ready)! Waiting in idle bay"
            else:
                self.says = f"Restocking supplies in Bay ({int(self.RESTOCK_DURATION - (now - self.restocking_start_time))}s left)..."

            # Log step while restocking and exit early
            with open("log.txt", "a") as f:
                f.write(f"{now:.2f},{MODE},{looking_word},{gesture_word},{obs},"
                        f"{belief[0]:.3f},{belief[1]:.3f},{belief[2]:.3f},"
                        f"{self.state},{self.supplies_count},{self.says}\n")
            return

        # If the nurse either looks down OR simply retracts their hand, reset handover timer.
        if self.item_delivered and (nurse_state == 0 or gesture_word == "none"):
            self.item_delivered = False
            self.handover_start_time = None

        # After handing over, hold at STAGING_AREA until the nurse retracts their hand
        if self.item_delivered:
            self.pos = 1  # STAGING_AREA
            self.handover_start_time = None
            says = f"Item delivered! Standing by at staging ({self.supplies_count} items remaining)"

        #HANDOVER EXECUTION:
        elif self.pos == 2 and nurse_state == 2:
            if self.handover_start_time is None:
                self.handover_start_time = now
                says = f"Transferring item #{4 - self.supplies_count} to nurse"
            elif now - self.handover_start_time >= self.HANDOVER_DURATION:
                # Delivery complete
                self.supplies_count -= 1
                self.item_delivered = True
                self.pos = 1
                self.handover_start_time = None
                says = f"Handover #{4 - (self.supplies_count + 1)} complete! Backing off to staging"
            else:
                says = "Handing over supplies to nurse"

        # -------------------- END OF HANDOVER --------------------

        # Otherwise, decide action based on belief and position
        else:
            self.handover_start_time = None  # Reset timer on any abort/retreat
            action = choose_action(belief, self.pos)
            new_pos = move(self.pos, action)

            if new_pos == self.pos:
                if self.pos == 0:
                    says = "Waiting in idle bay"
                elif self.pos == 1:
                    says = f"Standing by at staging threshold ({self.supplies_count} items ready)"
                else:
                    says = "Waiting at nurse"
            elif new_pos > self.pos:
                if new_pos == 1:
                    says = "Advancing to staging threshold"
                else:
                    says = "Advancing to nurse for handover"
            else:
                if new_pos == 1:
                    says = "Aborting: backing off to staging"
                else:
                    says = "Nurse focused: returning to bay"

            self.pos = new_pos

        self.state = positions[self.pos]
        self.says = says

        # Log entry (Gate G2)
        with open("log.txt", "a") as f:
            f.write(f"{now:.2f},{MODE},{looking_word},{gesture_word},{obs},"
                    f"{belief[0]:.3f},{belief[1]:.3f},{belief[2]:.3f},"
                    f"{self.state},{self.supplies_count},{self.says}\n")

    def decide_action(self):
        return self.says