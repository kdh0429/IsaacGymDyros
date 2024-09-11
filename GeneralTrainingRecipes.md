# 1. **Start with the simplest task and simple task reward:**
### **Network capacity:**
   - Generally not a significant issue.
   - Typically, 2-3 layers with 256 or 512 units are used.
### **MDP-related considerations:**
   - Reward tuning: If the reward is not well-tuned.
      - Early termination is a huge reward(penalty) shaping → Early termination is a way to impose (soft) constraint, and proper early termination leads to fast convergence
   - State selection: Some states can dramatically increase the training speed, but it is important to consider whether they are suitable for sim-to-real transfer.
   - Exploration issues: If exploration is insufficient and the model is stuck in a local minimum, adjust the action distribution standard deviation.
### **Learning parameter tuning:**
   - Important parameters:
       - Actor and critic learning rate: Reduce if the loss is large. Increase the actor learning rate if the KL or clip fraction is too small.
       - Horizon length.
       - Training iterations.

# 2. **Adjustments for real robot usage:**
### **Add regularization terms:** 
   - Commonly used terms include qvel, qacc, torque, torque diff, and contact force.
      - Tip: Observe the robot’s motion frequently! (Ideally, use a real-time simulator) → Visualize the policy, analyze data, and develop an intuition for the magnitude of data that results in aggressive movements → e.g., contact force < 1400N.
### **Tuning:**
   - Ensure these terms do not interfere with task learning.

# 3. **Sim-to-Real or Sim-to-Sim Transfer:**
### **Pre-transfer validation:** 
  - **ENSURE THAT THE PERFORMANCE IN THE TRAINED ENVIRONMENT IS GOOD BEFORE SIM-TO-REAL OR SIM-TO-SIM TRANSFER.**
  - **Testing environment:** Test in the nominal environment without randomization (e.g., noise, bias, dynamics randomization) in the trained simulator.
  - **Key checks:**
      - Ensure the robot does not deviate significantly when given a 0 velocity command.
      - Minimal drift to lateral direction when given a forward velocity command.
      - Contact force is not excessive.
      - The robot does not fall over time, especially for tasks requiring prolonged standing.
### **Strategies to reduce the sim-to-real/sim-to-sim gap:**
  - **Option 1:** Tune the reward to encourage motions favorable for sim-to-real transfer (e.g., flat foot, smooth contact force) $\approx$ regularization term tuning in step 2.
  - **Option 2:** Implement Sim-to-Real techniques (e.g., adding robustness, though this may reduce optimal performance. When increasing randomization, first verify performance in the trained environment as mentioned in [Pre-transfer validation](#pre-transfer-validation)).
  - **Option 3:** System identification - Adjust simulation parameters to closely match the real robot (most challenging).
      - **For Sim-to-Sim transfer:** It is important to understand the differences between simulators, especially with additional elements beyond those defined in the XML/URDF models (e.g., tendon, collision mesh, body friction, armature, damping).

Please contact kdh0429@snu.ac.kr(Donghyeon Kim) or any other seniors in the lab if you are stuck in training.
