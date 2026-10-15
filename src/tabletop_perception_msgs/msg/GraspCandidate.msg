# One antipodal grasp for a parallel-jaw gripper.
# tip_pose: gripper tip frame (fingers close along its x axis, approach along its +z axis -
#           i.e. +z points from the palm toward the object), in tip_pose.header.frame_id.
geometry_msgs/PoseStamped tip_pose
float64 width        # m, distance between the two contact points
float64 score        # higher is better; only meaningful relative to other candidates
geometry_msgs/Vector3 approach   # unit approach direction (= tip_pose +z), for convenience
