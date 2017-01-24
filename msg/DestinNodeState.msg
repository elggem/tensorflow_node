# DestinNodeState.msg
Header header
std_msgs/String id                   # might be 'ae-e239j0-1-3-2'
std_msgs/String type                 # might be 'autoencoder'
std_msgs/String[] input_nodes        # topics f. nodes used for input (layer0 empty)
std_msgs/String[] output_nodes       # topics f. nodes that use this node's output
std_msgs/Float32MultiArray state     # describes both dimension and values