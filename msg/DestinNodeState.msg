# DestinNodeState.msg
Header header
string id                   # might be 'ae-e239j0-1-3-2'
string type                 # might be 'autoencoder'
string input_nodes          # topics f. nodes used for input (layer0 empty)
string output_nodes         # topics f. nodes that use this node's output
float32[] state              # values