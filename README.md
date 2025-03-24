## AdvSGM
Differentially Private Graph Learning via Adversarial Skip-gram Model

### Requirements
- python 3.6.5
- Tensorflow 1.11.0
- numpy 1.19.5
- networkx 2.5

### Functions
- data_split.py: Splits the original dataset into training and test sets.
- graph_util.py: Loads the data.
- discriminator.py: Implements the discriminator module.
- generator.py: Implements the generator module.
- exp_clip.py: Applies bounds to the exponential function and bounds the sigmoid function.
- CalcAUC.py: Evaluates the performance of node embeddings with DP in link prediction task.
- CalcMI.py: Evaluates the performance of node embeddings with DP in node clustering task.
- rdp_accountant.py: Performs RDP analysis of the sampled Gaussian mechanism.

### Test run
-run AdvSGM.py in PrivateEmb
