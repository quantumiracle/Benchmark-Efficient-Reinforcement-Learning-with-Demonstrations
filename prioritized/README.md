# Test the Prioritized Experience Replay

Try to figure out if there is an alternative approach for prioritized experience replay: instead of labelling the trained samples with TD-error (which is also the loss of the critic network) and training with priority next time, why not directly update the those samples with large TD-error to be trained multiple times on current training process, until it has a small loss.  This is a test file on supervised learning, to testify if it is useful with loss sensitive training process.

* `data_ini2goal.p`: training data for reacher env mapping from arbitrary initial positions to arbitrary goal positions.
* `predict_test_prioritized.py`: supervised training with loss sensitivity.