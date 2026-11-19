"""The LEAP hand family (get-ur-kite-dataset) for training and simulating kinematic decoders.

- ``hands``: padded hand descriptors, joint graphs and batched forward kinematics (``HandBank``)
- ``intent``: KITE's frozen intent encoder psi (``IntentEncoder``)
- ``data``: training batches for a split (``LeapSplit``)
- ``mjcf``: floating, actuated MJCFs in KITE's simulator format
"""
