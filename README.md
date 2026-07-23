# ScaleBFM

The official implementation of the paper *Scaling Behavior Foundation Model for Humanoid Robots*.

> [!IMPORTANT]
> **Release Notice**
>
> We apologize for the delay. Preparing the full release has taken longer than anticipated. Over the coming week, we will gradually release the code and resources for retargeting, training, deployment, and other related components. We expect most of the code to be available by July 26, 2026.
>
> I am personally reviewing, cleaning, and organizing the entire codebase and am working to make everything available as soon as possible. Thank you for your patience and understanding.
>
> Please check this repository regularly for the latest releases and updates. If you encounter any problems or have any questions, feel free to open an issue.
>
> — Weishuai Zeng

## ScaleRetarget

**ScaleRetarget** is the motion-retargeting toolkit used to prepare human motion
data for ScaleBFM. It converts motion from common mocap representations into
robot joint trajectories and currently provides a complete configuration for the
Unitree G1 humanoid.


The retargeting code and documentation are available in
**[ScaleRetarget](ScaleRetarget/README.md)**. See the guide for supported datasets,
environment setup, motion retargeting, dataset preparation, and visualization.

<p align="center">
  <img src="ScaleRetarget/assets/demos/amass.gif" alt="AMASS motion retargeted to Unitree G1" width="32%">
  <img src="ScaleRetarget/assets/demos/finedance.gif" alt="FineDance motion retargeted to Unitree G1" width="32%">
  <img src="ScaleRetarget/assets/demos/mixamo.gif" alt="Mixamo motion retargeted to Unitree G1" width="32%">
</p>

## ScaleTrack

**ScaleTrack** provides the foundational implementation of the BFM pretraining
pipeline. It packages retargeted robot trajectories into motion datasets and
pretrains BFMs through motion tracking in IsaacLab using the bundled RSL-RL
implementation. It currently provides a complete pretraining pipeline for the
Unitree G1 humanoid, including single- and multi-GPU training, policy playback, and export.

The training code and documentation are available in
**[ScaleTrack](ScaleTrack/README.md)**. See the guide for environment and robot
asset setup, motion preparation and packaging, policy training, playback, and
export.

<p align="center">
  <img src="ScaleTrack/assets/demos/example_1.gif" alt="ScaleTrack motion-tracking demo 1" width="32%">
  <img src="ScaleTrack/assets/demos/example_2.gif" alt="ScaleTrack motion-tracking demo 2" width="32%">
  <img src="ScaleTrack/assets/demos/example_3.gif" alt="ScaleTrack motion-tracking demo 3" width="32%">
</p>
