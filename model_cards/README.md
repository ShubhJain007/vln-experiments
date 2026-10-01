# Model cards

> **Abbreviations:** VLN = Vision-and-Language Navigation; R2R-CE = Room-to-Room in Continuous Environments; LoRA = Low-Rank Adaptation; SR = success rate (own STOP within 3 m); end-SR = ended within 3 m; OS = oracle success (ever within 3 m); SPL = Success weighted by Path Length; nDTW = normalised Dynamic Time Warping; NE = navigation error (final distance to goal); SFT = supervised fine-tuning.

One card per trained model reported in the paper, written in Hugging Face model-card format, so each `README.md` can
be uploaded as-is as a model repository.

| Card | Approach | Base model (licence) | Headline result | Weights |
|---|---|---|---|---|
| [`pointing-fusion-2b`](pointing-fusion-2b/README.md) | ② pointing + controller | Cosmos-Reason2-2B ([NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)) | **SR 23.3 %** with the STOP threshold chosen on `val_seen` (OS 36.7 %; n = 150) | not published yet |
| [`reasoner-sft-v3`](reasoner-sft-v3/README.md) | ④ Cosmos3-Edge reasoner | Cosmos3-Edge ([OpenMDW-1.1](https://openmdw.ai/license/1-1/)) | SR 3.3 %, OS 37.5 % (3 strict runs, n = 40) | not published yet |
| [`latentpilot-stage1-2b`](latentpilot-stage1-2b/README.md) | ① LatentPilot (negative result) | Cosmos-Reason2-2B (NVIDIA Open Model License) | OS 10.7 % vs 17.3 % for the identity control; held-out shortcut test 84 % vs 31 % (n = 150) | not published yet |

Approach ③, the Cosmos3-Edge action model, was used zero-shot, so there is no trained model to describe.

All three were trained on data rendered from Matterport3D and are intended for **non-commercial research in
simulation**. Each is a LoRA adapter (25–33 MB) and inherits its base model's licence.
