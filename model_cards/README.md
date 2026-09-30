# Model cards

One card per trained model reported in the paper, written in Hugging Face model-card format, so each `README.md` can
be uploaded as-is as a model repository.

| Card | Approach | Base model (licence) | Headline result | Weights |
|---|---|---|---|---|
| [`pointing-fusion-2b`](pointing-fusion-2b/README.md) | ② pointing + controller | Cosmos-Reason2-2B ([NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)) | **SR 22.7 %** (end-SR 31.3 %, OS 42.7 %; n = 150) | not published yet |
| [`reasoner-sft-v3`](reasoner-sft-v3/README.md) | ④ Cosmos3-Edge reasoner | Cosmos3-Edge ([OpenMDW-1.1](https://openmdw.ai/license/1-1/)) | OS 47.5 % (n = 40, diagnostic) | not published yet |
| [`latentpilot-stage1-2b`](latentpilot-stage1-2b/README.md) | ① LatentPilot (negative result) | Cosmos-Reason2-2B (NVIDIA Open Model License) | OS 10.7 % vs 17.3 % for the identity control (n = 150) | not published yet |

Approach ③, the Cosmos3-Edge action model, was used zero-shot, so there is no trained model to describe.

All three were trained on data rendered from Matterport3D and are intended for **non-commercial research in
simulation**. Each is a LoRA adapter (25–33 MB) and inherits its base model's licence.
