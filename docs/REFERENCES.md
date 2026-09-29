# References

All 53 works cited by the paper ([`paper/main.pdf`](../paper/main.pdf)) and used by this repository. BibTeX for every
entry is in [`paper/references.bib`](../paper/references.bib); arXiv entries were exported from arxiv.org.

## Methods this project reimplements or builds on

- Haihong Hao et al. (2026). **LatentPilot: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning**. [arxiv.org/abs/2603.29165](https://arxiv.org/abs/2603.29165) — `latentpilot`
- Abdelaziz Bounhar et al. (2026). **Robostral Navigate**. [arxiv.org/abs/2607.20785](https://arxiv.org/abs/2607.20785) — `robostral`
- Yiming Qin et al. (2026). **Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens**. [arxiv.org/abs/2511.19418](https://arxiv.org/abs/2511.19418) — `covt`

## Task, data, simulator and metrics

- Peter Anderson et al. (2018). **Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments**. [arxiv.org/abs/1711.07280](https://arxiv.org/abs/1711.07280) — `anderson2018r2r`
- Jacob Krantz et al. (2020). **Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments**. [arxiv.org/abs/2004.02857](https://arxiv.org/abs/2004.02857) — `krantz2020vlnce`
- Jacob Krantz et al. (2020). **VLN-CE: Vision-and-Language Navigation in Continuous Environments (code and R2R-CE data)**. [github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE) — `vlnce2020code`
- Manolis Savva et al. (2019). **Habitat: A Platform for Embodied AI Research**. [arxiv.org/abs/1904.01201](https://arxiv.org/abs/1904.01201) — `savva2019habitat`
- Andrew Szot et al. (2022). **Habitat 2.0: Training Home Assistants to Rearrange their Habitat**. [arxiv.org/abs/2106.14405](https://arxiv.org/abs/2106.14405) — `szot2021habitat2`
- Angel Chang et al. (2017). **Matterport3D: Learning from RGB-D Data in Indoor Environments**. [arxiv.org/abs/1709.06158](https://arxiv.org/abs/1709.06158) — `chang2017mp3d`
- Peter Anderson et al. (2018). **On Evaluation of Embodied Navigation Agents**. [arxiv.org/abs/1807.06757](https://arxiv.org/abs/1807.06757) — `anderson2018evaluation`
- Gabriel Ilharco et al. (2019). **General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping**. [arxiv.org/abs/1907.05446](https://arxiv.org/abs/1907.05446) — `ilharco2019ndtw`

## Models used

- NVIDIA (2025). **Cosmos-Reason2-2B**. [huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) — `nvidia2025cosmosreason2`
- NVIDIA (2026). **Cosmos3-Edge**. [huggingface.co/nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) — `nvidia2026cosmos3edge`
- NVIDIA et al. (2025). **Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning**. [arxiv.org/abs/2503.15558](https://arxiv.org/abs/2503.15558) — `nvidia2025cosmosreason1`
- NVIDIA et al. (2025). **Cosmos World Foundation Model Platform for Physical AI**. [arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575) — `nvidia2025cosmoswfm`
- Peng Wang et al. (2024). **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution**. [arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191) — `wang2024qwen2vl`
- Shuai Bai et al. (2025). **Qwen2.5-VL Technical Report**. [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923) — `bai2025qwen25vl`
- Xiaohua Zhai et al. (2023). **Sigmoid Loss for Language Image Pre-Training**. [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343) — `zhai2023siglip`
- Michael Tschannen et al. (2025). **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features**. [arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786) — `tschannen2025siglip2`
- Yuanhan Zhang et al. (2025). **LLaVA-Video: Video Instruction Tuning With Synthetic Data**. [arxiv.org/abs/2410.02713](https://arxiv.org/abs/2410.02713) — `zhang2024llavavideo`

## Training and inference techniques

- Edward J. Hu et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) — `hu2021lora`
- Jianlin Su et al. (2023). **RoFormer: Enhanced Transformer with Rotary Position Embedding**. [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) — `su2021roformer`
- Jonathan Ho and Tim Salimans (2022). **Classifier-Free Diffusion Guidance**. [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598) — `ho2022cfg`
- Yaron Lipman et al. (2023). **Flow Matching for Generative Modeling**. [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747) — `lipman2023flow`
- Cheng Chi et al. (2024). **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**. [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137) — `chi2023diffusionpolicy`
- Ilya Loshchilov and Frank Hutter (2019). **Decoupled Weight Decay Regularization**. [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101) — `loshchilov2019adamw`
- Tim Dettmers et al. (2022). **8-bit Optimizers via Block-wise Quantization**. [arxiv.org/abs/2110.02861](https://arxiv.org/abs/2110.02861) — `dettmers2022optim8bit`
- Jason Wei et al. (2023). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903) — `wei2022cot`
- Guillaume Alain and Yoshua Bengio (2018). **Understanding intermediate layers using linear classifier probes**. [arxiv.org/abs/1610.01644](https://arxiv.org/abs/1610.01644) — `alain2016probes`

## Loss balancing and exposure bias

- Zhao Chen et al. (2018). **GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**. [arxiv.org/abs/1711.02257](https://arxiv.org/abs/1711.02257) — `chen2018gradnorm`
- Alex Kendall et al. (2018). **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**. [arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115) — `kendall2018multitask`
- Samy Bengio et al. (2015). **Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks**. [arxiv.org/abs/1506.03099](https://arxiv.org/abs/1506.03099) — `bengio2015scheduled`
- Marc'Aurelio Ranzato et al. (2016). **Sequence Level Training with Recurrent Neural Networks**. [arxiv.org/abs/1511.06732](https://arxiv.org/abs/1511.06732) — `ranzato2016sequence`
- Stephane Ross et al. (2011). **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**. [arxiv.org/abs/1011.0686](https://arxiv.org/abs/1011.0686) — `ross2011dagger`

## Related VLN and embodied-AI work

- Daniel Fried et al. (2018). **Speaker-Follower Models for Vision-and-Language Navigation**. [arxiv.org/abs/1806.02724](https://arxiv.org/abs/1806.02724) — `fried2018speaker`
- Yicong Hong et al. (2021). **A Recurrent Vision-and-Language BERT for Navigation**. [arxiv.org/abs/2011.13922](https://arxiv.org/abs/2011.13922) — `hong2021vlnbert`
- Shizhe Chen et al. (2023). **History Aware Multimodal Transformer for Vision-and-Language Navigation**. [arxiv.org/abs/2110.13309](https://arxiv.org/abs/2110.13309) — `chen2021hamt`
- Yicong Hong et al. (2022). **Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation**. [arxiv.org/abs/2203.02764](https://arxiv.org/abs/2203.02764) — `hong2022bridging`
- Dong An et al. (2024). **ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**. [arxiv.org/abs/2304.03047](https://arxiv.org/abs/2304.03047) — `an2023etpnav`
- Jiazhao Zhang et al. (2024). **NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**. [arxiv.org/abs/2402.15852](https://arxiv.org/abs/2402.15852) — `zhang2024navid`
- Jiazhao Zhang et al. (2025). **Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks**. [arxiv.org/abs/2412.06224](https://arxiv.org/abs/2412.06224) — `zhang2024uninavid`
- An-Chieh Cheng et al. (2025). **NaVILA: Legged Robot Vision-Language-Action Model for Navigation**. [arxiv.org/abs/2412.04453](https://arxiv.org/abs/2412.04453) — `cheng2024navila`
- Anthony Brohan et al. (2023). **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**. [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) — `brohan2023rt2`
- Moo Jin Kim et al. (2024). **OpenVLA: An Open-Source Vision-Language-Action Model**. [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246) — `kim2024openvla`
- Wentao Yuan et al. (2024). **RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**. [arxiv.org/abs/2406.10721](https://arxiv.org/abs/2406.10721) — `yuan2024robopoint`
- Matt Deitke et al. (2024). **Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models**. [arxiv.org/abs/2409.17146](https://arxiv.org/abs/2409.17146) — `deitke2024molmo`
- David Ha and Jürgen Schmidhuber (2018). **World Models**. [arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122) — `ha2018worldmodels`
- Danijar Hafner et al. (2024). **Mastering Diverse Domains through World Models**. [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104) — `hafner2023dreamerv3`
- Adrien Bardes et al. (2024). **Revisiting Feature Prediction for Learning Visual Representations from Video**. [arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471) — `bardes2024vjepa`

## Software

- Adam Paszke et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library**. [arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703) — `paszke2019pytorch`
- Thomas Wolf et al. (2020). **HuggingFace's Transformers: State-of-the-art Natural Language Processing**. [arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771) — `wolf2020transformers`
- Sourab Mangrulkar et al. (2022). **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods**. [github.com/huggingface/peft](https://github.com/huggingface/peft) — `mangrulkar2022peft`
- Patrick von Platen et al. (2022). **Diffusers: State-of-the-art diffusion models**. [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) — `vonplaten2022diffusers`
