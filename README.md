# An Anatomy of Vision-Language-Action Models  
### From Modules to Milestones and Challenges  

[![Paper](https://img.shields.io/badge/arXiv-2512.11362-b31b1b.svg)](https://arxiv.org/abs/2512.11362) [![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://suyuz1.github.io/VLA-Survey-Anatomy/)

This repository accompanies the survey paper:

> **An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges**  
> *Chao Xu, Suyu Zhang, Yang Liu, Baigui Sun, et al.*

We organize the rapidly growing Vision-Language-Action (VLA) literature **around core challenges**, rather than architectures or tasks, following the structure of **Section 4** of the paper.

---

## 🔍 What This Repository Provides

- A **challenge-centric taxonomy** of VLA research
- Fine-grained **sub-challenge decomposition** with representative papers
- A **living, continuously updated** survey website

---

## 🧭 Survey Roadmap

This survey follows a natural learning trajectory:

1. **Basic Modules** – What components make up a VLA system?
2. **Milestones & Evolution** – How did VLA models evolve over time?
3. **Core Challenges** – What are the fundamental bottlenecks today?
4. **Applications** – Where are VLA models deployed?

> **This repository focuses primarily on (3) Core Challenges.**

---

## 🚧 Core Challenges in Vision-Language-Action Models

We identify **five fundamental challenges**, each further decomposed into **sub-challenges**.  

![image](https://github.com/SuyuZ1/Survery/blob/main/imgs/challenges.png)

---

> ⭐Starred papers are submitted by authors who contacted us. We warmly welcome submissions and encourage researchers to share their latest results.

### 1️⃣ Multi-Modal Alignment and Physical World Modeling  
*(Sec. 4.1)*

<details>
<summary><strong>4.1.1 The GAP between Semantics Perception and Physical Interaction</strong></summary>

- **PhysBrain**[2025-12-19]: Train an egocentric-aware embodied brain on the E2E-3M dataset to bridge vision language models with physical intelligence and enable sample-efficient VLA fine-tuning. [[paper](https://arxiv.org/pdf/2512.16793)] [[website](https://zgc-embodyai.github.io/PhysBrain/)]
- **DexGrasp-VLA**[2025-12-13]: Develop an Arm-Hand Feature Enhancement module to explicitly capture distinct latent features of arm and hand movements. [[paper](https://arxiv.org/pdf/2511.00139)] [[website](https://dexvla-seed.github.io/dex-vla)]
- **InternVLA-M1⭐**[2025-10-15]: Employs a two-stage pipeline combining spatial grounding pre-training with spatially guided action post-training to bridge instructions and actions. [[paper](https://arxiv.org/pdf/2510.13778)] [[website](https://internrobotics.github.io/internvla-m1.github.io/)]
- **WALL-OSS⭐**[2025-09-08]: Employ a tightly coupled Mixture-of-Experts architecture with static routing to align action and vision-language features. [[paper](https://arxiv.org/pdf/2509.11766)] [[website](https://x2robot.com/en/research/68bc2cde8497d7f238dde690)]
- **ACT-LLM**[2025]: Formulates raw per- ceptual inputs into structured language representation. [[paper](https://arxiv.org/pdf/2506.21250)] [[website](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **Humanoid-VLA**[2025]: Pre-training language-action. [[paper](https://arxiv.org/pdf/2502.14795)] [[website](https://github.com/AllenXuuu/HumanVLA)]
- **Orion**[2025]: Introduces a high-level VLM planner with a separate low-level motion controller. [[paper](https://arxiv.org/pdf/2503.19755)] [[website](https://xiaomi-mlab.github.io/Orion/)]
- **Gemini RObotics**[2025]: Introduces a high-level VLM planner with a separate low-level motion controller. [[paper](https://arxiv.org/pdf/2503.20020)] [[website](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)]
- **KnowledgeVLA**[2025]: Introduces a high-level VLM planner with a separate low-level motion controller. [[paper](https://arxiv.org/abs/2505.23705)] [[website](https://www.pi.website/research/knowledge_insulation)]
- **Beyond Sight**[2025]: Incorporating additional modalities. [[paper](https://arxiv.org/pdf/2501.04693)] [[website](https://fuse-model.github.io/)]
- **TouchVLA**[2025]: Incorporating additional modalities. [[paper](https://arxiv.org/pdf/2507.17294)] [[website](https://clear-nus.github.io/blog/vla-touch)]
- **TLA**[2025]: Introduces tactile perception. [[paper](https://arxiv.org/pdf/2503.08548)] [[website](https://sites.google.com/view/tactile-language-action/)]
- **OmniVTLA**[2025]: Constructs a se mantically aligned tactile encoder. [[paper](https://arxiv.org/pdf/2508.08706)] [[website](https://github.com/linchangyi1/Awesome-Touch)]
- **Tactile-VLA**[2025]: Ranging from deep fusion across the full pipeline. [[paper](https://arxiv.org/pdf/2507.09160)] [[website](https://jialeihuang.github.io/tactileVLA.github.io/)]
- **ForceVLA**[2025]: Modular mixture-of-experts (MoE) fusion. [[paper](https://arxiv.org/pdf/2505.22159)] [[website](https://sites.google.com/view/forcevla2025/)]
- **MultiGen**[2025]: Uses multimodal generation for simulated multi-modal data. [[paper](https://arxiv.org/html/2507.02864v2)] [[website](https://multigen-audio.github.io/)]
- **Grounding MLLMs**[2024]: Fine-tuning a pretrained VLM to directly output action tokens. [[paper](https://arxiv.org/pdf/2406.07904)] [[website](https://github.com/mbzuai-oryx/groundingLMM)]
- **OpenVLA**[2024]: Fine-tuning a pretrained VLM to directly output action tokens. [[paper](https://arxiv.org/pdf/2406.09246)] [[website](https://openvla.github.io/)]
- **CLIP-RT**[2024]: Extends CLIP-style vision-language alignment. [[paper](https://arxiv.org/pdf/2411.00508)] [[website](https://clip-rt.github.io)]
- **LIV**[2023]: Introdues a contrastive framework on robot-control data to construct a joint vision-language embedding space. [[paper](https://proceedings.mlr.press/v202/ma23b/ma23b.pdf)] [[website](https://penn-pal-lab.github.io/LIV/)]
- **Look-Leap**[2023]: Structured action-plan generation from visual inputs. [[paper](https://arxiv.org/pdf/2311.17842)] [[website](robot-vila.github.io)]
- **RT-2**[2023]: Fine-tuning a pretrained VLM to directly output action tokens. [[paper](https://arxiv.org/abs/2307.15818)] [[website](https://robotics-transformer.github.io/)]
- **Prompt-a-Robot-to-Walk**[2023]: Fine-tuning a pretrained VLM to directly output action tokens. [[paper](https://arxiv.org/abs/2309.09969)] [[website](https://prompt2walk.github.io/)]
- **VoxPoser**[2023]: Generate intermediate programs and 3D affordance maps as strong intermediate representations by LLM. [[paper](https://arxiv.org/pdf/2307.05973)] [[website](https://voxposer.github.io/)]
- **RH-20T**[2023]: Incorporating additional modalities. [[paper](https://arxiv.org/pdf/2307.00595)] [[website](https://rh20t.github.io./)]
- **OTTER**[2021]: Introdues a text-aware feature extraction which preserves semantics aligned with task descriptions. [[paper](https://arxiv.org/pdf/2503.03734)] [[website](https://ottervla.github.io/)]

</details>

<details>
<summary><strong>4.1.2 From 2D Images to Spatial Temporal Representations</strong></summary>
</details>

<details>
<summary><strong>4.1.3 Dynamic and Predictive World Models</strong></summary>

- **GeoPredict**[2025-12-18]: Augments a continuous action policy with predictive kinematic priors that encode motion history and predict multi step 3D keypoint trajectories. [[paper](https://arxiv.org/pdf/2512.16811)] [[website](https://jingjingqian75.github.io/GeoPredict-Page/)]
- **ISS**[2025-12-17]: Introduces an implicit scene supervision module that predicts future point cloud features to enforce long-term geometric consistency. [[paper](https://arxiv.org/pdf/2512.15020)] [[website](-)]
- **Motus**[2025-12-15]: Integrates video generation and action experts via Mixture-of-Transformer architecture with a UniDiffuser-style scheduler. [[paper](https://arxiv.org/pdf/2512.13030)] [[website](https://motus-robotics.github.io/motus)]
- **VL-JEPA**[2025-12-11]: Predict continuous embeddings of target texts in an abstract representation space using a joint embedding predictive architecture. [[paper](https://arxiv.org/pdf/2512.10942)] [[website](-)]
- **VideoVLA**[2025-12-07]: Jointly denoise video latents and action vectors within a unified multi-modal diffusion transformer architecture. [[paper](https://arxiv.org/pdf/2512.06963)] [[website](https://videovla-nips2025.github.io/)]
- **HiMoE-VLA**[2025-12-05]: Train actions with flow matching to model multimodal action distributions. [[paper](https://arxiv.org/pdf/2512.05693)] [[website](https://github.com/ZhiyingDu/HiMoE-VLA)]
- **GigaWorld-0**[2025-11-30]: Unify video generation and 3D physics-aware reconstruction to synthesize controllable embodied interaction data. [[paper](https://arxiv.org/pdf/2511.19861)] [[website](https://giga-world-0.github.io)]
- **LatBot**[2025-11-28]: Learn instruction-guided latent actions from multi-frame videos and jointly decode future frames and inter-frame actions. [[paper](https://arxiv.org/pdf/2511.23034)] [[website](https://mm-robot.github.io/distill_latent_action)]
- **Mantis**[2025-11-20]: Disentangle visual foresight from action learning using a diffusion transformer head with latent action queries and progressive multimodal training. [[paper](https://arxiv.org/pdf/2511.16175)] [[website](https://github.com/zhijie-group/Mantis)]
- **AdaWorld**[2025-06-02]: Develops an autoregressive world model conditioned on latent actions extracted from videos in a self-supervised manner. [[paper](https://arxiv.org/pdf/2503.18938)] [[website](https://adaptable-world-model.github.io)]
- **TriVLA**[2025]: Augments a video diffusion model to produce  multi-step visual rollouts. [[paper](https://arxiv.org/pdf/2507.01424)] [[website](https://robertwyq.github.io/univla.github.io/)]
- **UP-VLA**[2025]: Leverages key subgoal image prediction to represent the next salient task state. [[paper](https://arxiv.org/pdf/2501.18867)] [[website](https://github.com/CladernyJorn/UP-VLA)]
- **CoT-VLA**[2025]: Leverages key subgoal image prediction to represent the next salient task state. [[paper](https://arxiv.org/abs/2503.22020)] [[website](https://cot-vla.github.io/)]
- **DreamVLA**[2025]: Predicting task-critical cues (dynamic regions, depth, and affordance features). [[paper](https://arxiv.org/pdf/2507.04447)] [[website](https://zhangwenyao1.github.io/DreamVLA/)]
- **V-jepa 2**[2025]: Leverages latent-space encoding and prediction to model future state evolution. [[paper](https://arxiv.org/pdf/2506.09985)] [[website](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)]
- **LUMOS**[2025]: Leverages multi-step internal rollouts in a learned world model to evaluate action sequences and select an optimal plan. [[paper](https://arxiv.org/pdf/2503.10370)] [[website](http://lumos.cs.uni-freiburg.de/)]
- **FlowVLA**[2025]: Leverages a visual chain-of-thought mechanism to forecast future frames and synthesize physically consistent scenes. [[paper](https://arxiv.org/abs/2508.18269)] [[website](https://irpn-lab.github.io/FlowVLA/)]
- **WorldVLA**[2025]: Leverages learned world dynamics to model low-level physics and synthesize future physical evolution. [[paper](https://arxiv.org/pdf/2506.21539)] [[website](https://github.com/alibaba-damo-academy/RynnVLA-002)]
- **VLM-in-the-loop**[2025]: Leverages internal multi-step rollouts in a world model to evaluate and select action sequences. [[paper](https://arxiv.org/pdf/2502.01828)] [[website](https://yilin-wu98.github.io/forewarn/)]
- **MinD**[2025]: Leverages internal multi-step rollouts to evaluate long-horizon outcomes and select optimal actions. [[paper](https://arxiv.org/abs/2502.07591)] [[website](https://github.com/news-vt/DMWM)]
- **WMPO**[2025]: Leverages imagined interaction in a video world model to replace costly physical interaction. [[paper](https://arxiv.org/pdf/2511.09515)] [[website](https://wm-po.github.io/)]

</details>

---

### 2️⃣ Instruction Following, Planning, and Robust Real-Time Execution  
*(Sec. 4.2)*

<details>
<summary><strong>4.2.1 Parsing Complex Instructions</strong></summary>

- **OE-VLA**[2025]: Leverages a shared visual encoder and text tokenizer to produce strictly interleaved token streams. [[paper](https://arxiv.org/pdf/2505.11214)] [[website](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)]
- **Interleave-VLA**[2025]: Leverages special tokenizer tags to insert image features seamlessly into text sequences. [[paper](https://arxiv.org/pdf/2505.02152)] [[website](https://interleave-vla.github.io/Interleave-VLA-Anonymous/)]
- **TinkAct**[2025]: infers and verifies the in tended target via scene parsing and feedback. [[paper]("https://arxiv.org/pdf/2507.16815)] [[website](https://jasper0314-huang.github.io/thinkact-vla/)]
- **DEEPTHINKVLA**[2025]: Leverages causal chain-of-thought and outcome-driven RL to resolve ambiguity and align subgoals. [[paper](https://arxiv.org/pdf/2511.15669)] [[website](https://github.com/wadeKeith/DeepThinkVLA)]
- **InSpire**[2025]: leverages explicit spatial queries to prompt the policy for target-robot relative location. [[paper](https://arxiv.org/pdf/2505.13888)] [[website](https://koorye.github.io/proj/Inspire/)]
- **AskToAct**[2025]: leverages explicit spatial queries to auto-fill missing target-robot information. [[paper](https://arxiv.org/pdf/2503.01940)] [[website](https://github.com/emrecanacikgoz/awesome-conversational-agents)]

</details>

<details>
<summary><strong>4.2.2 Hierarchical Planning and Task Decomposition</strong></summary>

- **Lumo-1**[2025-12-10]: Train with staged reasoning action alignment and flow matched action expert for efficient execution. [[paper](https://arxiv.org/pdf/2512.08580)] [[website](https://www.astribot.com/research/Lumo1)]
- **MM-ACT⭐**[2025-12-08]: Integrates text image and action in shared token space to perform task planning and future image prediction under shared context. [[paper](https://arxiv.org/pdf/2512.00975)] [[website](https://github.com/HHYHRHY/MM-ACT)]
- **ManualVLA**[2025-12-01]: Design a unified Mixture-of-Transformers framework that selectively activates planning and action experts for coherent manual generation and execution. [[paper](https://arxiv.org/pdf/2512.02013)] [[website](https://sites.google.com/view/maunalvla)]
- **MoE-DP**[2025-11-07]: Integrate a Mixture of Experts layer to decompose long-horizon tasks into specialized skills within the diffusion policy. [[paper](https://arxiv.org/pdf/2511.05007)] [[website](https://moe-dp-website.github.io/MoE-DP-Website/)]
- **WALL-OSS⭐**[2025-09-08]: Develop Unified Cross-Level CoT to seamlessly unify instruction reasoning subgoal decomposition and fine-grained action synthesis. [[paper](https://arxiv.org/pdf/2509.11766)] [[website](https://x2robot.com/en/research/68bc2cde8497d7f238dde690)]
- **OneTwoVLA**[2025]: leverages structured textual reasoning to generate scene descriptions high-level plans and next-step instructions. [[paper](https://arxiv.org/pdf/2505.11917)] [[website](https://one-two-vla.github.io/)]
- **PI0.5**[2025]: embeds hierarchical reason ing within a single inference chain. [[paper](https://arxiv.org/pdf/2504.16054)] [[website](https://www.pi.website/blog/pi05)]
- **Hi Robot**[2025]: employ a two-layer scheme where a VLM parses  instructions into atomic sub-tasks. [[paper](https://arxiv.org/pdf/2502.19417)] [[website](https://www.pi.website/research/hirobot)]
- **LoHoVLA**[2025]: leverages structured textual reasoning to generate scene descriptions high-level plans and next-step instructions. [[paper](https://arxiv.org/pdf/2506.00411)] [[website](-)]
- **CoT-VLA**[2025]: employs pixel-level subgoal images as explicit intermediates. [[paper](https://arxiv.org/pdf/2503.22020)] [[website](https://cot-vla.github.io/)]
- **CoA-VLA**[2025]: Treats each affordance link as an implicit planning signal. [[paper](https://arxiv.org/pdf/2412.20451)] [[website](https://chain-of-affordance.github.io/)]
- **VLP**[2025]: builds a fine-grained library for data-efficient reuse of manipulation patterns. [[paper](https://arxiv.org/pdf/2401.05577)] [[website](https://github.com/autodriving-heart/CVPR-2024-Papers-Autonomous-Driving)]
- **Agentic Robot**[2025]: Produces clear step subgoals sequences for verifiable task decomposition. [[paper](https://arxiv.org/pdf/2505.23450)] [[website](https://agentic-robot.github.io/)]
- **RoboBrain**[2025]: Leverages a hierarchical framework to map abstract instructions to executable atomic actions. [[paper](https://arxiv.org/pdf/2502.21257)] [[website](https://superrobobrain.github.io/)]
- **DexVLA**[2025]: Leverages temporal alignment to automatically annotate semantic sub-steps in long-horizon sequences. [[paper](https://arxiv.org/pdf/2502.05855)] [[website](https://github.com/juruobenruo/DexVLA)]
- **AgiBot**[2025]: Leverages explicit skills during data collection to learn latent action tokens that compress high-dimensional control. [[paper](https://arxiv.org/pdf/2503.06669)] [[website](https://opendrivelab.com/AgiBot-World/)]
- **RT-Affordance**[2024]: plans tasks by breaking manipulation into manageable affordance steps. [[paper](https://arxiv.org/pdf/2411.02704)] [[website](https://snasiriany.me/rt-affordance)]
- **Embodied-SlotSSM**[2023]: employs slot-based object-centric representations to create structured visual intermediates. [[paper](https://www.arxiv.org/pdf/2511.11478)] [[website](https://libero-mem.github.io/)]

</details>

<details>
<summary><strong>4.2.3 Error Detection and Autonomous Recovery</strong></summary>

- **AFI**[2025-12-08]: Detect memory traps using proprioception and execute affordance-guided rollback for recovery. [[paper](https://arxiv.org/pdf/2512.07472)] [[website](-)]
- **ViFailback**[2025-12-03]: Use a VLM to detect localize and classify failures then generate textual and visual corrective guidance. [[paper](https://arxiv.org/pdf/2512.02787)] [[website](https://x1nyuzhou.github.io/vifailback.github.io)]
- **MoE-DP**[2025-11-07]: Leverage dynamic expert routing to detect failures and reactivate appropriate experts for subtask retries. [[paper](https://arxiv.org/pdf/2511.05007)] [[website](https://moe-dp-website.github.io/MoE-DP-Website/)]
- **OneTwoVLA**[2025]: Actively queries humans to resolve uncertainty before acting. [[paper](https://arxiv.org/pdf/2505.11917)] [[website](https://one-two-vla.github.io/)]
- **CorrectNav**[2025]: Leverages iterative self-correction by using the model's own error trajectories to generate corrective actions and data. [[paper](https://arxiv.org/pdf/2508.10416)] [[website](https://correctnav.github.io/)]
- **FPC-VLA**[2025]: Leverages a VLM to assess action semantics and generate corrective language feedback. [[paper](https://arxiv.org/pdf/2509.04018)] [[website](https://fpcvla.github.io/)]
- **Agentic Robot**[2025]: Achieves autonomous correction via a plan-act-verify loop with VLM-based validation and recovery. [[paper](https://arxiv.org/pdf/2505.23450)] [[website](https://agentic-robot.github.io/)]
- **Yell At Your Robot**[2024]: Leverages  real-time language feedback for instant behavioral correction. [[paper](https://arxiv.org/pdf/2403.12910)] [[website](https://yay-robot.github.io/)]
- **CLIP-RT**[2024]: Leverages language feedback as an action template via similarity matching for retrain-free correction. [[paper](https://arxiv.org/pdf/2411.00508)] [[website](https://clip-rt.github.io/)]

</details>

<details>
<summary><strong>4.2.4 Real-Time Execution and Computing Efficiency</strong></summary>

- **ISS**[2025-12-17]: Uses a streamlined DiT architecture with sparse point cloud encoding to improve inference speed and training efficiency. [[paper](https://arxiv.org/pdf/2512.15020)] [[website](-)]
- **WholeBodyVLA**[2025-12-15]: Employ a discrete command interface with two-stage curriculum RL policy for precise execution. [[paper](https://arxiv.org/pdf/2512.11047)] [[website](https://opendrivelab.com/WholeBodyVLA)]
- **VL-JEPA**[2025-12-11]: Enable selective decoding by invoking the text decoder only when significant semantic changes are detected in the predicted embedding stream. [[paper](https://arxiv.org/pdf/2512.10942)] [[website](-)]
- **HiF-VLA**[2025-12-10]: Use compact motion representations to expand temporal reasoning with negligible latency. [[paper](https://arxiv.org/pdf/2512.09928)] [[website](https://hifvla.github.io)]
- **MM-ACT⭐**[2025-12-08]: Adopts a one-step parallel decoding strategy for action generation to achieve low-latency inference. [[paper](https://arxiv.org/pdf/2512.00975)] [[website](https://github.com/HHYHRHY/MM-ACT)]
- **HiMoE-VLA**[2025-12-05]: Cache intermediate VLM key value pairs for faster inference without degrading performance. [[paper](https://arxiv.org/pdf/2512.05693)] [[website](https://github.com/ZhiyingDu/HiMoE-VLA)]
- **DeepThinkVLA**[2025-10-31]: Use a hybrid-attention decoder with SFT then outcome-based RL to align reasoning with actions. [[paper](https://arxiv.org/pdf/2511.15669)] [[website](https://github.com/wadeKeith/DeepThinkVLA)]
- **CronusVLA⭐**[2025-10-30]: Optimizes inference speed by predicting learnable features and using a queue mechanism for feature caching. [[paper](https://arxiv.org/pdf/2506.19816)] [[website](-)]
- **BitVLA**[2025]: achieves ultra-low-precision efficiency  via ternary 1-bit compression and distillation. [[paper](https://arxiv.org/pdf/2506.07530)] [[website](https://github.com/ustcwhy/BitVLA)]
- **Evo-1**[2025]: Provides a lightweight 77M-parameter design. [[paper](https://arxiv.org/pdf/2511.04555)] [[website](https://github.com/MINT-SJTU/Evo-1)]
- **SQAP-VLA**[2025]: Introduces perceptual pruning strategies on the basis of quantization. [[paper](https://arxiv.org/pdf/2509.09090)] [[website](https://github.com/ecdine/SQAP-VLA)]
- **TinyVLA**[2025]: Adopt lightweight backbones directly. [[paper](https://arxiv.org/pdf/2409.12514)] [[website](https://tiny-vla.github.io/)]
- **VLA-Adapter**[2025]: Leverages lightweight adapters to graft large-model knowledge into smaller policies. [[paper](https://arxiv.org/pdf/2509.09372)] [[website](https://vla-adapter.github.io/)]
- **NORA**[2025]: Adopt lightweight backbones directly. [[paper](https://arxiv.org/pdf/2504.19854)] [[website](https://declare-lab.github.io/nora)]
- **MoLe-VLA**[2025]: leverages layer skipping to reduce FLOPs. [[paper](https://arxiv.org/pdf/2503.20384)] [[website](https://sites.google.com/view/mole-vla)]
- **CEED-VLA**[2025]: Design early exit mechanisms. [[paper](https://arxiv.org/pdf/2506.13725)] [[website](https://irpn-eai.github.io/CEED-VLA/)]
- **VLA-Cache**[2025]: Uuses adaptive caching that differentiates static and dynamic tokens. [[paper](https://arxiv.org/pdf/2502.02175)] [[website](https://vla-cache.github.io/)]
- **SpecPrune-VLA**[2025]: performs action aware pruning conditioned on history and current observations. [[paper](https://arxiv.org/pdf/2509.05614)] [[website](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)]
- **CogVLA**[2025]: reduces computation through  instruction-driven visual token sparsification. [[paper](https://arxiv.org/pdf/2508.21046)] [[website](https://jiutian-vl.github.io/CogVLA-page/)]
- **AcceleratingVLA**[2025]: Leverages parallel decoding to generate full action chunks in one pass. [[paper](https://arxiv.org/pdf/2503.02310)] [[website](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **OpenVLA-OFT**[2025]: Leverages parallel decoding to generate full action chunks in one pass. [[paper](https://arxiv.org/pdf/2502.19645)] [[website](https://github.com/moojink/openvla-oft)]
- **Spec-VLA**[2025]: Leverages speculative decoding to emit candidate action tokens in one pass. [[paper](https://arxiv.org/pdf/2507.22424)] [[website](https://github.com/PineTreeWss/SpecVLA)]
- **FAST**[2025]: Compresses action sequences. [[paper](https://arxiv.org/pdf/2501.09747)] [[website](https://www.pi.website/research/fast)]
- **VQ-VLA**[2025]: Leverages a VQ-VAE tokenizer to compress long trajectories into compact discrete tokens. [[paper](https://arxiv.org/pdf/2507.01016)] [[website](https://xiaoxiao0406.github.io/vqvla.github.io/)]
- **XR-1**[2025]: Leverages VQ-VAE–learned discrete visual–motor tokens to guide policy learning. [[paper](https://arxiv.org/pdf/2511.02776)] [[website](https://xr-1-vla.github.io/)]
- **SmolVLA**[2025]: Leverages asynchronous execution to predict the next action chunk during current execution. [[paper](https://arxiv.org/pdf/2506.01844)] [[website](https://github.com/huggingface/smollm)]
- **Time-Diffusion Policy**[2025]: Leverages a unified velocity field to replace time-varying denoising. [[paper](https://arxiv.org/pdf/2506.09422)] [[website](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **Discrete Diffusion VLA**[2025]: Discretizes actions into tokens and employs masked diffusion with parallel prediction. [[paper](https://arxiv.org/pdf/2508.20072)] [[website](https://github.com/Liang-ZX/DiscreteDiffusionVLA)]
- **ECoT-Lite**[2025]: Leverages reasoning traces during training while bypassing explicit reasoning at inference. [[paper](https://arxiv.org/pdf/2505.08243v1)] [[website](https://github.com/MichalZawalski/embodied-CoT)]
- **V-JEPA**[2025]: Predicting compressed semantic representations instead of raw pixels. [[paper](https://arxiv.org/pdf/2506.09985)] [[website](https://github.com/facebookresearch/vjepa2)]
- **Fast-in-Slow**[2025]: Leverages a dual-system model coordinating slow reasoning and fast reactions. [[paper](https://arxiv.org/pdf/2506.01953)] [[website](https://fast-in-slow.github.io/)]
- **AMS**[2025]: Introduces OS-level action context caching and replay mechanisms. [[paper](https://arxiv.org/pdf/2405.09045)] [[website](https://github.com/AMS-Net/ams-net.github.io)]
- **FedVLA**[2025]: Leverages federated learning for efficient distributed VLA training. [[paper](https://arxiv.org/pdf/2508.02190)] [[website](-)]
- **RoboMamba**[2024]: Leverages a Mamba backbone for linear-time scaling and faster inference. [[paper](https://arxiv.org/pdf/2406.04339)] [[website](https://sites.google.com/view/robomamba-web)]
- **DeeR-VLA**[2024]: Design early exit mechanisms. [[paper](https://arxiv.org/pdf/2411.02359)] [[website](https://github.com/yueyang130/DeeR-VLA)]

</details>

---

### 3️⃣ From Generalization to Continuous Adaptation  
*(Sec. 4.3)*

<details>
<summary><strong>4.3.1 Open-World Generalization</strong></summary>

- **Human to Robot**[2025-12-16]: Co-train human videos as an additional embodiment with robot data using unified trajectory and sub-task prediction objectives. [[paper](https://www.physicalintelligence.company/download/human_to_robot.pdf)] [[website](https://www.physicalintelligence.company/research/human_to_robot)]
- **OXE-AugE**[2025-12-15]: Fine-tunes generalist policies on augmented data to improve zero-shot transfer to unseen embodiments. [[paper](https://arxiv.org/pdf/2512.13100)] [[website](https://OXE-AugE.github.io/)]
- **WholeBodyVLA**[2025-12-15]: Train separate locomotion and manipulation latent action models on human videos to supervise VLA training. [[paper](https://arxiv.org/pdf/2512.11047)] [[website](https://opendrivelab.com/WholeBodyVLA)]
- **BayesVLA**[2025-12-12]: Decompose policy into vision-action prior and language-conditioned likelihood to mitigate modality imbalance. [[paper](https://arxiv.org/pdf/2512.11218)] [[website](https://xukechun.github.io/papers/BayesVLA)]
- **GLaD**[2025-12-10]: Pretrain on Bridge then fine tune with LoRA for robust policy generalization. [[paper](https://arxiv.org/pdf/2512.09619)] [[website](-)]
- **AFI**[2025-12-08]: Employ 3D Spatial Affordance Fields to guide VLA behavior in out-of-distribution scenarios. [[paper](https://arxiv.org/pdf/2512.07472)] [[website](-)]
- **VideoVLA**[2025-12-07]: Leverage pre-trained video generation models and a dual-prediction strategy to transfer physical knowledge to robotic manipulation. [[paper](https://arxiv.org/pdf/2512.06963)] [[website](https://videovla-nips2025.github.io/)]
- **HiMoE-VLA**[2025-12-05]: Use a hierarchical mixture of experts with action space and heterogeneity balancing experts to enable cross domain transfer. [[paper](https://arxiv.org/pdf/2512.05693)] [[website](https://github.com/ZhiyingDu/HiMoE-VLA)]
- **MergeVLA**[2025-11-24]: Introduce sparsely activated LoRA masks and cross-attention-only action experts with a training-free task router to enable mergeable multi-skill VLA policies. [[paper](https://arxiv.org/pdf/2511.18810)] [[website](https://mergevla.github.io)]
- **InternVLA-M1⭐**[2025-10-15]: Utilizes synthetic co-training and spatial prompting to achieve robust performance on unseen objects and long-horizon tasks. [[paper](https://arxiv.org/pdf/2510.13778)] [[website](https://internrobotics.github.io/internvla-m1.github.io/)]
- **X-VLA⭐**[2025-10-14]: Employs a two-step adaptation process that warms up prompts before joint policy finetuning. [[paper](https://arxiv.org/pdf/2510.10274)] [[website](https://thu-air-dream.github.io/X-VLA/)]
- **X-VLA**[2025-10-11]: Learn embodiment-specific soft prompts to absorb cross-embodiment heterogeneity and enable scalable pretraining and efficient adaptation. [[paper](https://arxiv.org/pdf/2510.10274)] [[website](https://thu-air-dream.github.io/X-VLA/)]
- **GENMANIP⭐**[2025-06-12]: Leverage LLM driven task oriented scene graph to synthesize diverse tasks using 10K annotated 3D object assets. [[paper](https://arxiv.org/pdf/2506.10966)] [[website](-)]
- **AdaWorld**[2025-06-02]: Enables efficient adaptation and action transfer to novel environments by initializing the control interface with learned latent actions. [[paper](https://arxiv.org/pdf/2503.18938)] [[website](https://adaptable-world-model.github.io)]
- **ROBOGROUND⭐**[2025-04-30]: Leverage grounding masks as effective intermediate representations to provide spatial guidance for policy networks. [[paper](https://arxiv.org/pdf/2504.21530)] [[website](https://robo-ground.github.io)]
- **GR00T N1**[2025-03-28]: Employ a dual-system architecture integrating a vision-language backbone with a flow-matching diffusion policy for cross-embodiment control. [[paper](https://arxiv.org/pdf/2503.14734)] [[website](https://research.nvidia.com/labs/gear/gr00t-n1_6/)]
- **DexVLA**[2025]: Pretrain diffusion action experts across morphologies with a three stage curriculum. [[paper](https://arxiv.org/pdf/2502.05855)] [[website](https://dex-vla.github.io/)]
- **EO-1**[2025]: Pretrain a shared backbone on 1.5M EO Data. [[paper](https://arxiv.org/pdf/2508.21112)] [[website](https://github.com/eo-robotics/EO1)]
- **TRA**[2025]: Use temporal contrastive loss to structure representation space. [[paper](https://arxiv.org/pdf/2502.05454)] [[website](https://tra-paper.github.io/)]
- **ObjectVLA**[2025]: Jointly train on robot trajectories and box labeled VL corpora. [[paper]("https://arxiv.org/pdf/2502.19250)] [[website](https://objectvla.github.io/)]
- **RUM**[2025]: Pair large scale home demonstrations with multimodal LLM reasoning. [[paper](https://arxiv.org/pdf/2508.12922)] [[website](https://github.com/kairanzhao/RUM)]
- **Align-Then-Steer**[2025]: Leverages a latent-space adapter to steer a frozen VLA model non-invasively. [[paper](https://arxiv.org/pdf/2509.02055)] [[website](https://github.com/TeleHuman/Align-Then-Steer)]
- **LMM-3DP**[2025]: Leverages a high-level planner for abstract knowledge and a low-level executor for reusable skills. [[paper](https://arxiv.org/pdf/2501.18733)] [[website](https://lmm-3dp-release.github.io/)]
- **Octo**[2024]: Pretrain Transformer on 800k trajectories and use lightweight adapters. [[paper](https://arxiv.org/pdf/2405.12213)] [[website](https://octo-models.github.io/)]
- **Dita**[2024]: Use OXE dataset and diffusion Transformers to learn cross environment behaviors. [[paper](https://arxiv.org/pdf/2503.19757)] [[website](https://robodita.github.io/)]
- **Gr-2**[2024]: Pretrain on massive human egocentric video datasets. [[paper](https://arxiv.org/pdf/2410.06158)] [[website](https://gr2-manipulation.github.io/)]
- **ICIL**[2024]: Leverages in-context learning to infer tasks from few-shot demonstrations without retraining. [[paper](https://arxiv.org/pdf/2408.15980)] [[website](https://icrt.dev/)]
- **BAKU**[2024]: Leverages dynamic multimodal sensor fusion. [[paper](https://arxiv.org/pdf/2406.07539)] [[website](https://baku-robot.github.io/)]
- **RoboCat**[2023]: Pretrain on heterogeneous multi robot data and update on real trajectories. [[paper](https://arxiv.org/pdf/2306.11706)] [[website](https://github.com/kyegomez/RoboCAT)]
- **GR-1**[2023]: Pretrain on massive human egocentric video datasets. [[paper](https://arxiv.org/pdf/2312.13139)] [[website](https://gr1-manipulation.github.io/)]
- **LERF**[2023]: Fuse CLIP with 3D NeRFs. [[paper](https://arxiv.org/pdf/2303.09553)] [[website](https://www.lerf.io/)]
- **GenAug**[2023]: Leverages text-to-image synthesis from few-shot demos and prompts to generate diverse consistent scenes. [[paper](https://arxiv.org/pdf/2302.06671)] [[website](https://genaug.github.io/)]
- **ROSIE**[2023]: Distill internet scale VLM knowledge into robot training. [[paper](https://arxiv.org/pdf/2302.11550)] [[website](https://diffusion-rosie.github.io/)]
- **R3M**[2022]: Pretrain visual encoders on massive human first-person videos. [[paper](https://arxiv.org/pdf/2203.12601)] [[website](https://sites.google.com/view/robot-r3m/)]
- **Ego4D**[2022]: Pretrain visual encoders on massive human first-person videos. [[paper](https://arxiv.org/pdf/2110.07058)] [[website](https://ego4d-data.org/)]
- **CACTI**[2022]: Leverages Stable Diffusion for zero-shot inpainting to diversify expert images without extra rollouts. [[paper](https://arxiv.org/pdf/2212.05711)] [[website](https://cacti-framework.github.io/)]
- **StructDiffusion**[2022]: Leverages language guided diffusion to generate multiple action structures. [[paper](https://arxiv.org/pdf/2211.04604)] [[website](https://structdiffusion.github.io/)]

</details>

<details>
<summary><strong>4.3.2 Continual Learning and Incremental Skill Acquisition</strong></summary>

- **DexGrasp-VLA**[2025-12-13]: Implement a Corrective Human-in-the-loop Teleoperation system to enable continuous policy improvement via failure recovery. [[paper](https://arxiv.org/pdf/2511.00139)] [[website](https://dexvla-seed.github.io/dex-vla)]
- **Think Small Act Big**[2025]: Leverages new prompts or codebook entries to add skills without modifying existing components. [[paper](https://arxiv.org/pdf/2504.00420)] [[website](-)]
- **SPECI**[2025]: Leverages new prompts or codebook entries to add skills without modifying existing components. [[paper](https://arxiv.org/pdf/2504.15561)] [[website](-)]
- **InstructVLA**[2025]: Leverages a two-stage paradigm with a MoE to route between reasoning and action modules. [[paper](https://arxiv.org/pdf/2507.17520)] [[website](https://yangs03.github.io/InstructVLA_Home/)]
- **iManip**[2025]: Add skill specific weights while freezing old ones. [[paper](https://arxiv.org/pdf/2503.07087)] [[website](https://github.com/Ghy0501/Awesome-Continuall-Learning-in-Generative-Models)]
- **ExpReSVLA**[2025]: Use Compressed Experience Replay. [[paper](https://arxiv.org/pdf/2511.06202)] [[website](-)]

</details>

<details>
<summary><strong>4.3.3 Sim-to-Real Gap in Deployment</strong></summary>

- **MiVLA**[2025-12-17]: Utilizes kinematic rules with left/right hand coordinate systems for bidirectional alignment between human and robot action spaces. [[paper](https://arxiv.org/pdf/2512.15411)] [[website](-)]
- **RoboWheel**[2025-12-02]: Uses a simulation-augmented data flywheel with domain randomization in Isaac Sim to enrich trajectory distributions for robust real-world transfer. [[paper](https://arxiv.org/pdf/2512.02729)] [[website](https://zhangyuhong01.github.io/Robowheel)]
- **DiG-Flow**[2025-12-1]: Compute a distributional discrepancy between observation and action embeddings to guide residual feature updates for robust flow matching. [[paper](https://arxiv.org/pdf/2512.01715)] [[website](https://beingbeyond.github.io/DiG-Flow)]
- **ManualVLA**[2025-12-01]: Construct a high-fidelity digital twin toolkit using 3D Gaussian Splatting to synthesize large-scale training data. [[paper](https://arxiv.org/pdf/2512.02013)] [[website](https://sites.google.com/view/maunalvla)]
- **SLIM**[2025]: Compress RGB into segmentation and depth maps. [[paper](https://arxiv.org/pdf/2410.09615)] [[website](https://github.com/Paramathic/slim)]
- **DreamGen**[2025]: Train a world model on massive real world data. [[paper](https://arxiv.org/pdf/2505.12705)] [[website](https://github.com/nvidia/GR00T-dreams)]
- **RynnVLA-001**[2025]: Pretrain large scale video generation with human centric trajectory modeling. [[paper](https://arxiv.org/pdf/2509.15212)] [[website](https://github.com/alibaba-damo-academy/RynnVLA-001)]
- **MaiSkill3**[2024]: Leverages GPU-parallel rendering domain randomization and background composition. [[paper](https://arxiv.org/pdf/2410.00425)] [[website](https://github.com/haosulab/ManiSkill)]
- **GenAug**[2023]: Leverages  web scale image generative models to synthesize images from few demonstrations. [[paper](https://arxiv.org/abs/2302.06671)] [[website](https://github.com/genaug/genaug)]

</details>

<details>
<summary><strong>4.3.4 Online Interaction and Reinforcement Learning</strong></summary>

- **EVOLVE-VLA**[2025-12-16]: Introduce accumulative progress estimation mechanism and progressive horizon extension strategy to tame noisy reward signals for test-time training. [[paper](https://arxiv.org/pdf/2512.14666)] [[website](https://showlab.github.io/EVOLVE-VLA)]
- **Lumo-1**[2025-12-10]: Leverage cross embodiment co training and RL to improve generalization. [[paper](https://arxiv.org/pdf/2512.08580)] [[website](https://www.astribot.com/research/Lumo1)]
- **ProphRL**[2025-11-25]: Pretrain a unified action-conditioned world model (Prophet) on diverse robot data. [[paper](https://arxiv.org/pdf/2511.20633)] [[website](https://LogosRoboticsGroup.github.io/ProphRL)]
- **Refined Policy Distillation**[2025]: Add MSE constraint to guide RL agent. [[paper](https://arxiv.org/pdf/2503.05833)] [[website](https://refined-policy-distillation.github.io/)]
- **iRe-VLA**[2025]: Freeze backbone and train lightweight action head in alternating phases. [[paper](https://arxiv.org/pdf/2501.16664)] [[website](https://github.com/HaochenZ11/IRef-VLA)]
- **CO-RFT**[2025]: designs a chunked temporal difference learning mechanism that feeds entire action sequences into the critic to predict multi step returns. [[paper](https://arxiv.org/pdf/2508.02219)] [[website](https://github.com/cccedric/conrft)]
- **VLA-RL**[2025]: Finetune VLM into a structured process reward model. [[paper](https://arxiv.org/pdf/2505.18719)] [[website](https://github.com/GuanxingLu/vlarl)]
- **RIDG**[2024]: Train specialist RL policies then distill trajectories into VLA. [[paper](https://arxiv.org/pdf/2412.09858)] [[website](https://generalist-distillation.github.io/)]
- **Affordance-Guided RL**[2024]: Convert VLM predicted grasp points and trajectories into dense rewards. [[paper](https://arxiv.org/pdf/2407.10341)] [[website](https://sites.google.com/view/affordance-guided-rl)]
- **RL VLM-F**[2024]: Leverages GPT-4V to infer preference-based rewards from observation pairs without human labels. [[paper](https://arxiv.org/pdf/2402.03681)] [[website](https://rlvlmf2024.github.io/)]
- **GRAPE**[2024]: Leverages VLMs to decompose tasks and generate stage-wise preferences for multi-objective rewards. [[paper](https://arxiv.org/pdf/2411.19309)] [[website](https://grape-vla.github.io/)]
- **VLM-RMs**[2023]: Infer rewards via perceptual alignment in shared embedding space. [[paper](https://arxiv.org/pdf/2310.12921)] [[website](https://github.com/AlignmentResearch/vlmrm)]
- **RoboCLIP**[2023]: Leverages video trajectories by computing video language similarity for sparse rewards. [[paper](https://arxiv.org/pdf/2310.07899)] [[website](https://sites.google.com/view/roboclip/home)]
- **Eureka**[2023]: Prompt LLM with environment code and task specs to generate rewards. [[paper](https://arxiv.org/pdf/2310.12931)] [[website](https://eureka-research.github.io/)]
- **VIP**[2023]: Perform implicit value optimization from video. [[paper](https://arxiv.org/pdf/2210.00030)] [[website](https://sites.google.com/view/vip-rl)]

</details>

---

### 4️⃣ Safety, Interpretability, and Reliable Interaction  
*(Sec. 4.4)*

<details>
<summary><strong>4.4.1 Reliability and Safety Assurance</strong></summary>

- **VLSA-AEGIS**[2025-12-09]: Introduce a plug-and-play safety constraint layer formulated via control barrier functions into VLA models. [[paper](https://arxiv.org/pdf/2512.11891)] [[website](https://vlsa-aegis.github.io/)]
- **AutoRT**[2025]: Leverages structured prompting to encode multi level constraints. [[paper](https://arxiv.org/pdf/2401.12963)] [[website](https://auto-rt.github.io/)]
- **SafeVLA**[2025]: Leverages a cost function in a constrained MDP to model physically hazardous behaviors. [[paper](https://arxiv.org/pdf/2503.03480)] [[website](https://pku-safevla.github.io/)]
- **Genimi Robotics**[2025]: Leverages Constitutional AI post-training on safety data to enforce human-centric principles. [[paper](https://arxiv.org/pdf/2503.20020)] [[website](https://github.com/google-deepmind/gemini-robotics-sdk)]
- **GPI**[2025]: Leverages confidence estimation probabilistic action generation and language guided backtracking to replan under uncertainty. [[paper](https://arxiv.org/pdf/2508.11960)] [[website](-)]
- **RationalVLA**[2025]: Uses a learnable refusal token to reject unsafe or invalid commands. [[paper](https://arxiv.org/pdf/2506.10826)] [[website](https://irpn-eai.github.io/RationalVLA/)]

</details>

<details>
<summary><strong>4.4.2 Interpretability and Trustworthy Interaction</strong></summary>

- **Lumo-1**[2025-12-10]: Provide explicit reasoning traces for transparent decisions. [[paper](https://arxiv.org/pdf/2512.08580)] [[website](https://www.astribot.com/research/Lumo1)]
- **ViFailback**[2025-12-03]: Use explicit on-frame visual symbols to provide interpretable corrective cues. [[paper](https://arxiv.org/pdf/2512.02787)] [[website](https://x1nyuzhou.github.io/vifailback.github.io)]
- **MoE-DP**[2025-11-07]: Apply auxiliary load balancing and entropy losses to enforce distinct mappings between experts and semantic task phases. [[paper](https://arxiv.org/pdf/2511.05007)] [[website](https://moe-dp-website.github.io/MoE-DP-Website/)]
- **CoT-VLA**[2025]: Adds visual subgoal images to render intermediate plans observable. [[paper](https://arxiv.org/pdf/2503.22020)] [[website](https://cot-vla.github.io/)]
- **CrayonRobo**[2025]: Uses structured semantically explicit visual prompts to externalize decision logic into a shared and interpretable language. [[paper](https://arxiv.org/pdf/2505.02166)] [[website](-)]
- **SwitchVLA**[2025]: Use structured task switching with rollback of conflicting actions. [[paper](https://arxiv.org/pdf/2506.03574)] [[website](https://switchvla.github.io/)]
- **Hi Robot**[2025]: Outputs readable low-level commands from a high-level planner. [[paper](https://arxiv.org/pdf/2502.19417)] [[website](https://www.pi.website/research/hirobot)]
- **GraSPVLA**[2025]: Uses symbolic state conversion of visual inputs for planning in a symbolic space. [[paper](https://arxiv.org/pdf/2511.04357)] [[website](https://github.com/PKU-EPIC/GraspVLA)]
- **DIARC-OpenVLA**[2025]: Train linear probes to map hidden activations to symbolic states for transparent monitoring. [[paper](https://arxiv.org/pdf/2502.04558v1)] [[website](-)]
- **Diffusion-VLA**[2024]: Condition diffusion policy on natural language reasoning. [[paper](https://arxiv.org/pdf/2412.03293)] [[website](https://diffusion-vla.github.io/)]
- **ECoT**[2024]: Leverages editable step-by-step rationales that users can correct via language. [[paper](https://arxiv.org/pdf/2407.08693)] [[website](https://embodied-cot.github.io/)]
- **RT-H**[2024]: Leverages separated language-action generation to enable self-explanation and language-level intervention. [[paper](https://arxiv.org/pdf/2403.01823)] [[website](https://rt-hierarchy.github.io/)]

</details>

---

### 5️⃣ Data Construction and Benchmarking Standards  
*(Sec. 4.5)*

<details>
<summary><strong>4.5.1 Multi-Source Heterogeneous Data</strong></summary>

- **PhysBrain**[2025-12-19]: Transform first-person videos into multi-level schema-driven VQA supervision with enforced evidence grounding and temporal consistency to construct the E2E-3M dataset. [[paper](https://arxiv.org/pdf/2512.16793)] [[website](https://zgc-embodyai.github.io/PhysBrain/)]
- **Motus**[2025-12-15]: Introduces latent actions derived from optical flow to unify large-scale heterogeneous data across embodiments. [[paper](https://arxiv.org/pdf/2512.13030)] [[website](https://motus-robotics.github.io/motus)]
- **OXE-AugE**[2025-12-15]: Augments existing datasets with diverse robot embodiments using a scalable pipeline combining simulation rendering and learned masks. [[paper](https://arxiv.org/pdf/2512.13100)] [[website](https://OXE-AugE.github.io/)]
- **DexGrasp-VLA**[2025-12-13]: Propose a Shared Autonomy framework that partitions control along macro-micro motion domains for efficient data collection. [[paper](https://arxiv.org/pdf/2511.00139)] [[website](https://dexvla-seed.github.io/dex-vla)]
- **Lumo-1**[2025-12-10]: Curate large scale multi source VLM and robot data and define evaluation suites. [[paper](https://arxiv.org/pdf/2512.08580)] [[website](https://www.astribot.com/research/Lumo1)]
- **RoboWheel**[2025-12-02]: Converts human hand-object interaction videos from diverse sources into training-ready supervision via a physics-aware reconstruction pipeline. [[paper](https://arxiv.org/pdf/2512.02729)] [[website](https://zhangyuhong01.github.io/Robowheel)]
- **InternVLA-M1⭐**[2025-10-15]: Builds a scalable simulation engine to generate large-scale diverse manipulation episodes and spatial grounding annotations. [[paper](https://arxiv.org/pdf/2510.13778)] [[website](https://internrobotics.github.io/internvla-m1.github.io/)]
- **X-VLA⭐**[2025-10-14]: Introduces distinct learnable soft prompt embeddings for each data source to absorb embodiment-specific variations. [[paper](https://arxiv.org/pdf/2510.10274)] [[website](https://thu-air-dream.github.io/X-VLA/)]
- **WALL-OSS⭐**[2025-09-08]: Aggregate over 10000 hours of data from self-collected robot trajectories open-source datasets and multimodal VQA with unified specifications. [[paper](https://arxiv.org/pdf/2509.11766)] [[website](https://x2robot.com/en/research/68bc2cde8497d7f238dde690)]
- **GR00T N1**[2025-03-28]: Structure training corpora as a data pyramid unifying human videos synthetic data and real robot trajectories using latent and pseudo actions. [[paper](https://arxiv.org/pdf/2503.14734)] [[website](https://research.nvidia.com/labs/gear/gr00t-n1_6/)]
- **Moto**[2025]: Uses unsupervised or self-supervised learning to acquire task-centric latent action representations from videos. [[paper](https://arxiv.org/pdf/2401.03306)] [[website](https://github.com/linhlpv/awesome-offline-to-online-RL-papers)]
- **UniVLA**[2025]: Uses unsupervised or self-supervised learning to acquire task-centric latent action representations from videos. [[paper](https://arxiv.org/pdf/2505.06111)] [[website](https://github.com/OpenDriveLab/UniVLA)]
- **AgiBot**[2025]: Map diverse robot actions into unified physical or latent semantic vectors. [[paper](https://arxiv.org/pdf/2503.06669)] [[website](https://opendrivelab.com/AgiBot-World/)]
- **ViSA-Flow**[2025]: Use unified tokenization semantic alignment or self supervised learning for VLA grounding. [[paper](https://arxiv.org/pdf/2505.01288)] [[website](https://visaflow-web.github.io/ViSAFLOW/)]
- **Humanoid-VLA**[2025]: Use unified tokenization semantic alignment or self supervised learning for VLA grounding. [[paper](https://arxiv.org/pdf/2502.14795)] [[website](https://github.com/AllenXuuu/HumanVLA)]
- **EgoVLA**[2025]: Use MANO hand models and inverse kinematics. [[paper](https://arxiv.org/pdf/2507.12440)] [[website](https://rchalyang.github.io/EgoVLA/)]
- **Dexwild**[2025]: Use MANO hand models and inverse kinematics. [[paper](https://arxiv.org/pdf/2505.07813)] [[website](https://dexwild.github.io/)]
- **CoVLA**[2025]: Provide large scale high fidelity digital environments. [[paper](https://arxiv.org/pdf/2408.10845)] [[website](https://turingmotors.github.io/covla-ad/)]
- **LAPA**[2024]: Uses unsupervised or self-supervised learning to acquire task-centric latent action representations from videos. [[paper](https://arxiv.org/pdf/2410.11758)] [[website](https://latentactionpretraining.github.io/)]
- **RDT-1B**[2024]: Map diverse robot actions into unified physical or latent semantic vectors. [[paper](https://arxiv.org/pdf/2410.07864)] [[website](https://rdt-robotics.github.io/rdt-robotics/)]
- **Cross-Embodied Learning**[2024]: Tokenize visual and proprioceptive inputs for a shared Transformer. [[paper](https://arxiv.org/pdf/2408.11812)] [[website](https://crossformer-model.github.io/)]
- **Re-Mix**[2024]: Adjust sampling weights of heterogeneous data subsets via performance feedback. [[paper](https://arxiv.org/pdf/2408.14037)] [[website](https://github.com/jhejna/remix)]
- **RoboCasa**[2024]: Provide large scale high fidelity digital environments. [[paper](https://arxiv.org/pdf/2406.02523)] [[website](https://robocasa.ai/)]
- **Ego-Exo4D**[2024]: Fuse first person and third person perspectives. [[paper](https://arxiv.org/pdf/2311.18259)] [[website](https://ego-exo4d-data.org/)]
- **RoboMM**[2024]: Use three level semantic alignment for joint training. [[paper](https://arxiv.org/pdf/2412.07215)] [[website](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)]
- **GenAug**[2023]: augment robot data via inpainting or restyling. [[paper](https://arxiv.org/pdf/2302.06671)] [[website](https://genaug.github.io/)]
- **ROSIE**[2023]: provides semantic-level enrichment using VLM priors. [[paper](https://arxiv.org/pdf/2302.11550)] [[website](https://diffusion-rosie.github.io/)]
- **RH20T**[2023]: Enforce strict temporal alignment across sensors. [[paper](https://arxiv.org/pdf/2307.00595)] [[website](https://rh20t.github.io/)]
- **BridgeData v2**[2023]: integrates diverse data types into a standardized format. [[paper](https://arxiv.org/pdf/2308.12952)] [[website](https://rail-berkeley.github.io/bridgedata/)]
- **OXE**[2023]: Aggregate multiple datasets into a single benchmark. [[paper](https://arxiv.org/pdf/2310.08864)] [[website](https://robotics-transformer-x.github.io/)]
- **RT-1**[2022]: Use unified tokenization semantic alignment or self supervised learning for VLA grounding. [[paper](https://arxiv.org/pdf/2212.06817)] [[website](https://robotics-transformer1.github.io/)]
- **CACTI**[2022]: augment robot data via inpainting or restyling. [[paper](https://arxiv.org/pdf/2212.05711)] [[website](https://cacti-framework.github.io/)]
- **Ego4D**[2022]: teaching robots to operate in human environments. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.pdf)] [[website](https://ego4d-data.org/)]
- **EPIC-KITCHENS**[2020]: teaching robots to operate in human environments. [[paper](https://arxiv.org/pdf/2005.00343)] [[website](https://epic-kitchens.github.io/2025)]

</details>

<details>
<summary><strong>4.5.2 Evaluation and Benchmark</strong></summary>

- **PolaRiS**[2025-12-18]: Utilizes neural reconstruction methods to turn video scans into interactive simulation environments and develops a co-training recipe to bridge real-to-sim gaps. [[paper](https://arxiv.org/pdf/2512.16881)] [[website](https://polaris-evals.github.io)]
- **VLSA-AEGIS**[2025-12-09]: Construct a comprehensive safety-critical benchmark SafeLIBERO spanning distinct manipulation scenarios with varying complexities. [[paper](https://arxiv.org/pdf/2512.11891)] [[website](https://vlsa-aegis.github.io/)]
- **ViFailback**[2025-12-03]: Build a large real-world failure VQA dataset and benchmark for diagnosis and correction. [[paper](https://arxiv.org/pdf/2512.02787)] [[website](https://x1nyuzhou.github.io/vifailback.github.io)]
- **RAGNet**[2025-7-31]: construct a large-scale reasoning-based affordance segmentation dataset and propose AffordanceNet for open-world grasping. [[paper](https://arxiv.org/pdf/2507.23734)] [[website](https://github.com/Dexmal-AI/RAGNet)]
- **GENMANIP⭐**[2025-06-12]: Introduce a realistic tabletop simulation platform tailored for policy generalization studies with 200 human curated scenarios. [[paper](https://arxiv.org/pdf/2506.10966)] [[website](-)]
- **ROBOGROUND⭐**[2025-04-30]: Design an automated pipeline to generate simulated manipulation data with diverse objects and instructions. [[paper](https://arxiv.org/pdf/2504.21530)] [[website](https://robo-ground.github.io)]
- **RLBench**[2025]: -. [[paper](https://arxiv.org/pdf/1909.12271)] [[website](https://github.com/stepjam/RLBench)]
- **EUQ**[2025]: Uses a human-assessed multidimensional scoring system to capture process quality beyond binary success. [[paper](https://arxiv.org/pdf/2502.13105v2)] [[website](-)]
- **From Intention to Execution**[2025]: Uses intention-execution gap probing to cover object diversity linguistic complexity and visual-language reasoning. [[paper](https://arxiv.org/pdf/2506.09930)] [[website](https://ai4ce.github.io/INT-ACT/)]
- **InstructVLA**[2025]: Releases SimplerEnv-Instruct with 80 zero-shot tasks. [[paper](https://arxiv.org/pdf/2507.17520)] [[website](https://yangs03.github.io/InstructVLA_Home/)]
- **ManiSkills**[2024]: Contributes standardized APIs and task suites. [[paper](https://arxiv.org/pdf/2410.00425)] [[website](https://github.com/haosulab/ManiSkill)]
- **Ego-Exo4D**[2024]: Introdues synchronized first-third-person recordings. [[paper](https://arxiv.org/pdf/2311.18259)] [[website](https://ego-exo4d-data.org/)]
- **Benchmarking VLAs**[2024]: Uses unified IO metrics and multi robot coverage as a blueprint shifting focus from tasks to metrics. [[paper](https://arxiv.org/pdf/2411.05821)] [[website](https://multinet.ai/static/pages/Multinetv01.html)]
- **LIBERO**[2023]: Introduces the first lifelong-robotics benchmark with standardized metrics. [[paper](https://arxiv.org/pdf/2306.03310)] [[website](https://libero-project.github.io/intro.html)]
- **CALVIN**[2022]: require the execution of long sequences of language-guided operations. [[paper](https://arxiv.org/pdf/2112.03227)] [[website](https://github.com/mees/calvin)]
- **RoboMimic**[2021]: Uses human demonstration data to evaluate offline learning methods and identify challenges in leveraging human-generated data. [[paper](https://arxiv.org/pdf/2108.03298)] [[website](https://github.com/ARISE-Initiative/robomimic)]
- **robosuits**[2020]: Contributes standardized APIs and task suites. [[paper](https://arxiv.org/pdf/2009.12293)] [[website](https://github.com/ARISE-Initiative/robosuite)]

</details>

---

## 📌 Citation

If you find this repository useful, please cite:

```text
@article{xu2025anatomyVLA,
          title   = {An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges},
          author  = {Xu, Chao and Zhang, Suyu and Liu, Yang and Sun, Baigui and Chen, Weihong and Xu, Bo and Liu, Qi and
          Wang, Juncheng and Wang, Shujun and Luo, Shan and Peters, Jan and
          Vasilakos, Athanasios V. and Zafeiriou, Stefanos and Deng, Jiankang},
          journal = {arXiv preprint arXiv:2512.11362},
          year    = {2025},
}
```

## 🙌 Contribution

This is a **living survey**.Contributions, corrections, and new papers are welcome.

Feel free to contact: zsy993115095@gmail.com