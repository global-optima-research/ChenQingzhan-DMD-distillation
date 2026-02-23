# CLAUDE.md — AI Assistant Context Configuration

## Developer Info

- **Name:** 陈庆展 / Chen Hing Chin
- **Email:** hcchenab@connect.ust.hk
- **Branch:** `task3-dev-ChenHingChin`
- **Role:** Task 3 — DMD Distillation & Acceleration (Progressive Distillation & Discriminator)
- **Team:** Task 3 subgroup (4 members: Li Yijia, Chen Qingzhan, Sze Long, Qiu Zhangxizi)
- **Group Leaders:** Li Zhiying, Jacky

### Task 3 Objectives for Chen Qingzhan

**Phase 0 (Week 1-6): Reproduce existing distillation methods**
- Reproduce DMD2 on Wan2.1 using NVIDIA FastGen framework
- Compare distillation methods: DMD2 vs ECT vs Consistency Distillation
- Record convergence speed, generation quality, and framework evaluation
- Submit reproduction report

**Phase 1+ (Week 7-16): Progressive distillation & Discriminator**
- Design and train a 3D Video Discriminator adapted for video editing (product region vs background)
- Stage 1: 50-step → 16-step distillation (~20K training steps)
- Stage 2: 16-step → 8-step distillation (~20K training steps)
- Stage 3: 8-step → 4-step distillation (~20K training steps)
- Quality gate control at each stage (CLIP-I / FVD thresholds)
- EMA (Exponential Moving Average) update strategy for Student model

---

## Project Context

**Project:** PVTT (Product Video Template Transfer) — IP-2026-Spring
**Team:** 14 members (4 management + 10 R&D), 16-week timeline
**Organization:** Global Optima Research

### Overall Pipeline

```
Task 1: Dataset Construction (3 people)
  → 20K+ high-quality training triplets (source_video, reference_image, edited_video, mask, caption)
  → Key output: Week 8-9

Task 2: Teacher Model Training (3 people)
  → Modify Wan2.1/2.2 DiT architecture for dual-condition input (source video + reference image)
  → LoRA fine-tuning → Progressive training (1.3B → 14B)
  → Target: CLIP-I > 0.85, FVD < 100
  → Key output: Week 12 (Teacher Model delivery)

Task 3: DMD Distillation & Acceleration (4 people) ← Chen Qingzhan is here
  → Compress 50-step Teacher Model to 4-step Student Model via DMD
  → 4-8x inference speedup, quality loss < 5%
  → Key output: Week 16 (Final 4-step Student Model)
```

### Key Dependencies

- Task 2 delivers Teacher Model at Week 12 → Task 3 uses it for distillation
- Task 3 must first reproduce existing distillation codebases (Phase 0) before deep R&D

---

## Task 3 Specs — DMD Distillation & Acceleration

### Core Technical Details

- **Base Model:** Wan2.1 (1.3B / 14B) — DiT-based video generation model by Alibaba
- **Distillation Method:** Distribution Matching Distillation (DMD / DMD2)
- **Target:** 50 steps → 4 steps progressive compression
- **Quality Threshold:** CLIP-I > 0.80, FVD < 120 for Student Model

### Key Codebases

| Codebase | GitHub | Priority |
|----------|--------|----------|
| NVIDIA FastGen | NVlabs/FastGen | Highest (primary reproduction target) |
| FastVideo | hao-ai-lab/FastVideo | Highest |
| distill_wan2.1 | azuresky03/distill_wan2.1 | High |
| LightX2V | ModelTC/LightX2V | High |
| CausVid | tianweiy/CausVid | Reference |

### Key Technologies

- **DMD2:** Distribution Matching Distillation v2 — adversarial training with GAN loss / f-divergence
- **3D Video Discriminator:** Adapted for video editing (product region vs background region awareness)
- **EMA:** Exponential Moving Average for stable Student model updates
- **Progressive Distillation:** 50 → 16 → 8 → 4 steps, with quality gates at each stage
- **Training Infrastructure:** FastVideo/FastGen + DeepSpeed ZeRO-2/3

### Evaluation Metrics

| Metric | Target (Student) |
|--------|-----------------|
| CLIP-I (Identity Preservation) | > 0.80 |
| FVD (Video Quality) | < 120 |
| Inference Steps | 4 steps |
| Speedup | 4-8x over Teacher |

---

## Instructions

### Language Rules (STRICT)
- **Always communicate with the user in Chinese (中文) unless they initiate in English.**
- **Keep all code, code comments, configuration files, system files, and git commit messages in English.**

### Development Guidelines
- This is a research/ML project focused on video diffusion model distillation.
- The primary development branch is `phase2-dev-ChenHingChin`.
- The main working directory for Chen Qingzhan's files is `03-dmd-distillation/`.
- When writing code, follow PyTorch and HuggingFace conventions.
- Use type hints in Python code where appropriate.
- Prefer clear, readable code over clever optimizations unless performance-critical.

### Project File Structure
```
IP-2026-Spring/
├── README.md                    # Project-wide Video Editing survey
├── TASK-ASSIGNMENT.md           # Detailed 10-person task assignment
├── PVTT-Presentation.html       # PVTT presentation
├── CLAUDE.md                    # This file — AI context configuration
├── 01-dataset-construction/     # Task 1: Dataset construction
├── 02-teacher-model-training/   # Task 2: Teacher model training
│   └── README.md                # Teacher model technical survey
└── 03-dmd-distillation/         # Task 3: DMD distillation & acceleration
    ├── README.md                # DMD distillation technical survey
    └── ChenHingChinReadMe.md    # Chen Qingzhan's personal work board
```
