# QUICKSTART

从零到跑通 GenRetrieval 全流程。**第一步就是环境依赖安装**，照着抄即可。

- 云平台 / Linux 全新环境：约 **2–4 分钟**
- Windows 本地：约 **5–10 分钟**（依赖本地 torch wheel，见下）

---

## 0. 环境要求

| 项目 | 云端训练（正式） | 本地迭代（可选） |
|------|------------------|------------------|
| GPU | RTX 3090 24GB（或同级） | RTX 3050 Ti 4GB |
| Python | 3.10 – 3.12 | 3.11 |
| CUDA | 11.8（驱动 ≥ 520） | 11.8 |
| 磁盘 | ≥ 25GB | ≥ 20GB |

依赖版本已锁死，请勿随意升级：
- **torch 2.6.0 + cu118**：`trl 0.24.0` 需要 torch ≥ 2.4
- **trl 0.24.0**：`minionerec_trainer.py` 依赖 `trl.trainer.utils.selective_log_softmax` 等内部 API，降版即崩
- **numpy 1.26.3（必须 < 2）**：torch 2.6 / pandas 2.2 均按 1.x ABI 编译

---

## 1. 环境安装（核心步骤）

### 1.1 云平台 / Linux / WSL —— 一键

```bash
git clone <your-repo-url> GenRetrieval && cd GenRetrieval
bash scripts/setup_env.sh
```

可选开关：

```bash
CUDA=cu121 bash scripts/setup_env.sh        # 换 CUDA 版本
SKIP_TORCH=1 bash scripts/setup_env.sh      # 平台已预装 torch，跳过 2.5GB 下载
NO_VENV=1 bash scripts/setup_env.sh         # 直接用系统 python，不建 venv
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_env.sh
```

### 1.2 云平台 / Linux —— 手动分步（等价于上面的脚本）

```bash
# 1) 虚拟环境
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel -i https://mirrors.cloud.tencent.com/pypi/simple/

# 2) PyTorch（cu118，约 2.5GB）
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 3) 核心依赖
pip install -r requirements-core.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
```

> `requirements.txt` 是上游原始文件，**不要直接 `pip install -r requirements.txt`**。
> 其中 `torchrec` / `fbgemm_gpu` / `triton` / `deepspeed` / `nvidia-*-cu1x` 在多数环境装不上或纯属冗余，
> 已裁剪为 `requirements-core.txt`。原因见该文件头部注释。

### 1.3 Windows 本地 —— 一键

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# 全新重建：
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Recreate
```

该脚本从 `.wheels\` 里的本地 wheel 安装 torch（**不做 2.6GB 下载**），并**刻意绕开 pip 的卸载流程**——
原因见 [§7 故障排除](#7-故障排除)。

### 1.4 Windows 本地 —— 手动分步

```powershell
cd D:\Codings\GenRetrieval
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .wheels\torch-2.6.0+cu118-*.whl .wheels\torchvision-0.21.0+cu118-*.whl
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
```

> **保持 `.wheels\` 目录**（约 2.6GB）。它让 Windows 环境可以秒级重装，是本地开发的关键资产。

### 1.5 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, trl, peft, datasets, faiss, ot; print('stack ok')"
```

期望输出：`2.6.0+cu118 True` 与 `stack ok`。
`setup_env.sh` / `setup_env.ps1` 末尾会自动跑一遍完整自检。

---

## 2. 数据准备

项目使用 Amazon Reviews 2023 的**两个子类，并行独立**（不合并）：

| 数据集 | 简称 | 规模 |
|--------|------|------|
| Industrial_and_Scientific | `IandS` | 25,847 items / 46,341 users |
| Video_Games | `VG` | 25,611 items / 91,540 users |

```bash
# 下载原始分片（走 hf-mirror.com）
bash scripts/data/download_amazon23.sh

# 预处理（5core/timestamp 分片 → 交互序列 + item 元数据）
python scripts/data/prepare_amazon23.py --category Industrial_and_Scientific --short IandS
python scripts/data/prepare_amazon23.py --category Video_Games            --short VG

# 图文多模态
python scripts/multimodal/download_images.py --short IandS
python scripts/multimodal/download_images.py --short VG
python scripts/multimodal/encode_text.py  --short IandS
python scripts/multimodal/encode_image.py --short IandS
python scripts/multimodal/fuse_embeddings.py --short IandS
```

各脚本均支持 `--help`。产出落盘于 `data/Amazon23/<short>/`。
数据细节与统计口径见 `docs/DATASET.md`。

### 2.5 SID 构建（本地 4GB 卡即可跑，无需上云）

```bash
# 1) 两模态编码（必须串行：4GB 卡上并行会显存 thrashing）
bash scripts/multimodal/encode_all.sh IandS

# 2) 门控融合 60 轮（InfoNCE，共现对监督）→ emb/long/emb_fused_gate_e60.npy
bash scripts/multimodal/fuse_long.sh IandS 60 5

# 3) RQ-VAE 训练 + SID 导出 + 三件套评估（定版配方，约 1h）
RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 \
  bash scripts/multimodal/run_sid_exp.sh IandS 5000 "gate"

# 产物：results/sid_e5000/IandS/gate/
#   ckpt/selected_model.pth        按碰撞率选出的 ckpt（不要按 loss 选）
#   sid_raw.npy  ← 默认交付（语义桶，ICR 0.9582）
#   sid_sk.npy      Sinkhorn 消解版（ICR 0.9997，存档对照）
#   eval_raw.json / eval_sk.json    三件套评估报告
```

| 环节 | 定版选择 | 说明 |
|---|---|---|
| 融合 | `gate`（门控） | R@10 0.1100，比纯文本 +33% |
| RQ-VAE | 3 层 × 256 码 × 32 维，5000 轮，bs 2048 | 对齐 MiniOneRec 口径 |
| 初始化 | `--init_samples 8192` | 第 0 层死码 0%（首 batch 初始化是 32.6%） |
| 碰撞 | **不消解，交付 `sid_raw`** | 碰撞组是同系列规格变体，整桶召回交给排序；详见 `docs/SID_PIPELINE.md` |

> 想换成 TIGER 系的唯一化口径，直接用同目录下的 `sid_sk.npy` 即可，无需重训。

---

## 3. 下载基础模型

```bash
bash scripts/download_models.sh                # 全部
bash scripts/download_models.sh --base-only    # 仅 Qwen3-0.6B（学生）
bash scripts/download_models.sh --mirror       # 强制 hf-mirror.com
```

| 角色 | 模型 |
|------|------|
| 学生（SFT + RL） | Qwen3-0.6B |
| 教师（OPD 蒸馏） | Qwen3-1.7B |

---

## 4. 训练与评估（云端 RTX 3090）

```bash
# SFT（语义 token 初始化 + 课程学习 + QLoRA）
bash sft_3090.sh

# SFT 评估
MODEL_PATH=./outputs/sft_IandS_3090/final_checkpoint bash evaluate_3090.sh

# RL（分层奖励 / OPD 双轨）
MODEL_PATH=./outputs/sft_IandS_3090/final_checkpoint bash rl_3090.sh

# RL 评估
MODEL_PATH=./outputs/rl_IandS_3090/final_checkpoint bash evaluate_3090.sh

# 断点续训
RESUME=./outputs/rl_IandS_3090/checkpoint-5280 bash rl_3090.sh
```

> 本地 4GB 显存跑不动完整训练，仅用于 SID 构建与小规模冒烟测试。
> 本地跑任何脚本时请关闭 `torch_compile`（Windows + 小卡不可用）。

---

## 5. 一键全流程

```bash
set -euo pipefail
bash scripts/setup_env.sh
bash scripts/download_models.sh --mirror
bash scripts/data/download_amazon23.sh
python scripts/data/prepare_amazon23.py --category Industrial_and_Scientific --short IandS
bash sft_3090.sh
MODEL_PATH=./outputs/sft_IandS_3090/final_checkpoint bash evaluate_3090.sh
```

---

## 6. 项目结构速查

```
GenRetrieval/
├── scripts/
│   ├── setup_env.sh            # 环境安装（Linux/云）
│   ├── setup_env.ps1           # 环境安装（Windows）
│   ├── download_models.sh      # 模型下载
│   ├── data/                   # 数据集下载与预处理
│   └── multimodal/             # 图文编码与融合
├── requirements-core.txt       # 可安装的核心依赖（推荐）
├── requirements.txt            # 上游原始依赖（仅供追溯，勿直接装）
├── .wheels/                    # 本地 torch wheel（Windows 重建环境用）
├── rq/                         # SID：RQ-VAE / RQ-KMeans
├── minionerec_trainer.py       # 自定义 GRPO trainer（依赖 trl 内部 API）
├── sft_3090.sh / rl_3090.sh    # 训练入口
├── evaluate_3090.sh            # 评估入口
└── docs/                       # 方案 / 实验日志 / 数据说明 / 本文件
```

---

## 7. 故障排除

### 7.1 `pip install` 卡住不动（Windows 特有，最常见）

**症状**：pip 输出停在 `Uninstalling ...` 或毫无输出，几十分钟不结束。

**根因**：pip 卸载多 GB 包（torch ≈ 4 万文件）时逐个删文件，Windows 的 Defender 实时扫描 + 文件索引
会让每删一个文件都产生可观开销，整体退化为「几十分钟删一个包」。
`sympy` / `torch` 这类包尤其明显。

**判断**：

```bash
ls -d .venv/Lib/site-packages/~*      # 若出现 ~torch / ~sympy / ~pip，说明正卡在卸载
```

`~xxx` 是 pip 卸载时的临时改名目录（`xxx` → `~xxx` → 删除）。它存在 = 卸载未完成。

**处理**：

```powershell
# 1) 找到并杀掉卡住的进程
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, CommandLine
Stop-Process -Id <PID> -Force

# 2) 把残留挪走（同盘瞬间完成，比删快得多）
Move-Item .venv\Lib\site-packages\~torch D:\tmp\venv-trash\ -Force

# 3) 重装（此时无包可卸，pip 不会再卡）
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
```

**根治**：`-Recreate` 重建环境（见 §1.3），或让 pip 始终「只装不卸」——本项目的 `setup_env.ps1` 已按此设计。

### 7.2 `module 'numpy' has no attribute 'ndarray'`

pip 卸载 numpy 到一半被中断（目录被删、元数据残留）。
删除 `.venv/Lib/site-packages/numpy*` 后重装即可；彻底方案是 `-Recreate`。

### 7.3 下载超时 / 极慢

- HuggingFace 一律走 `hf-mirror.com`（脚本内已默认）
- pip 镜像优先级实测：**腾讯云 > 阿里云 > pypi.org**；
  清华 `pypi.tuna.tsinghua.edu.cn` 与中科大 `mirrors.ustc.edu.cn/pypi` 在部分网络下返回 **403**
- **>100MB 的下载不要在受限沙箱/代理环境中执行**，实测限速可达 51 倍

### 7.4 `trl` 导入报 `selective_log_softmax` 找不到

trl 版本被换掉了。恢复 `trl==0.24.0`：

```bash
pip install trl==0.24.0 -i https://mirrors.cloud.tencent.com/pypi/simple/
```

### 7.5 CUDA 不可用

```bash
python -c "import torch; print(torch.__version__)"   # 必须是 2.6.0+cu118，不能是 2.6.0+cpu
```

若显示 `+cpu`，说明装成了 CPU 版，用 §1.1 / §1.2 的 pytorch 专用 index 重装。

---

## 8. 云平台安装为什么比本地快这么多？

| 因素 | 本地 Windows | 云平台 Linux |
|------|--------------|--------------|
| pip 卸载大包 | **几十分钟**（Defender + 索引拦截每次文件删除） | **几秒**（普通 unlink，无实时扫描） |
| CUDA 运行时 | 需从 pytorch.org 下 2.5GB | 镜像常预装 torch，可 `SKIP_TORCH=1` |
| 网络 | 代理/沙箱限速，实测最高差 51 倍 | 内网 pypi 镜像，直连高速 |

结论：**云上全新装环境 2–4 分钟**。本地慢的唯一原因就是 §7.1 的卸载问题，跟下载速度无关。

---

_环境版本与踩坑记录同步维护于 `docs/EXPERIMENT_LOG.md`。_
