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
SYS_SITE=0 bash scripts/setup_env.sh        # 关掉「venv 继承系统 site-packages」
NO_VENV=1 bash scripts/setup_env.sh         # 直接用系统 python，不建 venv
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_env.sh
```

#### 平台已预装 torch 时（推荐，省掉 2.5GB 下载）

```bash
# 0) 先确认平台自带的是什么（必须是 CUDA 版，且 cuda.is_available() 为 True）
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 1) 跳过 torch；脚本会自动用 --system-site-packages 建 venv 去继承它
SKIP_TORCH=1 bash scripts/setup_env.sh
```

版本锁会偏离（本仓锁 `torch 2.6.0+cu118`）。已核过 PyPI 元数据的**硬约束**：
`trl 0.24.0` 完全不锁 torch；`transformers 4.57.1` 只在 extras 里要求 `torch>=2.2`；
`peft 0.14.0` 要求 `torch>=1.13.0` ⟹ **torch 2.5.1 满足全部硬约束**。
脚本第 2 步会当场校验版本号与 `cuda.is_available()`，不满足直接 `exit 1`，
不会带着坏环境往下走。⚠️ 用非 2.6.0 跑出的结果，要在 `docs/EXPERIMENT_LOG.md` 里记一笔 torch 版本。

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

# 域 B：一条命令跑完（融合 60 轮 + SID 2 组 + 三件套 + 对比表，约 2h）
bash scripts/multimodal/run_vg_sid.sh

# 产物：results/sid_e5000/<IandS|VG>/<配置>/
#   gate__init8192/ckpt/selected_model.pth      按碰撞率选出的 ckpt（不要按 loss 选）
#   gate__init8192/sid_raw.npy  ← 默认交付（语义桶）
#   gate__init8192/sid_sk.npy     Sinkhorn 消解版（存档对照）
#   gate__init8192/eval_raw.json / eval_sk.json  三件套评估报告
#   rqkmeans/ 与 compare_rqkmeans.md             RQ-KMeans 对照路线
```

| 环节 | 定版选择 | 说明 |
|---|---|---|
| 融合 | `gate`（门控） | I&S R@10 0.1100（vs 纯文本 **+32.5%**）｜VG 0.2113（**+41.2%**）；随机基线 0.003 / 0.0066 |
| RQ-VAE | 3 层 × 256 码 × 32 维，5000 轮，bs 2048 | 对齐 MiniOneRec 口径 |
| 初始化 | `--init_samples 8192` | I&S 第 0 层死码 **0%**（首 batch 初始化是 32.6%）；VG 为 3.52%（L1/L2 为 0） |
| 碰撞 | **不消解，交付 `sid_raw`** | 碰撞组是同系列规格变体，整桶召回交给排序；详见 `docs/SID_PIPELINE.md` |
| 域 B | **只跑定版 2 组**（RQ-VAE + RQ-KMeans） | 不重做消融（配方已定，重跑=重复结论）；验收判据：ICR≥0.94 / LCP ratio≥150 / R²≥0.6 |

> 想换成 TIGER 系的唯一化口径，直接用同目录下的 `sid_sk.npy` 即可，无需重训。

**指标口径速查**（完整定义与公式见 [`docs/SID_PIPELINE.md` §0.1](SID_PIPELINE.md)）：

| 指标 | 一句话定义 | I&S | VG |
|---|---|---|---|
| R@K | 用融合向量检索 Top-K，命中留出共现伙伴的 query 占比；随机基线 `≈ avg_pos · K / N` | R@10 = 0.1100（基线 0.003） | 0.2113（基线 0.0066） |
| ICR（跨域主判据） | `不同 SID 元组数 / N`，碰撞率 `= 1 − ICR` | 0.9582 | 0.9744 |
| LCP ratio | 近邻对的 SID 公共前缀长度 ÷ 随机对同值（基线 ≡ 1）；**跨域不可横比** | 222.1 | 163.6 |
| 重建 R²（跨域主判据） | `1 − MSE / Var(x)`，其中 `x̂ = Dec(Σ_l C_l[c_l])` | 0.6530 | 0.8691 |
| L0 死码率 | `1 − 第 0 层被用到的码数 / 256` | 0% | 3.52% ⚠️ |

---

## 3. 下载基础模型

```bash
bash scripts/download_base_models.sh              # 幂等，已下过的会 skip；走 ModelScope + sha256 逐位校验
# bash scripts/download_base_models.sh --target all   # 连 teacher Qwen3-1.7B（4.06 GB）
```

| 角色 | 模型 | 体积 |
|------|------|------|
| 学生（SFT + RL） | Qwen3-0.6B | 1.5 GB |
| 教师（OPD 蒸馏） | Qwen3-1.7B | 4.06 GB（按需） |

> ⚠️ `scripts/download_models.sh` 是 **MiniOneRec 原版**（下 `Qwen2.5-0.5B` + 官方 1.5B ckpt），
> 与本项目用的 Qwen3 系列**不是同一个模型**，**不要用**。

---

## 4. 训练与评估（云端 RTX 3090）

> 🔴 入口是**根目录** `sft_run0.sh` / `evaluate_run0.sh` / `rl_run0.sh`。
> ⚠️ `sft.sh` / `rl.sh` / `evaluate.sh` 是 **MiniOneRec 原版**，数据路径写死 `./data/Amazon/...`
> （本仓不存在），**不要用**。原 `*_3090.sh` 三个变体已于 2026-09-19 删除（内容见 `V0_MINIONEREC_TECH_DOC.md`）。

```bash
# SFT 训练（EXP_ID = <域>-<RUN_TAG>[-<任务集>]）
TASKS=T1,T2a,T2b,T3 RUN_TAG=run0 bash sft_run0.sh

# SFT 评估（MODEL_PATH 必须指训练输出目录 —— 那里才有扩展 tokenizer）
MODEL_PATH=outputs/IandS-run0/final_checkpoint bash evaluate_run0.sh

# RL（--model_path 必须是 SFT 产物目录，指回基座会静默崩）
MODEL_PATH=outputs/IandS-run0/final_checkpoint bash rl_run0.sh

# RL 评估
MODEL_PATH=outputs/rl_IandS-run0/final_checkpoint bash evaluate_run0.sh

# 断点续训（同一个 run 的中断续训；换任务集要用 BASE_MODEL，不是 RESUME）
RESUME=outputs/rl_IandS-run0/checkpoint-5280 bash rl_run0.sh
```

**硬串行配方**（分阶段训任务，详见 `SFT_PIPELINE §6.6`）：

```bash
EVAL_BY_EPOCH=True TASKS=T2a,T2b RUN_TAG=hs1 bash sft_run0.sh
EVAL_BY_EPOCH=True TASKS=T1,T3 RUN_TAG=hs2 \
  BASE_MODEL=outputs/IandS-hs1-T2aT2b/final_checkpoint bash sft_run0.sh
```

> 本地 4GB 显存跑不动完整训练，仅用于 SID 构建与小规模冒烟测试。
> 本地跑任何脚本时请关闭 `torch_compile`（Windows + 小卡不可用）。

---

## 5. 一键全流程

```bash
set -euo pipefail
bash scripts/setup_env.sh                       # 平台已装 torch 时加 SKIP_TORCH=1
source .venv/bin/activate                       # ⚠️ 云端无 ./.venv/Scripts/python.exe
python scripts/check_deps.py --strict           # 依赖闸门（AST 扫真实 import）
bash scripts/download_base_models.sh            # 基座权重（ModelScope + sha256 校验）
# 数据需从本地 rsync 上来（不在 git 里）：
#   rsync -avP --relative data/Amazon23/IandS/sft user@<云主机>:~/GenRetrieval/
python scripts/sft/verify_run0_registration.py --domain IandS
TASKS=T1,T2a,T2b,T3 RUN_TAG=run0 bash sft_run0.sh
MODEL_PATH=outputs/IandS-run0/final_checkpoint bash evaluate_run0.sh
```

---

## 6. 项目结构速查

```
GenRetrieval/
├── scripts/
│   ├── setup_env.sh / .ps1     # 环境安装（Linux/云 | Windows）
│   ├── download_base_models.sh # 基座权重下载（ModelScope + sha256 逐位校验，幂等）
│   ├── check_deps.py           # 依赖闸门（--strict = 上云前必过）
│   ├── data/                   # 数据集下载与 SFT 数据构建（prepare_sft_data.py）
│   ├── sft/                    # 链路闸门（verify_run0_registration / probe_constrained_decoding）
│   └── multimodal/             # 图文编码与融合
├── requirements-core.txt       # 可安装的核心依赖（推荐）
├── requirements.txt            # 上游原始依赖（仅供追溯，勿直接装）
├── .wheels/                    # 本地 torch wheel（Windows 重建环境用）
├── config/                     # 提示词模板单一真源（prompt_templates.json）
├── rq/                         # SID：RQ-VAE / RQ-KMeans
├── minionerec_trainer.py       # 自定义 GRPO trainer（依赖 trl 内部 API）
├── sft_run0.sh                 # SFT 训练入口
├── rl_run0.sh                  # RL 训练入口
├── evaluate_run0.sh            # 评估入口
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
