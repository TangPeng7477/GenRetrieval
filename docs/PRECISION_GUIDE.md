# 混合精度速查：bf16 / fp16 / fp32

> 2026-09-17 ｜ 这篇回答三个问题：**① 精度之间的区别是什么 ② 对本项目有什么影响 ③ V100 能不能跑**。
> 配图：[`figures/float_formats.svg`](figures/float_formats.svg)（位分配 + 数值范围）
> 相关：`SFT_PIPELINE §3.10`（本地容量实测）· `RL_PIPELINE §6`（RL 成本）

---

## 0. 三句话结论

1. **精度 = 位怎么分**：指数位宽决定**范围**（能表示多大），尾数位宽决定**精度**（能表示多细）。
   fp32 是 `1+8+23`，bf16 是 `1+8+7`，fp16 是 `1+5+10` —— 同样 16 bit，两者把预算花在了不同地方。
2. **LLM 训练默认 bf16**（Ampere 及以后）：它和 fp32 共享 8 位指数，范围一样，所以**不需要 loss scaling**。
   fp16 指数只有 5 位，范围窄到 `6.1e-5 … 6.55e4`，梯度会下溢成 0，必须靠 loss scaling 救。
3. 🔴 **V100 没有 bf16**（Volta 架构），必须显式切 `fp16`。**而且它不会自动报错** —— 详见 §5.1。

---

## 1. 五种格式（`[公开规格]`，非本项目实测）

| 格式 | 位分配 (s/e/m) | 最大正规数 | 尾数间隔 (2⁻ᵐ) | 谁支持 |
|---|---|---|---|---|
| **fp32** | 1 + 8 + 23 = 32 | 3.40e38 | 1.2e-7 | 所有 GPU（CUDA core） |
| **tf32** | 1 + 8 + 10 = 19（存 32 位） | 3.40e38 | 9.8e-4 | Ampere+，矩阵乘自动启用 |
| **bf16** | 1 + 8 + 7 = 16 | 3.39e38 | **7.8e-3** | Ampere+（sm_80） |
| **fp16** | 1 + 5 + 10 = 16 | **65504** | 9.8e-4 | Volta+（含 V100） |
| **fp8** (E4M3) | 1 + 4 + 3 = 8 | 448 | 0.125 | Hopper+ |

**看指数位宽就够了**：8 位指数 ⇒ 量级到 `10^±38`；5 位指数 ⇒ 只能到 `10^±4.8`。
这解释了图中 Panel ②：fp32 / bf16 的范围宽到画出坐标轴，fp16 只有中间窄窄一条。

**尾数位宽决定"分辨能力"**：bf16 的 7 位尾数意味着 `1.0` 之后下一个能表示的数是 `1.0078`；
fp32 是 `1.0000001`。所以 bf16 不适合做累加（误差会滚），但**足够表示权重和梯度**（它们本来就是噪声）。

---

## 2. 为什么训练用 bf16 而不是 fp16

一句话：**fp16 的范围不够装梯度。**

`[看图 Panel ②]` 梯度/激活的常见量级落在 `1e-8 … 1e-4`，而 fp16 的最小正规数是 `6.1e-5`
—— 整段落在范围之外，直接算会**下溢成 0**，反向传播拿不到信号。

两条出路：

| 路线 | 做法 | 代价 |
|---|---|---|
| **fp16 + loss scaling** | 反传前把 loss 乘一个大系数（如 2048），让梯度整体上移进 fp16 范围，更新前再除回去 | 需要 `GradScaler`；系数要动态调；偶发 inf 会跳步 |
| **bf16（不用 scaling）** | 指数和 fp32 一样宽，梯度天然落得进去 | 精度只有 7 位尾数（对本任务无影响） |

所以：**Ampere+ 一律 bf16，V100 只能 fp16 + scaling。** 这不是偏好，是硬件决定。

---

## 3. 对本项目的具体影响

### 3.1 显存（影响最大）

显存由三块构成，**优化器状态通常是大头**：

| 项 | fp32 训练 | bf16 训练 | bf16 + 8bit AdamW |
|---|---|---|---|
| 权重 | 4 B/param | 2 B/param | 2 B/param |
| 梯度 | 4 B/param | 2 B/param | 2 B/param |
| AdamW 状态（两个动量） | **8 B/param** | **8 B/param**（fp32） | **2 B/param** |
| 合计 | 16 B/param | 12 B/param | 6 B/param |

🔴 **一个反直觉的点**：`bf16=True` **不会**把优化器状态也降到 bf16 —— AdamW 的动量始终是 fp32
（更新量太小，低精度会直接抹掉）。所以 bf16 只省了权重和梯度那 4 B，**省不了 8 B 的优化器状态**。
这就是为什么换 8-bit AdamW 的收益远大于换 dtype。

另外，`Trainer` 在 `fp16=True` 时会走 AMP，**要求 fp32 主权重**（详见 §5.2），
所以 fp16 模式的显存反而是 bf16 的约 **2 倍** —— 见 §3.4 实测。

### 3.2 算力

Tensor core 只在 fp16/bf16 下生效；fp32 走 CUDA core。以本项目 0.6B 模型 + `cutoff_len=320` 的规模，
**实际瓶颈几乎总是显存而不是算力**（`SFT_PIPELINE §3.10` 实测：把优化器状态拿掉，单步从 1.140 s
降到 0.279 s，4.1 倍差距全在显存换页上）。所以选卡的优先级是 **显存 > 带宽 > TFLOPS**。

### 3.3 数值（逐项检查过我们的代码）

| 我们的代码 | 低精度下有风险吗 | 依据 |
|---|---|---|
| `LogitProcessor.py` 的加性 mask（`float('-inf')` + `log_softmax`） | **无**。`-inf + 有限值 = -inf`，`exp(-inf)=0`，不会出现"全 -inf 求 softmax → NaN" | 读代码；掩码至少放行 1 个 token |
| `resize_token_embeddings` 的新行初始化（std **0.0094**） | **无**。bf16 尾数 7 位对 ~1e-2 量级分辨力足够；fp16 更精细 | `SFT_PIPELINE §6.4(3)` |
| SID 是三段离散 token（`[a,b,c,\n,EOS]`） | **无**。离散取值不涉及数值精度 | 模型只做分类 |
| GRPO 的 advantage（组内归一化 `(r-μ)/σ`） | **无**。`minionerec_trainer.py:954` 显式用 `dtype=torch.float32` 建奖励张量；组内 σ 很小也是 fp32 除法 | 读代码 |
| `semantic` 奖励的余弦相似度 | **偏低风险**。fp16 的 ~1e-3 分辨力对"当作奖励信号"够用，但它会放大近邻差异 —— 真要上 semantic 奖励建议 fp32 算 | 推断，未实测 |
| 梯度下溢（fp16 模式） | 🔴 **有**。这是 fp16 的固有短板，靠 `GradScaler` 兜底 | 见 §5.2 实测 |

### 3.4 实测数字（3050Ti 4 GB · LoRA r=32 + `modules_to_save` · cutoff 320 · micro=1）

基准脚本 `scripts/sft/bench_local_training.py`：

| 配置 | 峰值显存 | 单步 | 备注 |
|---|---:|---:|---|
| `bf16` + `adamw_torch` | **4.26 GiB** | 1.140 s | 超过物理 4.00 ⟹ Windows WDDM 换页 |
| `bf16` + `adamw_bnb_8bit` | 3.05 GiB | 0.332 s | 不换页 |
| `bf16` + `sgd`（诊断） | 2.44 GiB | 0.279 s | 无优化器状态 |
| **`fp16` + `adamw_torch`** | **8.23 GiB** | 5.468 s | fp32 主权重导致约 **2 倍**显存；`GradScaler scale=2048` 稳定 |

⟹ **fp16 模式的可训练规模约为 bf16 的一半**，这一点在评估 V100 时必须计入。

---

## 4. 硬件支持矩阵

| GPU | 架构 (sm) | bf16 | tf32 | fp16 TC | 显存 | FP32 | FP16/BF16 (TC) |
|---|---|:---:|:---:|:---:|---|---:|---:|
| **Tesla V100** | Volta **sm_70** (2017) | ❌ | ❌ | ✅ | **16 / 32 GB** HBM2 | 15.7 | ~125 (fp16) |
| RTX 3090 | Ampere sm_86 (2020) | ✅ | ✅ | ✅ | 24 GB GDDR6X | 35.6 | ~71 (fp16) |
| A100 | Ampere sm_80 | ✅ | ✅ | ✅ | 40 / 80 GB | 19.5 | ~312 (bf16) |
| H100 | Hopper sm_90 | ✅ | ✅ | ✅ + fp8 | 80 GB | 67 | ~989 (bf16) |

> V100 一行（bf16 ❌ / tf32 ❌ / 16·32 GB / 15.7 FP32 / ~125 FP16-TC）来自公开规格检索核实；
> 其余行为常见公开规格，**仅作量级参考**，未逐条核验。
>
> ⚠️ 注意 **V100 的 fp16 tensor 峰值（~125）反而高于 3090（~71）**，但这是 Gen1 tensor core、
> 峰值在窄算子/小 batch 下打不满；且它没有 tf32、带宽 900 GB/s 略低于 3090 的 936 GB/s。
> **别只看 TFLOPS 选卡** —— 对本项目，显存容量才是决定性的。

---

## 5. V100 能不能跑

**能，但有三个前提**：① 必须用 `fp16`；② 显存要 32 GB 才够全参 SFT；③ 没有 flash-attn（已自动降级）。

### 5.1 🔴 为什么不会自动报错（这条最坑）

`transformers` 在 `bf16=True` 时会检查支持性（`.venv/Lib/site-packages/transformers/training_args.py:1742-1747`），
不支持就抛 `ValueError: ... You need Ampere+ GPU with cuda>=11.0`。看起来 V100 会被友好拦下 —— **但不会**：

```python
# .venv/Lib/site-packages/transformers/utils/import_utils.py:637
def is_torch_bf16_gpu_available() -> bool:
    ...
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()      # :644 ← 默认参数 including_emulation=True
```

```python
# .venv/Lib/site-packages/torch/cuda/__init__.py:132（本环境 torch 2.6.0+cu118）
def is_bf16_supported(including_emulation: bool = True):
    ...
    if cuda_version and major >= 8:                # ← V100 的 major = 7，这里过不去
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)    # :159 ← 退到这个「弱探测」

def _check_bf16_tensor_supported(device):   # :163
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)   # ← 只分配，不做任何运算
        return True
    except Exception:
        return False
```

**这个探测只分配一个 bf16 张量、不做任何算术** —— 任何能分配显存的卡都会返回 `True`。
⟹ V100 上不会拿到那句友好报错，会**一路用 bf16 跑下去**，然后在某个 kernel 上撞
`not implemented for 'BFloat16'`，或者静默走软件模拟变慢。

`[代码核实]` 上述判据逻辑（本机 torch 2.6.0 已实际打印其源码与返回值）。
`[推断]` V100 上的具体表现（我无 V100 可实测）——**但结论不依赖推断：显式指定 `fp16` 就绕开了整条路径。**

### 5.2 fp16 模式的两个坑（都已实测修好）

**坑 1：不能把模型加载成 fp16。** `Trainer` 在 `fp16=True` 时挂 `GradScaler`，
而 `GradScaler` 拒绝 unscale fp16 梯度 —— `[实测]` 直接报：

```
ValueError: Attempting to unscale FP16 gradients.
```

正确形态是 AMP 标准做法：**fp32 主权重 + autocast 把计算降到 fp16**。
代价是权重显存翻倍（0.6B: 1.2 GB → 2.4 GB），换来数值稳定。已在 `sft.py` / `rl.py` 里按这个映射实现。

**坑 2：`LogitProcessor` 的 `-inf` 掩码**在 fp16 下依然成立（`-inf + 有限 = -inf`，`exp(-inf)=0`），
不会产生 NaN —— 已读代码确认，**不需要改**。

### 5.3 显存够不够（推算，`[设计]`）

全参 SFT Run-0（Qwen3-0.6B + 词表扩展到 152,437 ⟹ 全模型 **918 M** 参数）：

| 项 | fp32 主权重（fp16 模式必需） |
|---|---:|
| 权重 | 918 M × 4 B = **3.7 GB** |
| 梯度 | 918 M × 4 B = **3.7 GB** |
| AdamW 状态（2 个动量） | 918 M × 8 B = **7.3 GB** |
| **固定开销小计** | **≈ 14.7 GB** |
| 激活（micro=4 / cutoff 320） | ~2–3 GB |
| **合计** | **≈ 17–18 GB** |

⟹ **V100 16 GB 跑不了全参 Run-0**；**V100 32 GB 可以**。
16 GB 上的出路（按性价比排序）：
1. **LoRA**（`USE_LORA=True`）—— 可训参数从 918 M 降到 321 M，实测 4 GB 卡就能跑 bf16
2. **8-bit 优化器** —— `optim="paged_adamw_8bit"` 把 7.3 GB 压到 ~1.8 GB
3. **换回 3090 / A100**（如果价格相近，24 GB 的 3090 更省心）

### 5.4 已经改好了什么

原来 `bf16` 在三处**硬编码**（`sft.py` 2 处、`rl.py` 2 处、`evaluate.py` 1 处），V100 上无法使用。
现在全部参数化：

| 位置 | 改动 |
|---|---|
| `sft.py` | 新增 `--precision`；`_dt/_bf16/_fp16` 映射；`dtype=_dt` + `bf16=_bf16, fp16=_fp16` |
| `rl.py` | 同上（`GRPOConfig` 的 `bf16/fp16`） |
| `evaluate.py` | 新增 `--precision`；**推理侧 fp16 直接加载 fp16 权重**（没有 GradScaler，不需要 fp32 主权重） |
| `sft_run0.sh` / `rl_run0.sh` | 新增 `PRECISION` 环境变量并透传 |
| `scripts/sft/bench_local_training.py` | 新增 `--dtype`，并**手写 GradScaler/autocast** 以真实复现 Trainer 的 AMP 行为 |

---

## 6. 怎么用

```bash
# 3090 / A100 / 30 系 / 40 系（Ampere+）：默认就是 bf16，不用管
bash sft_run0.sh

# V100 / 任何 Volta、Turing 卡：显式切 fp16
PRECISION=fp16 bash sft_run0.sh
PRECISION=fp16 bash rl_run0.sh
PRECISION=fp16 bash evaluate_run0.sh

# 调试用 fp32（最慢但最稳）
PRECISION=fp32 bash sft_run0.sh

# 实测某个卡上的显存/速度（不训练，不进生产）
./.venv/Scripts/python.exe scripts/sft/bench_local_training.py --dtype fp16
```

`PRECISION` 的映射（`sft.py` / `rl.py`）：

| `PRECISION` | 权重加载 | `bf16` | `fp16` | GradScaler |
|---|---|:---:|:---:|:---:|
| `bf16`（默认） | bf16 | ✅ | ❌ | 不需要 |
| `fp16` | **fp32** | ❌ | ✅ | ✅ 自动 |
| `fp32` | fp32 | ❌ | ❌ | ❌ |

---

## 7. 一句话给 V100 的结论

**能跑，但要选 32 GB 版本，并且必须 `PRECISION=fp16`**（16 GB 只能走 LoRA 或 8-bit 优化器）。
如果 24 GB 的 3090 价格相近，**优先 3090** —— 对本项目，显存容量比 fp16 峰值吞吐重要得多。
