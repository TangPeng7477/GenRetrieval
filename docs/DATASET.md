# 数据集档案：Amazon Reviews 2023 → IandS + Video_Games（双域并行）

> 用途：复习 + 面试问答的**数据侧底稿**。所有数字均为本机实测（非论文引用、非估计）。
> 实测日期：2026-09-11 ｜ 复现脚本：`scripts/data/download_amazon23.sh`、`scripts/data/prepare_amazon23.py`
> 上游统计口径：官方 `McAuley-Lab/Amazon-Reviews-2023` 的 `benchmark/5core/timestamp` 分片
>
> **本项目采用「双域并行」设定**：两个类目**各自独立成集、独立跑完整 pipeline**，**不做合并**
> （合并只在脚本里保留为一个默认关闭的可选开关）。目的：用同一套 pipeline 在两个域距离最远的
> 类目上各验证一遍，证明方法**不依赖单一类目的偶然性** → 这就是"泛化性"的可证伪证据。
> - 域 A：`Industrial_and_Scientific`（简称 **IandS**）—— 工业器材，规格/参数主导
> - 域 B：`Video_Games`（简称 **VG**）—— 娱乐内容，封面艺术主导
>
> 两域**并行对照表**见 §13。

---

## 1. 为什么换数据（这一段是面试开场的关键）

V0（MiniOneRec 复刻版）用的是 **Amazon Reviews 2018**，存在三个硬伤：

| # | 问题 | 具体表现 | 后果 |
|---|---|---|---|
| 1 | **数据陈旧** | 数据时间窗 2016-10 ~ 2018-11，距今 8 年 | 商品目录、品类结构、用户行为模式与现在脱节；面试时无法回答"为什么不用新数据" |
| 2 | **纯文本，信息维度单一** | 只有 `title` + `description` | 外观/风格/规格这类**视觉强相关**的品类（工具、仪器、耗材）无法区分；多模态召回无从谈起 |
| 3 | **SID 目标空间过小** | I&S 仅 3,686 个 item | 256³ = 16.7M 的 SID 空间里只用了 3,686 个，**codebook 利用率天然极低**，量化实验的结论不可迁移到工业规模 |

本次升级换成 **Amazon Reviews 2023**，一举解决三点：数据新（至 2023-08）、含**商品图像 + 结构化字段**、item 规模放大到 25,847（**7 倍**）。

---

## 2. 数据源

| 项目 | 内容 |
|---|---|
| 数据集 | **Amazon Reviews 2023**（McAuley-Lab） |
| 论文 | *Bridging Language and Items for Retrieval and Recommendation* (arXiv:2403.03952) |
| 主页 | https://amazon-reviews-2023.github.io/ |
| HF 仓库 | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| 镜像（本机唯一可达） | `https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023` |
| 规模（全量） | 571.54M 评论 / 48.19M 商品 / 33 个类目 / 时间跨度 1996-05 ~ 2023-09 |
| 许可 | 仅限学术研究使用 |

> ⚠️ 实测：`huggingface.co` 国内直连 12s 超时（http=000），`hf-mirror.com` 200/1.1s。所有下载脚本已硬编码 mirror。

**数据集的三种分片形态**（一定要讲清楚你用的是哪一种）：

| 形态 | 路径 | 特点 | 本项目是否使用 |
|---|---|---|---|
| 全量评论 | `raw/review_categories/*.jsonl` | 每类几十 GB | ❌ 不用 |
| 全量 metadata | `raw/meta_categories/meta_*.jsonl` | I&S 为 1.13 GB，**图像 URL 的唯一来源** | ✅ 用 |
| **官方 5-core 基准分片** | `benchmark/5core/timestamp/*.csv` | 已做好 k-core 过滤 + 时间切分，I&S 仅 23.9 MB | ✅ **用** |

**关键工程决策**：官方 benchmark 分片直接给出了 5-core 过滤 + 全局时间切分的结果，因此**省掉了 2.2 GB 的全量评论下载与自跑 k-core**（原 V0 的 `amazon18_data_process.sh` 就是干这个的）。代价是失去了"自定义 k 值 / 自定义过滤规则"的自由度——如果需要，再回到全量评论自己跑。

---

## 3. 为什么选 Industrial_and_Scientific

Amazon 2023 有 33 个类目。选择 I&S 的理由：

| 理由 | 说明 |
|---|---|
| **纵向对照最干净** | V0 用的也是 I&S（2018 版），同类目跨年份对比，能直接量化"数据升级带来了多少增益"，排除类目差异的干扰 |
| **规模适中** | 5-core 后 25,847 个 item，图像只需下 2.6 万张（约 2GB），本地 4GB 显卡 + 17GB 内存可完整跑通 SID/编码 |
| **视觉信息有区分度但不过分** | 工业品有大量规格/型号/外观差异（如示波器、3D 打印耗材、实验器皿），图像能提供文本没有的信息；又不像服装那样依赖主观风格 |
| **文本质量高** | `features` 覆盖率 92.2%，是结构化的卖点短语，比评论摘要更适合做 item 表征 |
| **学术可比性** | MiniOneRec / LETTER / 大量 SID 类论文都在 I&S 上报告结果，指标有参照系 |

代价：**5-core 后用户极度稀疏**（见 §8.2 的长尾分析），序列推荐比密集场景难做。

---

## 3.5 第二个域为什么选 Video_Games（实测选型，非拍脑袋）

为了验证方法的泛化性，第二个类目的选择标准是：**与 I&S 域距离最远 × 规模量级可比 × metadata 体积最小（下载成本低）× 图像信息有区分度**。

做法：把 HF 上全部 **28 个类目**的 `benchmark/5core/timestamp` 分片体积拉下来排序，再对 4 个候选**真实下载 train.csv 逐个统计**（不是看体积估算）：

| 候选 | 全量交互 | users | items | metadata 体积 | 判定 |
|---|---:|---:|---:|---:|---|
| **Video_Games** ✅ | 814,586 | 94,762 | **25,612** | **417 MB（最小）** | **选用**：与 I&S 的 25,848 items 量级几乎相同（差 0.9%），但域距离最远 |
| Baby_Products | — | — | 30,754 | 659 MB | 淘汰：商品图多为白底摆拍，图像区分度低 |
| Musical_Instruments | — | — | 22,753 | 603 MB | 淘汰：与 I&S 同属"器材/工具"，域距离不够 |
| Office_Products | — | — | 65,873 | 2,047 MB | 淘汰（留作 V0 对照候选）：体量过大，图像下载翻 2.5 倍 |

**两域互补性论证**：
- **域距离**：工业器材（参数/规格驱动决策，用户看型号）vs 电子游戏（IP/题材/封面驱动决策，用户看画面）——同一种 SID 量化方式要在两种语义结构上都成立，结论才有说服力。
- **图像信息角色不同**：I&S 图像提供"型号/外观规格"这类**补充性**信息；VG 图像（封面）本身**就是**用户决策的主要依据。若图像融合在 VG 上升幅更明显，恰好能反向印证融合模块真的在"读图"。
- **量级对齐**：25,847 vs 25,612（差 0.9%），硬件与超参**不需要为第二个域做任何调整** → 严格的同构对照，排除"规模差异"这个混淆变量。
- **用户重叠**：两域有 **3,452 个共同用户**，为后续可能的跨域迁移分析留了钩子（但本项目不做跨域实验）。

---

## 4. 原始数据：官方 5core / timestamp 分片

### 4.1 文件与格式

```
data/Amazon23/raw/
├── Industrial_and_Scientific_5core_timestamp.train.csv   17.3 MB
├── Industrial_and_Scientific_5core_timestamp.valid.csv    2.6 MB
├── Industrial_and_Scientific_5core_timestamp.test.csv     4.1 MB
├── meta_Industrial_and_Scientific.jsonl                1130.0 MB (1,130,006,344 B)
└── images/  （实际落在 data/Amazon23/IandS/images/）
```

CSV 是标准四列，**rating 为 1.0~5.0 的浮点，timestamp 为毫秒**（需 //1000 转秒）：

```csv
user_id,parent_asin,rating,timestamp
AE22EETNRS4ZQO7PQ3XZPPPQGDNQ,1414212385,5.0,1282444800000
```

| 列 | 含义 | 注意点 |
|---|---|---|
| `user_id` | 用户 ID（哈希串） | 非数字，需建 `user2idx` |
| `parent_asin` | 商品 ID | Amazon 用 parent（父体）而非子体，**已合并变体**（同款不同颜色/尺码算一个 item） |
| `rating` | 评分 1.0~5.0 | 本项目按隐式反馈处理（`min_rating=0.0`，全保留），rating 只用于分析 |
| `timestamp` | 毫秒时间戳 | **单位是毫秒**，容易踩坑 |

### 4.2 实测统计（原始分片）

**双域并行对照**（两域格式完全相同，同一份脚本处理）：

| | 域 A：IandS（工业器材） | 域 B：VG（电子游戏） |
|---|---:|---:|
| **全量交互** | 412,947 | **814,586**（1.97×） |
| **全量用户（去重）** | 50,985 | 94,762 |
| **全量商品（去重）** | 25,848 | **25,612**（−0.9%，量级对齐） |
| 5★ 占比 | 72.30% | 63.84% |
| metadata 体积 | 1.13 GB | 417 MB |

分 split 明细：

| 集 | split | 交互数 | 用户数 | 商品数 | 评分均值 | 时间范围 |
|---|---|---:|---:|---:|---:|---|
| **IandS** | train | 297,922 | 46,341 | 21,593 | 4.459 | 2001-07-04 ~ 2021-08-11 |
| | valid | 44,128 | 17,932 | 11,781 | 4.288 | 2021-08-11 ~ 2022-07-16 |
| | test | 70,897 | 17,440 | 12,164 | 4.515 | 2022-07-16 ~ 2023-08-25 |
| **VG** | train | 736,827 | 91,562 | 22,976 | 4.270 | 1999-10-22 ~ 2021-08-11 |
| | valid | 34,510 | 15,655 | 7,620 | 4.176 | 2021-08-11 ~ 2022-07-16 |
| | test | 43,249 | 13,963 | 7,671 | 4.314 | 2022-07-16 ~ 2023-09-02 |

> 校验：I&S 三 split 行数之和 = 297,922 + 44,128 + 70,897 = 412,947；VG = 736,827 + 34,510 + 43,249 = 814,586。
> **两域的切分时间点完全一致**（`max(train)=2021-08-11`、`valid/test` 边界同为 2021-08-11 / 2022-07-16）——官方是按全局时间轴统一切分的，这给"双域对照"又加了一层严格性。

**两域的行为差异也很有意思**（面试可讲）：VG 交互量是 I&S 的 2 倍，但商品数几乎相同 → VG 单商品平均交互 31.8 次 vs I&S 16.0 次，**VG 是更"热"的域**；同时 VG 的 5★ 占比低 8.5 个百分点（63.84% vs 72.30%），评分分布更分散。这说明两个域的"数据密度"与"评分偏斜"确实不同，不是同一分布的简单复制。

### 4.3 时间结构（无泄漏，已数学验证）

| split | IandS 时间范围 | VG 时间范围 |
|---|---|---|
| train | 2001-07-04 ~ **2021-08-11** | 1999-10-22 ~ **2021-08-11** |
| valid | 2021-08-11 ~ 2022-07-16 | 2021-08-11 ~ 2022-07-16 |
| test | 2022-07-16 ~ **2023-08-25** | 2022-07-16 ~ **2023-09-02** |

**泄漏判定**：I&S `max(train_ts)=1628643178` (2021-08-11) < `min(test_ts)=1658002837` (2022-07-17)；VG `max(train_ts)=1628643144` < `min(test_ts)=1658003248`。**两域都是全局时间切分，无跨期泄漏**（脚本内 `time_leak=false` 为自动断言，两个域都通过）。

> 这一条比 V0 的 leave-one-out 更有说服力：V0 是"每个用户留最后一条做 test"，用户内部无泄漏，但**全局时间轴上 train/test 交错**——一个 2018 年的 train 样本可能包含 2017 年的交互，而同一天另一个用户的 test 也来自 2017 年。官方分片是**纯时间轴切分**，更接近线上的真实场景（模型部署后面对的一定是未来数据）。
> 唯一的代价：训练时不能跨 split 借用未来信息（比如"用户特征统计"要从 train 单独算）。

---

## 5. 商品 metadata

### 5.1 文件

`meta_Industrial_and_Scientific.jsonl`，1.13 GB，每行一个 JSON 对象，**全类目共 1.13 GB 但我们只需要其中 25,848 个 item**。

处理方式（`prepare_amazon23.py: build_item_table`）：**流式逐行读取**，用 `if '"parent_asin"' not in line: continue` 先做字符串级预筛，再 `json.loads`，只保留 `item_set` 里的商品。实测扫描 1.13GB 全量耗时约 1 分钟，内存峰值只存 25,847 条记录。

### 5.2 抽取的字段与实测覆盖率

产物：`data/Amazon23/IandS/IandS.item.json`（39.8 MB）

| 字段 | 覆盖率 | 说明 | 用途 |
|---|---:|---|---|
| `asin` | 100% | 商品 ID（parent_asin） | 对齐 item2id |
| `title` | **100%** | 标题（截断 300 字符） | 文本编码 / SFT 语料 |
| `images` | **100%** | 图像 URL 列表（主图在前，最多存 5 张） | 多模态编码 |
| `has_image` | 100% | 是否有图（布尔） | 快速过滤 |
| `item_id` | 100% | 本项目分配的 0-based 整数 id | 索引 |
| `brand` | 99.9% | 品牌（来自 `store` 字段） | 文本编码可选字段 |
| `main_category` | 99.0% | 主类目（如 `Industrial & Scientific`） | 层级特征 |
| `categories` | 97.3% | 类目路径列表（如 `['Industrial & Scientific', 'Additive Manufacturing Products', '3D Printing Supplies']`） | 层级特征 / 冷启动 |
| `features` | **92.2%** | 卖点短语列表（截断 8 条 × 200 字符） | **文本编码主力** |
| `price` | 74.7% | 价格（浮点） | 可选特征 |
| `description` | **57.5%** | 长描述（截断 1000 字符） | 文本编码补充 |

> ⚠️ 注意 `description` 覆盖率只有 **57.5%**（V0 里它是主力字段之一）。所以本项目文本编码的字段组合必须做消融：**`title` 是唯一 100% 可用的字段**，`title + features + categories` 是覆盖率与信息量的较优组合，`description` 只能作为增益项而非必需项。

**真实样本**（item_id=1）：

```json
{
  "asin": "079451524X",
  "title": "AmScope BK-WM The World of the Microscope A Practical Introduction with Projects and Activities,multicolor",
  "features": ["The World of the Microscope is perfect for childern grade 3-6 or ...", ...],
  "description": "The World of the Microscope is perfect for childern ...",
  "categories": [],
  "brand": "AmScope",
  "main_category": "Industrial & Scientific",
  "price": 13.48,
  "images": ["https://m.media-amazon.com/images/I/71cYRHqe3lL._SL1500_.jpg", ...],
  "has_image": true,
  "item_id": 1
}
```

---

## 6. 商品图像资产

这是本次数据升级的**最大增量**，图像覆盖率直接决定多模态方案是否成立。**两个域都各下一套**（文件名是各自的 `item_id`，互不共用）。

| 指标 | 域 A：IandS | 域 B：VG |
|---|---:|---:|
| 商品总数 | 25,847 | 25,611 |
| 有图像 URL 的商品 | **25,847（100.0%）** | **25,610（100.0%）** |
| 下载任务数 | 25,807 | 25,610 |
| 下载成功 | 25,784 | **25,570** |
| 失败明细 | `bad_image` 19 / `error`(超时) 3 / `http_404` 1 = 23 | `bad_image` 16 / `error` 19 / `http_404` 5 = 40 |
| **落盘图片文件数** | **25,824** | **25,570** |
| **有效覆盖率** | **99.91%** | **99.84%** |
| 总耗时 | 1,575.5 s（≈26 min，16 img/s） | 1,532.4 s（≈25.5 min，16.7 img/s） |
| 磁盘占用 | 约 **2.7 GB** | 约 **3.2 GB** |

> **图像 URL 覆盖率两域都是 100%**——设计时预判会有 10~20% 失效，还专门为"纯文本回退"设计了一条降级路径（图像向量补零 + 门控降权）。实测下来 URL 侧 100%、下载侧 99.84~99.91%，**多模态方案不需要靠回退兜底**，回退路径只保留作消融对照。
> VG 的失败数（40）比 IandS（23）略多，主要多在 `error`(超时) 与 `http_404`——**娱乐品类商品下架/换图的频率更高**，这本身也印证了两域的差异。

产物文件（两域结构相同，只换目录名）：

```
data/Amazon23/{IandS,VG}/
├── images/
│   ├── <item_id>.jpg          文件名 = 该域自己的 item_id
│   └── ...
├── image_manifest.tsv         逐条下载记录（item_id / url / status / bytes / error）
└── image_stats.json           汇总统计（上面那张表的机器可读版）
```

下载脚本 `scripts/multimodal/download_images.py` 的设计要点（面试可讲工程细节）：
1. **纯标准库**（`urllib` + `ThreadPoolExecutor`），不依赖 requests/PIL，避免环境未就绪时无法启动；
2. **断点续传**：以 `item_id.jpg` 落盘，已存在的文件直接跳过 → 中断后重跑只补缺失；
3. **魔数校验**：不用 PIL，直接读文件头判断 JPEG/PNG magic number，过滤"HTTP 200 但返回 HTML 错误页"的情况（IandS 的 19 张 / VG 的 16 张 `bad_image` 就是这样揪出来的）；
4. **manifest 全量留痕**：每次下载写 `item_id / url / status / error`，失败可精确重试而非全量重跑；
5. **`--reuse_from`（双域新增）**：按 **asin** 桥接复用其他 short 已下好的图片 —— 因为图片文件名是各 short 自己的 `item_id`，跨 short 不可直接复用；若将来加第三个类目且与已有类目有商品重叠，可直接复用。实测两域 ASIN 交集为 0，故本次 `reused_this_run=0`。

---

## 7. 处理产物（本项目自建的双域数据集）

### 7.1 文件清单

```
data/Amazon23/IandS/
├── IandS.item.json        39.8 MB   25,847 个商品的完整元信息（§5.2）
├── IandS.item2id         454 KB     asin → item_idx 映射（按 asin 字典序确定性生成）
├── IandS.train.inter      9.67 MB   251,576 条训练样本
├── IandS.valid.inter       707 KB   16,074 条验证样本
├── IandS.test.inter        643 KB   13,232 条测试样本
├── IandS.stats.json        688 B    统计摘要（下表的机器可读版）
├── images/                ~2.0 GB   25,824 张商品图
├── image_manifest.tsv     2.08 MB   下载留痕
└── image_stats.json        337 B    图像统计
```

`.inter` 采用 **RecBole 风格**三列（带表头），与 V0 的 CSV 格式解耦、便于后续接 RecBole 做 baseline 对照：

```
user_id:token	item_id_list:token_seq	item_id:token
18808	4526	18955        # 用户 18808：历史 [4526] → 目标 18955
```

### 7.2 切分逻辑（本项目自建，与官方分片对齐）

| split | 构造方式 | 样本数 |
|---|---|---:|
| train | **滑窗**：对每个用户的 train 段，第 i 个物品为 target、前 i 个（最多 20 个）为 history（i 从 1 开始） | 251,576 |
| valid | 该用户 **train 全部物品** → 预测 valid 物品（每用户 1 条，取 valid 段最后一条） | 16,074 |
| test | 该用户 **train + valid 全部物品** → 预测 test 物品（每用户 1 条） | 13,232 |

设计要点：
- 历史统一截断到**最近 20 个**（`--history_len 20`），与 V0 一致，保证指标可比；
- 同一 split 内部按物品去重（一个用户重复买同一商品只算一次），避免自重复样本；
- 严格**只用过去预测未来**：train 滑窗时第 i 个 target 的 history 只含第 i 个之前的物品，不会用到 valid/test。

### 7.3 实测统计（`IandS.stats.json`）

| 指标 | 值 |
|---|---:|
| 商品数 `n_items` | **25,847** |
| 用户数 `n_users` | **46,341** |
| train / valid / test 样本 | **251,576 / 16,074 / 13,232** |
| 图像覆盖率 | **100%**（URL 侧） |
| features 覆盖率 | 92.22% |
| train 历史长度 min/p50/mean/max | 1 / 3 / 4.8 / 20 |
| test 历史长度 min/p50/mean/max | 1 / 5 / 6.38 / 20 |

**两个数的口径要说清楚**（面试容易被追问）：

- `n_users = 46,341`：这是**有 train 历史的用户数**（= 原始 train 段的去重用户数），而不是原始全量 50,985。差集来自：valid 段有 1,856 个用户、test 段有 4,207 个用户在 train 段**完全没出现**——没有历史就无法构造"历史 → 目标"样本，只能丢弃（已记入 `dropped_valid_users_no_train_history` / `dropped_test_users_no_train_history`，口径可追溯）。
- `n_items = 25,847` vs 原始 25,848：差 1 个，因为有一个商品在 metadata 里**没有 title**，被过滤（无 title 无法做文本编码，也无法进 SFT 语料）。过滤条件写在 `prepare_amazon23.py:253`。

---

## 8. 数据的四个关键特性（面试最有价值的部分）

### 8.1 评分极度偏斜 → 必须按隐式反馈处理

| 评分 | 数量 | 占比 |
|---:|---:|---:|
| 1★ | 20,970 | 5.08% |
| 2★ | 12,900 | 3.12% |
| 3★ | 23,792 | 5.76% |
| 4★ | 56,718 | 13.73% |
| 5★ | **298,567** | **72.30%** |

**72.3% 是 5 星**，均值 4.4 左右。这种分布下：
- 不能把 rating 当回归目标（几乎全是高分，方差极小，模型会学成"恒输出 4.5"）；
- 如果按 `rating >= 4` 过滤，会丢掉 15% 的数据且引入"只有好评才算交互"的偏置；
- **本项目按隐式反馈处理（`min_rating=0.0` 全保留）**：有交互即正向信号，与工业推荐的主流做法一致。rating 保留在 `item.json` 里，供后续做"加权损失"或"负反馈"的消融。

### 8.2 长尾严重 + 用户极稀疏 → 这是生成式召回的机会点

| 维度 | min | p50 | mean | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 每个商品的交互数 | 5 | 9 | 16.0 | 29 | 120 | **2,133** |
| 每个用户的交互数 | 5 | **6** | 8.1 | 13 | 31 | 204 |

- 商品侧：5-core 保证最少 5 次交互，但**一半的商品不足 9 次**，而头部商品有 2,133 次——**238 倍**的长尾比。
- 用户侧：**中位数只有 6 次交互**，一半用户的序列长度 ≤ 6。这意味着序列推荐模型可用的行为信号非常有限，**ID embedding 类方法（如 SASRec）会因训练样本不足而退化**——这正是引用 LLM 世界知识（用 title/features 描述商品而非纯 ID）的动机所在。

### 8.3 33% 的测试目标是"训练集从未出现过的新物品" → 真·冷启动场景

实测：test 段 12,164 个目标商品中，有 **4,021 个（33.06%）在 train 段完全没有出现过**。

```
test 目标商品总数            12,164
其中未在 train 出现（冷启动物品） 4,021  (33.06%)
```

**这个数字非常重要**，直接把本项目与协同过滤类方法区分开：
- 传统 CF / ID-embedding 方法对未见过的 item **结构上无法推荐**（没有 embedding 可查）；
- 生成式召回（LLM 生成 SID）在原理上**可以泛化**——因为 SID 是从 item 的**文本/图像内容**量化出来的，新商品只要有内容就能编码出 SID，LLM 也能基于语义生成对应的 SID 前缀。
- **面试话术**：「在我们的测试集里，33% 的目标商品在训练集中从未出现，这是 ID-based 方法的结构性盲区。我们把 item 内容（title/features/图像）作为 SID 的来源，使新商品天然可召回——这是选生成式范式而非双塔 CF 的核心论据。」

> 诚实提示：这里 33% 是**原始分片**的口径（test 段目标 vs train 段是否出现）。本项目自建样本的 test 目标同样以官方 test 段物品为准，所以这个比例成立。但如果后续引入"过滤冷启动物品"的评估协议，需要重新统计——口径要写清楚，不能混用。

### 8.4 时间漂移明显 → 支持"时序泛化"类实验

train 跨越 2001~2021（20 年），test 是 2022-07 ~ 2023-08。20 年前的商品（数据里确实有 2001 年的评论）与现代商品风格差异很大，模型如果过度拟合 train 的早期分布，会在 test 上损失明显。可以做：
- 时间衰减加权采样；
- 只用最近 N 年数据训练（消融）；
- 分析"train 早期样本占比 vs test 指标"的关系。
这些都能作为面试时"你如何理解数据"的加分项。

### 8.5 同品多 listing 与"孪生表示"：SID 碰撞的数据根因

这是**决定 SID 唯一性上限的数据特性**，也是本项目"语义桶"策略的直接依据（处置见
`SID_PIPELINE.md` §3.6）。

#### 现象：原始商品里存在内容完全相同的多个 listing

Amazon 上同一商品常有多个 listing（不同卖家 / 不同 SKU / 重复上架），它们的
`title + features + categories` **完全相同，连主图 URL 都一字不差**。本项目用
`title+features+category` 拼文本做编码，于是这些 listing 得到**逐位相同（bit-identical）**
的文本向量；主图若也相同，图像向量同样逐位相同。

实测（对编码后的向量按行 `np.unique(axis=0)` 统计）：

| 域 | 文本向量重复组 | 图像向量重复组 | **融合（gate）后仍重复** | 融合后 ICR 理论上限 |
|---|---:|---:|---:|---:|
| IandS（N=25,847） | 40 组 / 82 物品 | 366 组 / 907 物品 | **30 组 / 60 物品** | **0.9988** |
| VG（N=25,611） | 150 组 / 331 物品 | 286 组 / 664 物品 | **133 组 / 297 物品** | **0.9936** |

> 融合后重复 = 文本与图像**同时**重复的那些组；ICR 上限 = 去重后行数 ÷ N。

#### 三个反直觉的实测结论

1. **图像重复比文本重复更普遍得多**（IandS 366 vs 40 组；VG 286 vs 150 组）。
   原因是大量不同商品共用了同一张通用图（配件通用图、占位图、同系列共用主图）。
2. **多模态融合本身在降低孪生率**：融合后只剩 30 / 133 组，比任一单模态都少。
   只要有一个模态不重复，融合向量就不重复（机制证据见下）。
   → 这是多模态表示的附带收益：**concat 提高了表示的区分度**，不只是"加信息"。
3. **VG 的孪生是 IandS 的 4 倍多**（133 vs 30 组），因为电子游戏的同品多 listing
   （同一款游戏的不同版本 / 卖家）远多于工业器材。这直接压低了 VG 的 ICR 上限。

#### 机制：门控没有饱和，"孪生"不是模型的锅

一开始怀疑是门控把图像分支关掉了：`g = sigmoid(MLP([e_t ; e_i]))`
（`fuse_embeddings.py:121-134`），若 `g` 饱和到 0/1，输出就退化为纯文本分支。
逐组实测推翻了这个猜测：

| 域 | A 文同 + 图同 → 融合必同 | B 文同 + 图不同 + 融合仍同 | C 文同 + 图不同 → 融合不同 |
|---|---:|---:|---:|
| IandS（文本重复 40 组） | 30 | **0** | 10 |
| VG（文本重复 150 组） | 133 | **0** | 17 |

反向也验过：**图同 + 文不同**的组，IandS 337 组 / VG 164 组，**融合后仍相同的 0 组**。

**B = 0** 说明：只要任一模态有差异，融合输出必然有差异——门控对两个模态的差异**双向敏感**，
从未把任一模态关掉。剩下的 A 类是**输入本身完全相同**，任何确定性映射都会给出相同输出，
与融合方式无关，是纯数据问题（复现脚本 `.tmp/probe_gate_twins.py <IandS|VG>`）。

#### 影响与处置

- **数学上不可分**：A 类孪生组在任何确定性 encoder 下都同码，Sinkhorn 也只能靠 batch 分桶
  "抽签"拆开一部分（IandS 实测拆开约 76%），不可能彻底消除。
- **处置：不消除，成桶召回**。这些组本来就是同一商品的不同 listing，语义等价，
  整桶召回再交排序消歧是正确行为。
- **VG 需要单独看**：孪生占 1.16%（297 / 25,611），碰撞账本必然比 IandS 更重。

#### 可以看图验证（文件名 = `item_idx`）

| 域 | item_idx | 标题 | 情况 |
|---|---|---|---|
| VG | 20 / 2151 | The Legend of Zelda: The Wind Waker | 同一 URL `51W8C95QDDL.jpg` → 两张图字节级相同 |
| VG | 576 / 7041 | Illusion of Gaia | 同一 URL `91ESi5MR+XL._SL1500_.jpg` |
| VG | 727 / 728 | TMNT IV: Turtles in Time | 同一 URL `81yqu9YpEnL._SL1500_.jpg` |
| VG | 346 / 4696 | Pokemon Green（日版 Game Boy） | **图不同**（`614BDAW826L` vs `71iaEYnapnL`）→ 被融合拆开（C 类） |
| VG | 1500 / 1523 | Sony PS2 红外 DVD 遥控 | **图不同**（`51VDb5tpLJL` vs `61UDPLZtaxL`）→ C 类 |
| VG | 2329 / 2330 | NCAA Football 2004 | **图不同**（`618IU-9JkXL` vs `81qThxqftHL`）→ C 类 |
| IandS | 107 / 1803 | VELCRO Brand Tape, Black 15ft | 同一 URL `61Pb1VeHTCL._SL1500_.jpg` |
| IandS | 914 / 2203 | SE Stainless Steel Funnel HQ93 | 同一 URL `61RVgRW8pKL._AC_SL1285_.jpg` |
| IandS | 954 / 1284 | Primacare DS-9290-BK 听诊器 | 同一 URL `61tYBoiE3DL._AC_SL1500_.jpg` |

图片路径 `data/Amazon23/{IandS,VG}/images/<item_idx>.jpg`，完整 URL 见各自 `image_manifest.tsv`。

---

## 9. 与 V0（Amazon Reviews 2018）严格对比

| 维度 | V0（MiniOneRec 复刻） | 本项目（GenRetrieval） | 倍数 |
|---|---|---|---|
| 数据集 | Amazon Reviews 2018 | **Amazon Reviews 2023** | — |
| 类目 | I&S + Office_Products | **I&S**（单类目，便于纵深） | — |
| 数据时间 | 2016-10 ~ 2018-11 | **1996/2001 ~ 2023-08** | — |
| 商品数（I&S） | 3,686 | **25,847** | **7.0×** |
| train 样本 | 36,259 | **251,576** | **6.9×** |
| valid / test 样本 | 4,532 / 4,533 | **16,074 / 13,232** | 3.5× / 2.9× |
| 商品字段 | title, description | title, **features, categories, brand, main_category, price**, description, images | — |
| 图像 | ❌ 无 | ✅ **25,824 张（99.91%）** | 从 0 到 1 |
| 切分方式 | leave-one-out（用户级） | **官方全局时间切分** | 更严格 |
| 泄漏风险 | 用户内无，全局时间轴交错 | **全局无泄漏（已验证）** | — |
| SID 空间利用 | 3,686 / 16.7M ≈ 0.02% | 25,847 / 16.7M ≈ 0.15% | 7.5× |
| 用户交互中位数 | 密集（5-core 后仍较多） | **6** | 更稀疏 |

> 说明：V0 数字取自原项目 `PROJECT_DOCUMENTATION.md`（已归档为 `docs/V0_MINIONEREC_TECH_DOC.md`）。**V0 的绝对指标不可与本项目的绝对指标直接比较**——数据不同、切分不同，只能作为"方法有效性"的历史参照。公平对比必须在本项目的数据上重跑 V0 配置（记入 `EXPERIMENT_LOG.md`）。

---

## 10. 复现命令（双域各跑一遍，命令完全同构）

```bash
# ---------- 域 A：Industrial_and_Scientific ----------
# 1) 下载官方 5core 分片（23.9 MB）+ 商品 metadata（1.13 GB）
bash scripts/data/download_amazon23.sh Industrial_and_Scientific
#    内部走 hf-mirror.com；>100MB 的下载需在沙箱外执行（沙箱限速 ~110kB/s）

# 2) 构建 IandS.{item.json,item2id,*.inter,stats.json}
python scripts/data/prepare_amazon23.py \
    --category Industrial_and_Scientific --short IandS --history_len 20
#    实测耗时 ~1 分钟（主要花在流式扫描 1.13GB metadata）

# 3) 下载商品图像（32 线程，约 26 分钟，可断点续传）
python scripts/multimodal/download_images.py --short IandS --workers 32

# 4) 冒烟测试（先跑 40 张确认链路通）
python scripts/multimodal/download_images.py --short IandS --limit 40 --workers 16

# ---------- 域 B：Video_Games（同一套命令，只换 --category/--short）----------
bash scripts/data/download_amazon23.sh Video_Games          # metadata 仅 417 MB
python scripts/data/prepare_amazon23.py \
    --category Video_Games --short VG --history_len 20
python scripts/multimodal/download_images.py --short VG --workers 32

# 说明：两域产物分别落在 data/Amazon23/{IandS,VG}/，互不干扰；
#       默认不做合并（多类目合并能力仅在脚本里作为可选开关保留）。
```

---

## 11. 数据侧踩坑表

> 编号与 `docs/EXPERIMENT_LOG.md` §4.2 **一致**（那里是唯一真源，本表只摘数据相关的条目并补充上下文）。

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| D-01 | 原处理脚本（`amazon23_data_process.py`）需要 **2.2GB 全量评论 jsonl** 才能跑 | 下载慢 + k-core 迭代久 | 改用官方 `benchmark/5core/timestamp` 分片（23.9MB），省 2.2GB 下载、免自跑 k-core |
| D-02 | **官方切分是"全局时间切分"，违反直觉** | 以为"每用户留最后两条"，实际 test 用户里有 **4,207 个在 train 段没有任何交互** | 构造样本时显式丢弃"无 train 历史"的 valid/test 用户，并把丢弃数写入 `stats.json`（口径可追溯） |
| D-05 | 手工下载的文件名（`IandS_5core_*.csv`）与脚本约定（`<Category>_5core_*.csv`）不一致 | `FileNotFoundError` | 统一为 `<Category>_5core_timestamp.<split>.csv`；下载脚本与处理脚本的命名约定写在同一处 |
| D-06 | HF tree API 报的 metadata 大小（1077.7MB）≠ 实际下载（**1130.0MB**） | 完成度校验误判 | 以实测字节数为准，写回脚本 `EXPECT_BYTES` 默认值 |
| D-07 | **`timestamp` 单位是毫秒**（不是秒） | 直接当秒用会算出公元 5 万年，时间比较全错 | `read_split_csv` 统一 `int(ts) // 1000` |
| D-08 | **`description` 覆盖率只有 57.5%** | 设计时以为它是主力字段（V0 里确实是），实际近半为空 → 会静默退化 | 文本字段组合必须消融；默认取 `title+features+category`（97.3%），title 是唯一 100% 字段 |
| D-09 | 部分商品在 metadata 里**没有 title** | `n_items` 25,848 → 25,847，且"为什么少 1 个"难解释 | 显式过滤并记录（`prepare_amazon23.py:253`）；无 title 无法编码也无法进 SFT 语料 |
| D-10 | **图像 URL 返回 200 但内容是 HTML 错误页** | 直接 `PIL.Image.open` 会在编码阶段才炸，那时已浪费大量时间 | 下载时做**文件头魔数校验**（JPEG `FFD8FF` / PNG `89504E47`），实测揪出 19 张 `bad_image` |
| D-11 | **1.13GB metadata 直接 `json.load` 会 OOM** | 内存峰值 >8GB（本机可用仅 8.3GB） | **流式逐行** + 字符串预筛 `'"parent_asin"' in line`，内存只留命中的 25,847 条 |
| D-12 | ⚠️ **用通配符 `rm -f *_5core_timestamp.train.csv` 清临时文件，把已下好的正式分片一起删了** | I&S 的 train.csv（17.3MB）被连带删除 | 立刻重下 + **HF 官方字节数（17,281,344）+ 行数（297,922）双重校验**确认无损。**铁律：`data/` 下永远不用通配符删文件**，清临时文件只指向 `.tmp/` |
| D-13 | 官方 benchmark 分片的**文件名是 `<Category>.<split>.csv`**（不是 `<Category>_5core_timestamp.<split>.csv`） | 按错的名字下回来的是 15 字节的 HTML 错误页（404 页），`wc -c` 才发现 | 下载后用**字节数下限断言**（`>1000`）拦截；本项目统一重命名为 `<Category>_5core_timestamp.<split>.csv` 以匹配处理脚本 |
| D-14 | **单模态向量重复 ≠ 融合后重复**（图像重复组数是文本的 2~9 倍，融合后却只剩 30 / 133 组） | 看到融合向量仍有逐位重复，一度误判为"门控饱和把图像分支关掉了" | 逐组细分验证：文同+图不同 → 融合必不同，图同+文不同 → 也必不同（**B=0**）。真因是 Amazon 同品多 listing **连主图 URL 都相同**，输入本身完全相同（§8.5） |

---

## 12. 一页速记（面试前 3 分钟过）

```
数据集：Amazon Reviews 2023 · 双域并行（不合并）· 官方 5core/timestamp 分片
  域 A  Industrial_and_Scientific (IandS)  工业器材 / 规格参数驱动
  域 B  Video_Games              (VG)      电子游戏 / 封面艺术驱动

         ┌─────────── IandS ───────────┐   ┌─────────── VG ──────────────┐
items    25,847                        │   25,612          （差 0.9%，量级对齐）
users    46,341（有 train 历史）        │   91,540
原始交互 412,947                        │   814,586         （1.97×）
样本     train 251,576 / valid 16,074 / test 13,232
         train 645,055 / valid 14,231 / test 11,276
图像     25,824 张 @99.91% 有效 / 2.7GB   │   25,570 张 @99.84% 有效 / 3.2GB
5★ 占比  72.30%                         │   63.84%
切分     两域完全同一套时间边界：max(train)=2021-08-11，valid/test 边界 2021-08-11 / 2022-07-16
         两域 time_leak=false（脚本自动断言）

IandS 数据特性：用户交互中位数 6 ｜ 商品中位数 9（头部 2133，长尾 238×）
               test 目标 33.06% 未在 train 出现 → 生成式召回 vs CF 的核心论据
泛化性主张：同一套 pipeline 在两个域距离最远的类目上各跑一遍，结论不依赖单一类目
对比 V0： item ×7、样本 ×6.9、字段 +图像/features/categories、切分 leave-one-out → 全局时间切分
```

若要一句话回答"为什么要两个数据集"：**因为单类目上的增益无法区分"方法有效"和"这个类目恰好合适"；
两个域距离最远、规模几乎相同的类目上都能复现，才算证明了方法本身有效。**

---

## 13. 双域并行对照总表

> 这是本项目的**核心实验设定**。两域**各自独立成集、独立跑完整 pipeline**（编码 → SID → SFT → RL → 评估），
> **不做跨域合并、不做跨域迁移**。目的：把"方法有效性"与"类目偶然性"分离开。

| 维度 | 域 A：IandS（工业器材） | 域 B：VG（电子游戏） | 对照价值 |
|---|---:|---:|---|
| 商品数 | 25,847 | 25,612（−0.9%） | **量级对齐** → 硬件/超参无需调整，排除规模混淆 |
| 用户数（有 train 历史） | 46,341 | 91,540 | 1.98× |
| 原始交互 | 412,947 | 814,586 | 1.97× |
| 单商品均交互 | 16.0 次 | **31.8 次** | VG 是更"热"的域 |
| train / valid / test 样本 | 251,576 / 16,074 / 13,232 | 645,055 / 14,231 / 11,276 | — |
| 5★ 占比 | 72.30% | 63.84% | VG 评分分布更分散 |
| features 覆盖 | 92.2% | 87.1% | 两域均高 |
| 图像 URL 覆盖 | 100% | 100% | 多模态前提在两域都成立 |
| 图像**有效**覆盖 | **99.91%**（25,824 张 / 2.7GB） | **99.84%**（25,570 张 / 3.2GB） | 两域都可直接做多模态，无需回退兜底 |
| 文本向量孪生组 | 40 组 / 82 物品 | 150 组 / 331 物品 | VG 同品多 listing 约 4 倍于 IandS |
| 图像向量孪生组 | 366 组 / 907 物品 | 286 组 / 664 物品 | 通用图 / 占位图导致，比文本重复更普遍 |
| **融合后孪生组** | **30 组 / 60 物品** | **133 组 / 297 物品** | **决定 ICR 上限 0.9988 vs 0.9936**（§8.5） |
| metadata 体积 | 1.13 GB | 417 MB | 下载成本 |
| 时间边界 | 同 | 同 | **官方全局统一切分** |
| 域语义结构 | 参数/型号/规格驱动 | IP/题材/封面驱动 | **域距离最远** |
| 图像在决策中的角色 | 补充性（文本已含大部分信息） | **主导性**（封面即决策依据） | 若融合在 VG 上升幅更大 → 反证融合模块真在"读图" |
| 用户重叠 | — | 与 IandS 有 **3,452** 个共同用户 | 留作跨域分析钩子（本项目不做） |

**预期与判据（先写下来，避免事后解释）**：
1. **若两个域上 SID 评估三件套（ICR / MSE / LCP）的相对排序一致**（例如 RQ-VAE vs KMeans 谁更优），说明结论稳健 → 可写进简历。
2. **若排序不一致**，则结论必须限定到具体域，并分析"是不是图像信息在决策中的权重差异导致的"——这本身也是有价值的负结果（记入 `EXPERIMENT_LOG.md`）。
3. **图像融合的增益在 VG 上应 ≥ IandS 上**（因 VG 图像是主导性信息）。若相反，说明融合模块可能只是在拟合噪声。

