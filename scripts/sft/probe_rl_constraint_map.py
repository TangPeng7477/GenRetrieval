"""验证 RL 阶段（ReReTrainer）的约束解码映射。

复刻两处真实逻辑，不手抄数值：
  A) 建表：`minionerec_trainer.py:529-572`（info_file -> prefixID -> hash_dict）
  B) 查表：`LogitProcessor.py:45-70`（count==0 取 prompt 末 prefix_index 个 token，
          之后取末 count 个）

检查三件事：
  1. `prefix_index = 3` 的前提：`'### Response:\n'` 是否恰好切成 3 个 token
  2. 每个条目的 SID 段是否恰好 3 个 token（否则表会退化成子 token 链）
  3. hash_dict 的步数形状是否与 `evaluate.py` 的 Trie 同构（5 步，末两步各 1 个候选）

用法：
    ./.venv/Scripts/python.exe scripts/sft/probe_rl_constraint_map.py --domain IandS
    ./.venv/Scripts/python.exe scripts/sft/probe_rl_constraint_map.py --max-items 2000
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

PROMPT_SUFFIX = "### Response:\n"


def get_hash(x):
    """与 minionerec_trainer.py:582-584 完全一致"""
    return "-".join(str(_) for _ in x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default=None)
    ap.add_argument("--model-dir", default="models/Qwen3-0.6B")
    ap.add_argument("--max-items", type=int, default=0, help="0=全部")
    a = ap.parse_args()
    sft = a.sft_dir or f"data/Amazon23/{a.domain}/sft"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    n0 = len(tok)
    with open(f"{sft}/info/sid_vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    tok.add_tokens(vocab)
    print(f"[env] tokenizer {n0} -> {len(tok)}（+{len(vocab)} SID，模拟 SFT 产物自带扩展词表）")
    print(f"[env] model      = {a.model_dir}")
    print(f"[env] info_file  = {sft}/info/{a.domain}.item_info.txt\n")

    ok = True

    # ---------- 1) prefix_index=3 的前提 ----------
    print("-" * 74)
    print("1. prefix_index = 3 的前提")
    print("-" * 74)
    for flag in (True, False):
        ids = tok(PROMPT_SUFFIX, add_special_tokens=flag).input_ids
        tag = f"add_special_tokens={flag}"
        print(f"  {tag:26s} -> {ids}  len={len(ids)}  {[tok.decode([i]) for i in ids]}")
    ids_def = tok(PROMPT_SUFFIX).input_ids              # ReReTrainer 用的是默认（True）
    ids_ns = tok(PROMPT_SUFFIX, add_special_tokens=False).input_ids
    if len(ids_def) == 3 and ids_def == ids_ns:
        print(f"  [PASS] 两种调用都 = 3 token 且相同 -> prefix_index=3 成立")
        PREFIX_IDS = ids_def
    else:
        print(f"  [FAIL] 默认调用 len={len(ids_def)}、no-special len={len(ids_ns)} -> prefix_index 需改")
        PREFIX_IDS = ids_def
        ok = False

    # ---------- 2) 复刻 ReReTrainer 的建表 ----------
    print()
    print("-" * 74)
    print("2. 复刻 ReReTrainer 建表（minionerec_trainer.py:529-572）")
    print("-" * 74)
    with open(f"{sft}/info/{a.domain}.item_info.txt", encoding="utf-8") as f:
        info = f.readlines()
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
    info_semantic = [f"{PROMPT_SUFFIX}{_}" for _ in semantic_ids]
    if a.max_items:
        info_semantic = info_semantic[: a.max_items]
    prefixID = [tok(_).input_ids for _ in info_semantic]
    prefix_index = 3
    print(f"  条目数 = {len(prefixID)}")

    lens = Counter(len(x) for x in prefixID)
    print(f"  tokenize 后长度分布 = {dict(sorted(lens.items()))}")
    sid_seg = Counter(len(x) - len(PREFIX_IDS) - 1 for x in prefixID)   # 减 prefix、减末尾 '\n'
    print(f"  SID 段长度分布      = {dict(sorted(sid_seg.items()))}")
    bad_head = [i for i, x in enumerate(prefixID) if x[: len(PREFIX_IDS)] != PREFIX_IDS]
    print(f"  首段 != '### Response:\\n' 的条目数 = {len(bad_head)}")
    if not bad_head and set(sid_seg) == {3}:
        print("  [PASS] 每条 = 3 前缀 + 3 SID + 1 换行")
    else:
        print("  [FAIL] 存在非预期切分")
        ok = False

    hash_dict = defaultdict(set)
    for ID in prefixID:
        ID = list(ID)
        ID.append(tok.eos_token_id)
        for i in range(prefix_index, len(ID)):
            key = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict[key].add(ID[i])
    hash_dict = {k: list(v) for k, v in hash_dict.items()}
    print(f"  hash_dict 条目数 = {len(hash_dict)}")

    # ---------- 3) 形状检查 ----------
    print()
    print("-" * 74)
    print("3. 映射形状（与 evaluate.py 的 Trie 是否同构）")
    print("-" * 74)
    depth_size = defaultdict(list)
    for k, v in hash_dict.items():
        depth_size[len(k.split("-"))].append(len(v))
    for d in sorted(depth_size):
        sizes = depth_size[d]
        c = Counter(sizes)
        # 深度 = key 里的 token 数（get_hash 用 '-' 连接）：
        #   1 -> 只有 1 个 SID（a 层）  2 -> 2 个 SID(a,b)  3 -> 3 个 SID(a,b,c) 或 prompt 末 3 token
        #   4 -> 3 个 SID + 换行
        label = {1: "1 个 SID (a 层)", 2: "2 个 SID (a,b)",
                 3: "3 个 SID 或 prompt 末3tok", 4: "3 SID + 换行",
                 5: "3 SID + 换行 (查表用)"}.get(d, f"{d} 段")
        top = ", ".join(f"{s}个候选 x{n}" for s, n in c.most_common(3))
        print(f"  深度 {d} [{label:22s}] 键数={len(sizes):6d}  候选数分布: {top}")

    # ---- 结构性断言（这三条就是"约束对不对"的判据）----
    print()
    print("  -- 结构断言 --")
    a_keys = sum(1 for k in hash_dict if len(k.split("-")) == 1)
    head = hash_dict.get(get_hash(PREFIX_IDS), [])
    n_a_layer = sum(1 for t in vocab if t.startswith("<a_"))
    sid3_keys = [k for k, v in hash_dict.items() if len(k.split("-")) == 3 and k != get_hash(PREFIX_IDS)]
    bad_sid3 = [k for k in sid3_keys if hash_dict[k] != [198]]
    four = [v for k, v in hash_dict.items() if len(k.split("-")) == 4]
    bad_four = [v for v in four if v != [tok.eos_token_id]]
    checks = [
        (f"prompt 末3token -> 候选数 == a 层码数({n_a_layer})", len(head) == n_a_layer),
        (f"a 层键数 == a 层码数({n_a_layer})", a_keys == n_a_layer),
        (f"3-SID 键（{len(sid3_keys)} 个）候选恒为 [换行 198]", not bad_sid3),
        (f"3SID+换行 键（{len(four)} 个）候选恒为 [EOS]", not bad_four),
    ]
    for name, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    # ---------- 4) 模拟一次解码 ----------
    print()
    print("-" * 74)
    print("4. 模拟解码（按 LogitProcessor 的查表规则走 5 步）")
    print("-" * 74)
    fake_prompt = tok(f"### User Input: \nwhatever\n\n{PROMPT_SUFFIX}", add_special_tokens=False).input_ids
    sent = list(fake_prompt)
    count = 0
    steps = []
    while count < 8:
        key_list = sent[-prefix_index:] if count == 0 else sent[-count:]
        allowed = hash_dict.get(get_hash(key_list), [])
        if not allowed:
            steps.append((count, "无候选 -> 不约束/警告"))
            break
        # 取一个候选继续走（等价于 beam 里的确定性选择）
        nxt = allowed[0]
        steps.append((count, f"候选 {len(allowed):>3d} 个 -> 选 {nxt} ({tok.decode([nxt])!r})"))
        sent.append(nxt)
        count += 1
        if nxt == tok.eos_token_id:
            break
    for c, s in steps:
        print(f"  step {c}: {s}")
    print()
    # 直接判定：最后一步是否 EOS
    last_ok = bool(steps) and str(tok.eos_token_id) in steps[-1][1]
    print(f"  末步 = EOS({tok.eos_token_id}) ? {'是 [PASS]' if last_ok else '否 [FAIL]'}")
    ok = ok and bool(last_ok)

    print()
    print("=" * 74)
    print("总结论:", "全部通过 ✔" if ok else "存在 FAIL ✗")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
