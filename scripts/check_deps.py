# -*- coding: utf-8 -*-
"""依赖完备性检查：扫真实 import，对比 requirements-core.txt

动机：`fire` 曾经漏在裁剪版 requirements 里，直到 `python sft.py` 才抛
`ModuleNotFoundError`（`fire.Fire(train)` 是入口）。云端 `setup_env.sh` 装的正是
requirements-core.txt，所以"少一个包"会直接卡住上云训练。

做法（不猜、按实际代码）：
  1. 用 AST 扫目标脚本的**顶层 import**（含函数内 import，`ast.walk` 覆盖）；
  2. 去掉标准库、本项目自己的模块（根目录 .py / 一级子包）；
  3. 用 `importlib.metadata.packages_distributions()` 把「模块名 -> 发行包名」映射出来
     （`sklearn` -> `scikit-learn`、`PIL` -> `pillow` 这类别名靠它才能对上）；
  4. 与 requirements-core.txt 里声明的包名做比对，列出**缺的**与**多余的**。

用法：
  ./.venv/Scripts/python.exe scripts/check_deps.py
  ./.venv/Scripts/python.exe scripts/check_deps.py --strict   # 有缺失则退出码 1
"""
import argparse
import ast
import importlib.metadata as md
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ_CORE = ROOT / "requirements-core.txt"

# 已知例外：不是"缺包"，各有明确理由（写在这里避免每次跑都误报）
KNOWN_EXCEPTIONS = {
    "torch": "按 CUDA 版本单独安装（见 requirements-core.txt 头部说明），故意不进 core 列表",
    "torchvision": "同上",
    "flash_attn": "可选加速；sft.py:215 有 try/except ImportError -> 自动降级到 sdpa，不装也能跑",
    "packaging": "只有 minionerec_trainer.py 直接 import（V0 复刻链路，本项目未启用）；"
                 "实际由 transformers / accelerate / datasets / peft / wandb 间接带上",
    "deepspeed": "本项目不用（requirements-core.txt 明确排除）",
    "torchrec": "同上",
    "fbgemm_gpu": "同上",
    "triton": "Linux only，本项目用不到",
}

# 主干 + 上云训练/评估会走到的脚本
DEFAULT_TARGETS = [
    "sft.py", "evaluate.py", "data.py", "calc.py", "LogitProcessor.py",
    "utility.py", "minionerec_trainer.py", "sasrec.py",
    "prompt_templates.py",
    "scripts/sft/*.py",
    "scripts/data/prepare_sft_data.py",
    "rq/models/rqvae.py",
]


def parse_req(path):
    """读 requirements 文件，返回 {规范化包名: 原始行}。跳过注释/空行/-r/-- 选项。"""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[;]", line, 1)[0].strip()
        if name:
            out[name.lower().replace("_", "-")] = line
    return out


def local_modules():
    """本项目自己的顶层模块：根目录 *.py 的 stem + 一级子目录名。"""
    mods = {p.stem for p in ROOT.glob("*.py")}
    mods |= {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    mods |= {p.name for p in ROOT.iterdir() if p.is_dir() and p.name in
             {"rq", "baseline", "scripts", "refs"}}
    return mods


def imports_of(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        print(f"  [WARN] 解析失败 {path}: {e}", file=sys.stderr)
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:          # 跳过相对 import
                mods.add(node.module.split(".")[0])
    return mods


def main():
    ap = argparse.ArgumentParser(description="依赖完备性检查（AST 扫 import 对比 requirements-core.txt）")
    ap.add_argument("--req", default=str(REQ_CORE))
    ap.add_argument("--strict", action="store_true", help="有缺失时退出码 1")
    a = ap.parse_args()

    req = parse_req(Path(a.req))
    stdlib = set(sys.stdlib_module_names)
    local = local_modules()
    pkg2dist = md.packages_distributions()

    files = []
    for pat in DEFAULT_TARGETS:
        files.extend(sorted(ROOT.glob(pat)))
    files = [f for f in files if f.is_file() and "\.venv" not in str(f)]

    print(f"目标脚本 {len(files)} 个；requirements-core.txt 声明 {len(req)} 个包\n")

    # 模块 -> 用到它的脚本
    usage = {}
    for f in files:
        for m in imports_of(f):
            if m in stdlib or m in local or m.startswith("_"):
                continue
            usage.setdefault(m, set()).add(f.relative_to(ROOT).as_posix())

    missing, ok, unresolved, excused = [], [], [], []
    for mod in sorted(usage):
        if mod in KNOWN_EXCEPTIONS:
            excused.append((mod, KNOWN_EXCEPTIONS[mod]))
            continue
        dists = pkg2dist.get(mod) or []
        if not dists:
            unresolved.append(mod)
            continue
        hit = [d for d in dists if d.lower().replace("_", "-") in req]
        if hit:
            ok.append((mod, hit[0]))
        else:
            missing.append((mod, dists))

    print("=" * 78)
    print(f"A. 已声明  ({len(ok)})")
    print("=" * 78)
    for mod, dist in ok:
        print(f"  [OK]   {mod:22s} <- {dist}")

    print()
    print("=" * 78)
    print(f"B. import 了但 requirements-core.txt 里没有，且**不在已知例外里**  ({len(missing)})")
    print("=" * 78)
    if not missing:
        print("  (无)  —— 依赖声明完整 ✔")
    for mod, dists in missing:
        print(f"  [MISS] {mod:22s} 由这些发行包提供: {dists}")
        for f in sorted(usage[mod]):
            print(f"           used by: {f}")

    print()
    print("=" * 78)
    print(f"C. 已知例外（**不是缺包**，理由见下）  ({len(excused)})")
    print("=" * 78)
    for mod, why in excused:
        files_used = ", ".join(sorted(usage[mod])[:2])
        print(f"  [EXC ] {mod:22s} {why}")
        if files_used:
            print(f"           used by: {files_used}")

    print()
    print("=" * 78)
    print(f"D. 映射不到发行包（多为子模块/本地误判，人工看一眼）  ({len(unresolved)})")
    print("=" * 78)
    if not unresolved:
        print("  (无)")
    for mod in unresolved:
        print(f"  [????] {mod:22s} used by: {', '.join(sorted(usage[mod])[:3])}")

    print()
    print("=" * 78)
    verdict = f"缺 {len(missing)} 个" + ("  ❌" if missing else "  ✔ 完备")
    print(f"结果：{verdict}（已知例外 {len(excused)} 个不计）")
    print("=" * 78)
    return 1 if (missing and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
