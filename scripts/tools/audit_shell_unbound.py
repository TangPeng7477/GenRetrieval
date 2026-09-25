# -*- coding: utf-8 -*-
"""静态体检：找出 shell 脚本里「裸引用早于赋值」的变量 —— `set -u` 下必崩的炸弹。

为什么需要它
------------
本项目已发生**两次**同一类事故，第二次还发生在自己的代码里：

  · `AUTO_COMMIT`（`evaluate_run0.sh`，修于 5e13c54）
    写法 `case "${AUTO_COMMIT:-0}"` —— **只取默认值、不赋值**，下游却用裸 `${AUTO_COMMIT}`。
  · `USE_LORA`（`rl_run0.sh`，修于 769f420）
    第 36 行 `LORA_SUFFIX` 就用它，而归一化赋值在文件后半第 73 行。

两次的共同点、也是最值得记的一点：**隔离测试里每个用例都显式设了该变量**
⟹ 「不传」这条**真实默认路径从未被测到**（`RL_PIPELINE §6.5` 的本地配方写的正是
`USE_LORA=True ...`）。人肉 review 容易漏，故用本脚本兜一层。

判据
----
命中 = 「**裸引用**」且「首次引用行早于首次赋值行（或从未赋值）」。

  · 裸引用 = `${VAR}`（不带 `:-` `:+` `:=` `:?` 修饰）或 `$VAR`
    ⟹ 带 `:-默认` 的引用在 `set -u` 下安全，**不算炸弹**
  · 赋值 = 行首（可缩进）`VAR=...` / `export VAR=...` / `VAR+=...`

自动跳过（shell 语义上必然有值，非炸弹）
----------------------------------------
  ① `for VAR in ...`                 —— 循环变量
  ② `local VAR=...` / `read ... VAR` —— 声明/读取即赋值
  ③ 上一非空行有 `"${VAR:-}"` / `-n "${VAR` / `-z "${VAR` 形式的**前置判空守卫**
  ④ `REVIEWED_SAFE` 里人工登记过的

用法
----
    python scripts/tools/audit_shell_unbound.py            # 扫默认的 4 个入口脚本
    python scripts/tools/audit_shell_unbound.py a.sh b.sh  # 扫指定脚本
    python scripts/tools/audit_shell_unbound.py --all      # 扫仓库根全部 *.sh

退出码：0 = 干净，1 = 有未解释的命中（可作闸门）。
⚠️ 假阴性风险：③ 的守卫识别是**行级**启发式，跨多行的复杂守卫可能识别不到（会多报，不会漏报）。
"""

import io
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SCRIPTS = [
    "rl_run0.sh",
    "sft_run0.sh",
    "evaluate_run0.sh",
    "beam_sweep.sh",
]

# ---------------------------------------------------------------------------
# 人工甄别过、确认安全的命中（复核后在此登记，保持脚本 0/1 输出有意义）
# ---------------------------------------------------------------------------
REVIEWED_SAFE = {
    ("evaluate_run0.sh", "COLLECT"):
        "auto-detect 的 `case` 两个分支都显式赋值，到 `case \"${COLLECT}\"` 处必然已设",
}

ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\+?=")
REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:?[-+?=][^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)")

BUILTIN = {
    "PATH", "HOME", "PWD", "SHELL", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
    "RANDOM", "SECONDS", "LINENO", "BASH_SOURCE", "BASH_VERSION", "FUNCNAME",
    "IFS", "TMPDIR", "HOSTNAME", "OSTYPE", "MACHTYPE", "PS1", "PYTHONPATH",
}


def _auto_bound_vars(lines):
    """① ② 语义上必然被赋值的变量名集合。"""
    bound = set()
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            continue
        m = re.search(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\b", ln)
        if m:
            bound.add(m.group(1))
        m = re.search(r"\blocal\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", ln)
        if m:
            bound.add(m.group(1))
        m = re.search(r"\bread\b([^;#]*)$", ln)
        if m:
            for name in m.group(1).replace("-r", " ").replace("-a", " ").split():
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    bound.add(name)
    return bound


def _guarded_at(lines, var, ref_line):
    """③ 判空守卫：同一行内，或上一个非空（非注释）行。

    同行形如 `if [ -n "${X:-}" ]; then echo "${X}"; fi` ；
    上一行形如 `if [ -n "${X:-}" ]; then` / `if [ -z "${X:-}" ] && ...`。
    ⚠️ 只认 `-n` / `-z` 紧跟该变量的形式，避免把 `echo "${X:-默认} ${X}"` 这类误判成守卫。
    """
    cur = lines[ref_line - 1]
    if re.search(r'-[nz]\s+"\$\{%s(?::[-+?][^}]*)?\}"' % re.escape(var), cur):
        return True
    for j in range(ref_line - 2, max(ref_line - 5, -1), -1):
        prev = lines[j]
        if not prev.strip() or prev.strip().startswith("#"):
            continue
        return bool(re.search(
            r"(\$\{%s:[-+?]\}|\s-[nz]\s+\"\$\{%s)" % (re.escape(var), re.escape(var)), prev))
    return False


def audit(path):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")

    first_assign = {}
    first_bare_ref = {}
    for i, ln in enumerate(lines, start=1):
        if ln.strip().startswith("#"):
            continue
        m = ASSIGN.match(ln)
        if m:
            first_assign.setdefault(m.group(1), i)
        for m2 in REF.finditer(ln):
            braced, modifier, plain = m2.group(1), m2.group(2), m2.group(3)
            var = braced or plain
            if not var or var in BUILTIN or var.isdigit():
                continue
            if bool(plain) or (bool(braced) and not modifier):
                first_bare_ref.setdefault(var, i)

    auto_bound = _auto_bound_vars(lines)

    hits, skipped = [], []
    for var, ref_line in sorted(first_bare_ref.items(), key=lambda kv: kv[1]):
        assign_line = first_assign.get(var)
        why = None
        if assign_line is not None and ref_line >= assign_line:
            continue                                     # 正常：先赋值后引用
        if var in auto_bound:
            why = "for/local/read 语义赋值"
        elif _guarded_at(lines, var, ref_line):
            why = "上一行有判空守卫"
        elif (os.path.basename(path), var) in REVIEWED_SAFE:
            why = "人工复核登记：" + REVIEWED_SAFE[(os.path.basename(path), var)]
        if why:
            skipped.append((var, ref_line, why))
        else:
            hits.append((var, ref_line, assign_line))
    return lines, hits, skipped


def main(argv):
    if "--all" in argv:
        names = sorted(n for n in os.listdir(REPO_ROOT) if n.endswith(".sh"))
    else:
        names = [a for a in argv if not a.startswith("-")] or DEFAULT_SCRIPTS

    total = 0
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            print("%-22s [skip] 不存在" % name)
            continue
        lines, hits, skipped = audit(path)
        tag = "✅ 干净" if not hits else "🔴 %d 处未解释" % len(hits)
        print("%-22s %-16s (%d 行，自动跳过 %d)" % (name, tag, len(lines), len(skipped)))
        for var, ref_line, why in skipped:
            print("      · 跳过 %-16s L%-4d %s" % (var, ref_line, why))
        for var, ref_line, assign_line in hits:
            tail = "L%d" % assign_line if assign_line else "从未赋值"
            print("    🔴 %-18s 首次裸引用 L%d  vs 首次赋值 %s" % (var, ref_line, tail))
            print("        %s" % lines[ref_line - 1].strip()[:96])
        total += len(hits)

    print("-" * 64)
    if total == 0:
        print("结论：干净 —— 无 `set -u` 下未赋值就被裸引用的变量。")
        return 0
    print("结论：%d 处未解释。真命中须把归一化赋值 `VAR=\"${VAR:-默认}\"` "
          "提到**首次引用之前**；确认安全的登记进 REVIEWED_SAFE。" % total)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
