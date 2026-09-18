#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提示词模板的**唯一读取入口**（配 `config/prompt_templates.json` 单一真源）。

为什么要有这个模块（2026-09-18 收敛）：
    在此之前同一套模板有 **3 份实现**，改一处忘一处就是静默漂移：
      ① `data.py` 的 `generate_prompt()` / 各 Dataset 里的 f-string   ← 训练真正在用（事实权威）
      ② `scripts/data/prepare_sft_data.py` 的 `PROMPT_TEMPLATES` → `info/prompt_templates.json`
         （自称"训练端直接读、防漂移"，但**当时没有任何训练代码读它**）
      ③ `scripts/data/build_sft_prompts.py` 自带的 V_*（verbatim）+ 定版两套
    现在收敛为：**`config/prompt_templates.json`（git 跟踪）+ 本模块**。
    ① 改成 import 本模块；② 只负责把 config 拷进每个域的快照；
    **③ 连同它产出的 `prompts/`（2.6 GB）与 `tasks/`（0.36 GB）一起删除** ——
    零消费方，且它的唯一用途（预渲染出 alpaca/chatml 两套明文供格式消融）已被
    `PROMPT_FORMAT=alpaca` 这个运行时开关取代（见 `effective_format`）。

两种取模板的方式（都有意保留）：
    • `config/prompt_templates.json` —— 仓库里的**当前**口径（默认）
    • `<sft>/info/prompt_templates.json` —— 建数据时**拷进去的快照**（锚定 data_path 时优先用）
      这样旧数据永远用它建时的口径，不会被后来的模板改动污染。

只依赖 stdlib，可被 data.py / evaluate.py / minionerec_trainer.py / scripts 任意 import。
"""
import io
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CONFIG = os.path.join(_HERE, "config", "prompt_templates.json")
SNAPSHOT_NAME = os.path.join("info", "prompt_templates.json")
ENV_OVERRIDE = "PROMPT_TEMPLATES_JSON"
ENV_FORMAT = "PROMPT_FORMAT"   # 覆盖 default_format（格式消融用：PROMPT_FORMAT=alpaca bash sft_run0.sh）

_cache = {}
_anchor = [None]          # 模块级默认锚点（由 set_anchor 设置）
REQUIRED = ("schema_version", "default_format", "formats", "response_prefix", "tasks", "completion")
SCHEMA_VERSION = 2


def _validate(doc, path):
    """拦下"旧格式快照"：v1 的 info/prompt_templates.json 只有 alpaca_header+tasks，
    缺 default_format/formats/response_prefix/completion —— 直接 KeyError 很难查。"""
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        raise ValueError(
            f"模板文件 {path} 是旧格式（缺 {missing}）。\n"
            f"  schema_version 需为 {SCHEMA_VERSION}，真源在 config/prompt_templates.json。\n"
            f"  修法：重新跑 scripts/data/prepare_sft_data.py（它会把 config 拷成快照），\n"
            f"        或设 PROMPT_TEMPLATES_JSON 指向 config/prompt_templates.json 临时绕过。")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"模板文件 {path} 的 schema_version={doc.get('schema_version')}，"
                         f"本代码要求 {SCHEMA_VERSION}")
    return doc


# --------------------------------------------------------------------- 定位
def _find_snapshot(anchor, max_up=4):
    """从 anchor（一个数据文件或目录路径）向上找 `<...>/sft/info/prompt_templates.json`。"""
    p = os.path.abspath(str(anchor))
    if os.path.isfile(p):
        p = os.path.dirname(p)
    for _ in range(max_up + 1):
        cand = os.path.join(p, SNAPSHOT_NAME)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def resolve_path(anchor=None):
    """决定这次用哪个模板文件。优先级：env > anchor 快照 > 仓库 config。"""
    env = os.environ.get(ENV_OVERRIDE)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"{ENV_OVERRIDE}={env!r} 不存在")
        return env
    snap = _find_snapshot(anchor if anchor is not None else _anchor[0])
    if snap:
        return snap
    if not os.path.isfile(REPO_CONFIG):
        raise FileNotFoundError(
            f"找不到模板文件：既没有快照（anchor={anchor!r}），也没有 {REPO_CONFIG}")
    return REPO_CONFIG


def set_anchor(anchor):
    """设置模块级默认锚点（data.py 的 Dataset 在自己 __init__ 里调用）。"""
    _anchor[0] = anchor


def load(anchor=None, path=None):
    """读模板 dict（按文件路径缓存）。"""
    if path is None:
        path = resolve_path(anchor)
    key = os.path.abspath(path)
    if key not in _cache:
        with io.open(key, encoding="utf-8") as f:
            _cache[key] = _validate(json.load(f), key)
    return _cache[key]


def source(anchor=None):
    """当前实际用的是哪个文件（排查漂移时很有用）。"""
    return resolve_path(anchor)


def effective_format(t):
    """本次实际生效的格式：env `PROMPT_FORMAT` > JSON 的 `default_format`。

    存在的意义：格式消融（Run-1 之后要做）不再需要"预渲染两套数据集"，
    只要 `PROMPT_FORMAT=alpaca bash sft_run0.sh` —— 训练端与评估端（都走本模块）自动一致。
    """
    fmt = os.environ.get(ENV_FORMAT) or t["default_format"]
    if fmt not in t["formats"]:
        raise ValueError(f"{ENV_FORMAT}={fmt!r} 不在可用格式 {list(t['formats'])} 中")
    return fmt


# --------------------------------------------------------------------- 取值
def default_format(anchor=None, path=None):
    return effective_format(load(anchor, path))


def tasks(anchor=None, path=None):
    return load(anchor, path)["tasks"]


def task_input(task, fmt=None, anchor=None, path=None, **fields):
    """取某个任务的 input 文本。fmt='verbatim' 且该任务有 verbatim_input 覆盖时用覆盖值。"""
    t = load(anchor, path)
    spec = t["tasks"][task]
    tmpl = spec["input"]
    if (fmt or effective_format(t)) == "verbatim" and spec.get("verbatim_input"):
        tmpl = spec["verbatim_input"]
    return tmpl.format(**fields) if fields else tmpl


def task_instruction(task, anchor=None, path=None):
    return load(anchor, path)["tasks"][task]["instruction"]


def task_target(task, anchor=None, path=None):
    return load(anchor, path)["tasks"][task]["target"]


def response_prefix(fmt=None, anchor=None, path=None):
    """响应前缀 —— 约束解码（Trie / LogitProcessor / ReReTrainer）构建 key 时必须用这个，
    不能用别处手抄的字符串。"""
    t = load(anchor, path)
    fmt = fmt or effective_format(t)
    return t["response_prefix"][fmt]


def completion_suffix(anchor=None, path=None):
    """监督段尾 = sid_sentinel + eos_token（Trie 的 step3/step4 硬依赖）。"""
    c = load(anchor, path)["completion"]
    return c["sid_sentinel"] + c["eos_token"]


# --------------------------------------------------------------------- 渲染
def wrap_input(task, input_text, fmt=None, anchor=None, path=None):
    """把**已构造好的 input 文本**包进指定格式的骨架（不再从模板求 input）。

    用在 input 不是简单格式化、而是由别处算出来的场合
    （如 RLSeqTitle2SidDataset 的 title 序列、FusionSeqRecDataset 的 history_str）。
    """
    t = load(anchor, path)
    fmt = fmt or effective_format(t)
    fs = t["formats"][fmt]
    instruction = t["tasks"][task]["instruction"]
    if fs["kind"] == "chat":
        body = fs["user_body"].format(instruction=instruction, input=input_text)
        return (fs["system_prefix"] + t["system_msg"] + fs["system_suffix"]
                + fs["user_prefix"] + body + fs["user_suffix"]
                + fs["assistant_prefix"])
    return fs["header"] + fs["body"].format(instruction=instruction, input=input_text)


def render(task, fmt=None, anchor=None, path=None, **fields):
    """渲染完整 prompt（不含 completion）。`fields` 填模板里的占位符（hist/sid/title/text）。"""
    input_text = task_input(task, fmt=fmt, anchor=anchor, path=path, **fields)
    return wrap_input(task, input_text, fmt=fmt, anchor=anchor, path=path)


def render_completion(sid_or_title, anchor=None, path=None):
    """监督段的文本形式：`<内容>` + 哨兵 + eos。SID 任务 5 token；title 任务长度不定。"""
    return str(sid_or_title) + completion_suffix(anchor=anchor, path=path)


def describe(anchor=None, path=None):
    """自检用：打印当前真源与关键口径。"""
    p = resolve_path(anchor)
    t = load(path=p)
    lines = [
        f"  真源文件      : {p}",
        f"  schema_version: {t.get('schema_version')}",
        f"  default_format: {t['default_format']}"
        + (f"  -> 生效 {effective_format(t)}（由 {ENV_FORMAT} 覆盖）"
           if effective_format(t) != t["default_format"] else ""),
        f"  可用格式      : {list(t['formats'])}",
        f"  任务          : {list(t['tasks'])}",
    ]
    for f in t["formats"]:
        lines.append(f"  response_prefix[{f}] = {response_prefix(f, path=p)!r}")
    c = t["completion"]
    lines.append(f"  completion    : sentinel={c['sid_sentinel']!r} eos={c['eos_token']!r}")
    return chr(10).join(lines)


if __name__ == "__main__":
    print(describe())
