from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):
    """Trie 约束解码。只允许 hash_dict 里登记的"下一个 SID 码"。

    🔴 2026-09-25 修复「吐违规 token」（RL 路径实测刷屏
       `No valid tokens found for hash_key [17] at step 1`，违规 id 全是 0..17）：
       本类原先把**不允许的位置**一律写成 `-inf`。这在 HF 的 `_beam_search` 里有漏洞 ——
       `utils._get_top_k_continuations` 每步会向
           torch.multinomial(softmax(scores), num_samples=max(2, 1+n_eos)*num_beams,
                             replacement=False)
       要一批**互不重复**的候选；而 SID 约束在 `c / \\n / eos` 这几位**只放行 1 个 token**
       ⟹ 支持集 < 需要的票数 ⟹ `torch.multinomial` 只能拿**零概率位置**凑数。
       实测这些凑数位置就是"扁平索引 0..k-1"（= beam0 的 token 0..k-1），落地后正是
       云端看到的那批 id（0,1,7..17，与 `2*num_beams` 同量级）。

       正常情况下这些凑数条目的分数是 `-inf`、排名垫底、**不会**被
       `_get_running_beams_for_next_iteration` 保留（已实测：三组配置下垃圾=0）。
       **但**只要某些 beam 的合法项分数本身不是有限值（模型给出 `-inf`），
       `#finite < num_beams`，凑数条目就会被保留进 beam ⟹ 序列被污染。

       修法（零回归）：合法项若分数非有限，**兜底成一个有限值** `row_max - 1000`，
       保证每个 beam 至少贡献 1 张"有限票"（`#finite >= num_beams` 恒成立）；
       不允许的位置仍是 `-inf`（排名绝对垫底，不引入新风险）。
       ⚠️ 合法项分数正常时**逐位不改**（clamp 不触发），因此 SFT/评估结果不变。
    """

    _ALLOW_FLOOR_GAP = 1000.0   # 合法项分数非有限时的兜底：row_max - 1000（有限即可）

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self._prev_len = -1          # 见 __call__ 里的"复用自愈"兜底
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 兜底自愈：若本实例被复用到第二轮 generate（序列长度回退），把 count 归零。
        # 生成过程中 cur_len 单调递增，故不会误触发。
        _cur_len = input_ids.shape[-1]
        if _cur_len <= self._prev_len:
            self.count = 0
        self._prev_len = _cur_len

        scores = torch.nn.functional.log_softmax(scores, dim=-1)

        # 合法项分数非有限时的兜底值：有限、且低于任何正常 logprob。
        _row_max = torch.nan_to_num(scores.amax(dim=-1, keepdim=True),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        allow_floor = _row_max - self._ALLOW_FLOOR_GAP

        mask = torch.full_like(scores, float('-inf'))

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                row = batch_id * self._num_beams + beam_id

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        _v = scores[row, self.eos_token_id]
                        mask[row, self.eos_token_id] = _v if torch.isfinite(_v) else allow_floor[row, 0]
                    continue 

                # 🔴 关键：合法项分数若为 -inf/NaN，抬到有限的 allow_floor，
                #    保证本 beam 至少贡献一张"有限票"。
                _v = scores[row, prefix_allowed_tokens]
                mask[row, prefix_allowed_tokens] = torch.where(
                    torch.isfinite(_v), _v, allow_floor[row, 0].expand_as(_v))

        self.count += 1

        # ⚠️ 注意：此处**不再** `scores + mask`。mask 已直接承载"最终分数"
        #    （合法位 = 原 logprob，禁止位 = -inf），等价于旧写法
        #    `scores + mask_old`（mask_old 合法位为 0）—— 分数正常时逐位相同。
        return mask