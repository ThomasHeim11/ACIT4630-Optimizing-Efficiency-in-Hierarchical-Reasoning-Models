from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.hrm.error_singals import get_error_signal  # SHREK: error signal module


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    # SHREK: prev_pred stores last step's argmax predictions for flip rate computation.
    # zeros = fresh start (first step after init or reset gives flip_rate ≈ 1.0)
    prev_pred: torch.Tensor        # (B, seq_len) int32
    # SHREK: cached Q-values from previous ACT step.
    # used instead of running inner() a second time — removes the double forward pass.
    prev_q_halt: torch.Tensor      # (B,) float32
    prev_q_continue: torch.Tensor  # (B,) float32


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # SHREK: q_head takes hidden_size + 1 because we append the stagnation delta scalar
        # (stagnation delta = how much z_H changed this step — tells the halt decision if the model is stuck)
        self.q_head       = CastedLinear(self.config.hidden_size + 1, 2, bias=True)

        # SHREK: error_encoder maps the scalar error score -> hidden_size vector for injection into z_H
        # alpha is a learned gate that starts near 0 so the model first learns the base task,
        # then gradually learns to use the error signal as training progresses
        self.error_encoder  = nn.Linear(1, self.config.hidden_size)
        self.alpha          = nn.Parameter(torch.tensor(0.01))
        # SHREK: error_estimator reads z_H and predicts how wrong the model is.
        # trained via auxiliary loss in pretrain.py using the real lm_loss as target.
        # catches "stuck but wrong" — a model confidently on the wrong answer.
        # flip rate catches oscillation; estimator catches confident-but-wrong.
        self.error_estimator = nn.Linear(self.config.hidden_size, 1)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            # SHREK: zeros = no previous prediction — first step gives flip_rate ≈ 1.0
            # device=H_init.device ensures prev_pred is on CUDA, matching z_H and logits
            prev_pred=torch.zeros(batch_size, self.config.seq_len, dtype=torch.int32, device=self.H_init.device),
            # SHREK: init Q cache to -5.0 matching q_head bias init — starts at low confidence
            # device=H_init.device ensures these are on CUDA, matching reset_flag in reset_carry()
            prev_q_halt=torch.full((batch_size,), -5.0, device=self.H_init.device),
            prev_q_continue=torch.full((batch_size,), -5.0, device=self.H_init.device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry, use_default=True):
        # SHREK: zero out prev_pred for reset sequences so they start fresh.
        # a reset sequence is one that just halted — it will solve a new puzzle next.
        # zeroing prev_pred means first step gives flip_rate ≈ 1.0 (maximum uncertainty).
        new_prev_pred = torch.where(
            reset_flag.view(-1, 1),
            torch.zeros_like(carry.prev_pred),
            carry.prev_pred
        )
        # SHREK: reset cached Q-values for reset sequences back to low confidence (-5.0)
        new_prev_q_halt     = torch.where(reset_flag, torch.full_like(carry.prev_q_halt,     -5.0), carry.prev_q_halt)
        new_prev_q_continue = torch.where(reset_flag, torch.full_like(carry.prev_q_continue, -5.0), carry.prev_q_continue)

        if use_default:
            return HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
                z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
                prev_pred=new_prev_pred,           # SHREK: carry forward or zero on reset
                prev_q_halt=new_prev_q_halt,       # SHREK: carry forward or reset to -5.0
                prev_q_continue=new_prev_q_continue,
            )
        else:
            # SHREK: removed AugmentedHRM random perturbation (trunc_normal noise on reset).
            # Random noise is replaced by error-conditioned injection in the forward pass,
            # which gives the model informed feedback instead of a random push.
            # Reset to clean default init — same as use_default=True.
            return HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
                z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
                prev_pred=new_prev_pred,           # SHREK: carry forward or zero on reset
                prev_q_halt=new_prev_q_halt,       # SHREK: carry forward or reset to -5.0
                prev_q_continue=new_prev_q_continue,
            )


    # SHREK: removed task_type parameter — error signal is now universal (no task rules needed)
    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], require_trace=False):
    #  -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H_trace = []

        # SHREK: store z_H from the START of this ACT step to compute stagnation delta later.
        # stagnation delta = how much z_H changed during this full reasoning step.
        # if z_H barely changed, the model is stuck and the Q-head should know that.
        z_H_start = carry.z_H

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)
                    if require_trace:
                        z_H_trace.append(z_H.detach().cpu().clone())

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        if require_trace:
            z_H_trace.append(z_H.detach().cpu().clone())

        # LM Outputs — decode z_H into token predictions
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]   # (B, seq_len, vocab_size)

        # SHREK Component 1: Combined Error Signal
        # Signal A — flip rate: what fraction of tokens changed from last step?
        #   catches oscillation (model changing its mind between wrong options)
        flip_err, current_pred = get_error_signal(output, carry.prev_pred)     # (B,), (B, seq_len)

        # Signal B — learned estimator: reads z_H and predicts how wrong the model is.
        #   catches stuck-but-wrong (model confidently on wrong answer without oscillating)
        #   average over content positions (skip puzzle embedding prefix positions)
        z_H_mean    = z_H[:, self.puzzle_emb_len:].mean(dim=1)                 # (B, hidden_size)
        learned_err = torch.sigmoid(self.error_estimator(z_H_mean.float()))    # (B, 1)
        learned_err = learned_err.squeeze(-1)                                  # (B,)

        # SHREK: combined error = 50/50 blend of both signals
        # flip_err works immediately from step 1 (no learning needed)
        # learned_err becomes accurate over training and takes over as the stronger signal
        error = 0.5 * flip_err + 0.5 * learned_err                            # (B,)

        # SHREK: inject combined error into z_H
        # error_encoder maps scalar -> hidden_size vector
        # alpha (clamped 0-1) controls injection strength — starts at 0.01
        error_emb = self.error_encoder(error.unsqueeze(-1))                    # (B, hidden_size)
        with torch.no_grad():
            self.alpha.clamp_(0.0, 1.0)                                        # SHREK: keep alpha in safe range
        z_H = z_H + (self.alpha * error_emb.unsqueeze(1)).to(z_H.dtype)       # (B, seq_len, hidden_size)

        # SHREK Component 2: Stagnation Delta for Q-head
        # measure how much z_H changed compared to when this ACT step started.
        # small delta = model is stuck in the same state = stagnation signal.
        # we compute this in float32 for numerical precision, then pass to Q-head.
        z_H_f     = z_H.float()                                                # (B, seq_len, hidden_size)
        z_start_f = z_H_start.float()                                          # (B, seq_len, hidden_size)
        delta     = (z_H_f - z_start_f).norm(dim=(1, 2)) / \
                    (z_start_f.norm(dim=(1, 2)) + 1e-6)                        # (B,)

        # Q head: append stagnation delta to the CLS token before the halt decision.
        # CLS token (position 0) summarises the full sequence state.
        # delta tells the Q-head "I moved this much — am I still making progress?"
        cls_token = z_H[:, 0].to(torch.float32)                                # (B, hidden_size)
        q_input   = torch.cat([cls_token, delta.unsqueeze(-1)], dim=-1)        # (B, hidden_size + 1)
        q_logits  = self.q_head(q_input).to(torch.float32)                     # (B, 2)

        # New carry: store error-injected z_H so next ACT step starts from it
        # SHREK: q_logits must be computed before new_carry so we can cache them
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            # SHREK: store current predictions — next step compares against these for flip rate
            prev_pred=current_pred.detach(),
            # SHREK: cache Q-values — outer forward uses these instead of running inner() again
            prev_q_halt=q_logits[..., 0].detach(),
            prev_q_continue=q_logits[..., 1].detach(),
        )

        # SHREK: also return learned_err so pretrain.py can compute auxiliary loss
        # auxiliary loss trains the estimator: predicted error should match real lm_loss
        if require_trace:
            return z_H_trace, new_carry, output, (q_logits[..., 0], q_logits[..., 1]), learned_err
        else:
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), learned_err


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    # SHREK: removed task_type parameter — error signal is universal, no task rules needed
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor], require_trace=False):
        # -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor], torch.Tensor]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # SHREK: unpack learned_err from inner forward — needed for auxiliary loss in pretrain.py
        if require_trace:
            z_H_trace, new_inner_carry, logits, (q_halt_logits, q_continue_logits), learned_err = self.inner(new_inner_carry, new_current_data, require_trace=require_trace)
        else:
            new_inner_carry, logits, (q_halt_logits, q_continue_logits), learned_err = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "learned_err": learned_err,   # SHREK: for auxiliary loss in pretrain.py
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            # is_last_step = new_steps >= 32
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # SHREK: removed second inner() call — was doubling training compute.
                # original HRM ran the full model twice to get next-step Q-values.
                # we use the Q-values cached in new_inner_carry.prev_q_halt/continue instead.
                # these were stored at the end of the inner forward just completed above.
                # one-step delayed Q target is a valid approximation (similar to DQN target network).
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(is_last_step,
                        new_inner_carry.prev_q_halt,
                        torch.maximum(new_inner_carry.prev_q_halt, new_inner_carry.prev_q_continue))
                )

        if require_trace:
            return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs, new_steps, (q_halt_logits > q_continue_logits), z_H_trace
        else:
            return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs, new_steps, (q_halt_logits > q_continue_logits)
