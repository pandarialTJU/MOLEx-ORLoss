"""
Lightweight MoLEx (Mixture-of-LoRA-Experts) model that fine-tunes WavLM with
custom adapters and an LSTM classifier.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import str_to_bool
from WavLM import WavLM, WavLMConfig, TransformerSentenceEncoderLayer
from Attm_merge import AttM


class SimpleLoRA(nn.Module):
    """Minimal LoRA layer used inside each expert."""

    def __init__(self, in_features, out_features, r=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.lora_A = nn.Parameter(torch.empty(out_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=0.01)
        nn.init.zeros_(self.lora_B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        delta_w = self.lora_A @ self.lora_B
        return x @ (self.weight + delta_w).T + (self.bias if self.bias is not None else 0.0)


class Expert(nn.Module):
    """LoRA-based expert block."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lora_fc = SimpleLoRA(input_dim, input_dim, r=hidden_dim)

    def forward(self, x):
        return self.lora_fc(x)


class NoisyTopkRouter(nn.Module):
    """Router with deterministic tie breaking."""

    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        if self.training:
            noise_logits = self.noise_linear(mh_output)
            sigma = F.softplus(noise_logits)
            logits = logits + torch.randn_like(logits) * sigma

        eps = (1e-8) * torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
        noisy_logits = logits + eps

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    """Sparse MoE layer composed of LoRA experts."""

    def __init__(self, n_embed, num_experts, top_k, ex_hidden_dim):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, ex_hidden_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k
        self.ex_hidden_dim = ex_hidden_dim


    def forward(self, x):
        gating_output, indices = self.router(x)  # [seq, batch, experts], indices [seq, batch, top_k]
        batch_size, seq_len, feature_dim = x.shape
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, feature_dim)
        flat_gating_output = gating_output.view(-1, self.num_experts)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            expert_input = flat_x.clone()
            expert_input[~expert_mask.view(-1)] = 0

            expert_output = expert(expert_input)
            gating_scores = flat_gating_output[:, i].unsqueeze(-1)
            weighted_output = expert_output * gating_scores

            final_output += weighted_output.view(batch_size, seq_len, feature_dim)

        return final_output


class Transformer_MOE(TransformerSentenceEncoderLayer):
    """Transformer layer with an attached Sparse MoE branch."""

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        expert: int = 0,
        ex_hidden_dim: int = 16,
        topk: int = 2,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.num_experts = expert
        self.top_k = topk
        self.ex_hidden_dim = ex_hidden_dim
        self.smoe = SparseMoE(self.embedding_dim, self.num_experts, self.top_k, self.ex_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)

            expert_output = self.smoe(x)

            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            x = self.dropout3(x)
            x = residual + x + expert_output

        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            expert_output = self.smoe(x)

            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)

            x = residual + x + expert_output
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class LSTM_head(nn.Module):
    """Simple LSTM classifier head."""

    def __init__(self, d_args):
        super().__init__()
        self.input_dim = d_args["SSL_dim"]
        self.hidden_dim = 192
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, embed_192=False):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)

        if embed_192:
            return x

        return self.classifier(x)


class Model_MoLEx(nn.Module):
    """
    Default MoLEx model that swaps top Transformer blocks with MOE layers and
    keeps the lightweight LSTM classifier head.
    """

    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        self.out_dim = d_args["SSL_dim"]
        self.layer_num = d_args["SSL_layer_num"]
        self.num_expert = d_args["num_expert"]
        self.num_MOE_layer = d_args["num_MOE_layer"]
        self.ex_hidden_dim = d_args["expert_rank"]
        self.topk = d_args["topk"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        wavlm_checkpoint = d_args.get("wavlm_checkpoint") or os.environ.get("MOLEX_WAVLM_CHECKPOINT")
        if wavlm_checkpoint is None:
            raise ValueError(
                "Please set 'wavlm_checkpoint' in the model config or the MOLEX_WAVLM_CHECKPOINT environment variable."
            )

        checkpoint = torch.load(wavlm_checkpoint, map_location="cpu")
        self.cfg = WavLMConfig(checkpoint["cfg"])
        self.ssl_model = WavLM(self.cfg)

        self.ssl_model.encoder.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.cfg.encoder_embed_dim,
                    ffn_embedding_dim=self.cfg.encoder_ffn_embed_dim,
                    num_attention_heads=self.cfg.encoder_attention_heads,
                    dropout=self.cfg.dropout,
                    attention_dropout=self.cfg.attention_dropout,
                    activation_dropout=self.cfg.activation_dropout,
                    activation_fn=self.cfg.activation_fn,
                    layer_norm_first=self.cfg.layer_norm_first,
                    has_relative_attention_bias=(self.cfg.relative_position_embedding and i == 0),
                    num_buckets=self.cfg.num_buckets,
                    max_distance=self.cfg.max_distance,
                    gru_rel_pos=self.cfg.gru_rel_pos,
                )
                for i in range(self.layer_num - self.num_MOE_layer)
            ]
            + [
                Transformer_MOE(
                    embedding_dim=self.cfg.encoder_embed_dim,
                    ffn_embedding_dim=self.cfg.encoder_ffn_embed_dim,
                    num_attention_heads=self.cfg.encoder_attention_heads,
                    dropout=self.cfg.dropout,
                    attention_dropout=self.cfg.attention_dropout,
                    activation_dropout=self.cfg.activation_dropout,
                    activation_fn=self.cfg.activation_fn,
                    layer_norm_first=self.cfg.layer_norm_first,
                    has_relative_attention_bias=(self.cfg.relative_position_embedding and i == 0),
                    num_buckets=self.cfg.num_buckets,
                    max_distance=self.cfg.max_distance,
                    gru_rel_pos=self.cfg.gru_rel_pos,
                    expert=self.num_expert,
                    ex_hidden_dim=self.ex_hidden_dim,
                    topk=self.topk,
                )
                for i in range(self.layer_num - self.num_MOE_layer, self.layer_num)
            ]
        )

        self.msg = self.ssl_model.load_state_dict(checkpoint["model"], strict=False)
        self.ssl_model.encoder.layers = torch.nn.Sequential(*[self.ssl_model.encoder.layers[i] for i in range(self.layer_num)])
        self.ssl_model.feature_extractor.requires_grad_(False)

        finetune_ssl = str_to_bool(d_args.get("Finetune_id", "False"))
        if not finetune_ssl:
            for name, param in self.ssl_model.named_parameters():
                param.requires_grad = name in self.msg.missing_keys

        self.featfusion = AttM(d_args)
        self.decoder = LSTM_head(d_args)


    def forward(self, x):
        x = x.squeeze(-1)
        _, layer_outputs = self.ssl_model.extract_features(
            x, output_layer=self.ssl_model.cfg.encoder_layers, ret_layer_results=True
        )[0]
        x = torch.stack([i.transpose(0, 1) for i, _ in layer_outputs])
        x = x.transpose(0, 1)  # Bs x layers x Time x hidden
        x = x[:, 1:, :, :]  # drop conv features

        x = self.featfusion(x)
        x = self.decoder(x)
        return x

    def get_MOE_param_list(self):
        moe_params = []
        for name, param in self.ssl_model.named_parameters():
            if name in self.msg.missing_keys:
                moe_params.append(param)
        return moe_params
