import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .backbones import CLIPTextEncoder, VPTCLIPVisionTransformer
from clip.clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
class PromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, clip_model, class_token_position='end', csc=False):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.text_projection.dtype
        device = clip_model.text_projection.device
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        if csc:
            # print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, device=device)
        else:
            # print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
        # nn.init.normal_(ctx_vectors, std=0.02)
        trunc_normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS (n_cls, 1, dim)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # EOS (n_cls, *, dim)

        self.csc = csc
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts # [n_cls, 77, C]
    
    def extra_repr(self):
        if self.csc:
            prompt_str = '(ctx): Parameter({}, {}, {})'.format(self.n_cls, self.n_ctx, self.ctx_dim)
        else:
            prompt_str = '(ctx): Parameter({}, {})'.format(self.n_ctx, self.ctx_dim)
        return prompt_str
    
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAProjection(nn.Module, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        n_class, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.text_projection = nn.Parameter(torch.empty(in_features, out_features))
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.text_projection.new_zeros((in_features, r)))
            self.lora_B = nn.Parameter(self.text_projection.new_zeros((n_class, r, out_features)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.text_projection.requires_grad = False
        self.reset_parameters()

    def extra_repr(self):
        extra_str = '(text_projection): Parameter({}, {})\n'.format(self.text_projection.shape[0], self.text_projection.shape[1]) + \
                    '(lora_A): Parameter({}, {})\n'.format(self.lora_A.shape[0], self.lora_A.shape[1]) + \
                    '(lora_B): Parameter({}, {}, {})'.format(self.lora_B.shape[0], self.lora_B.shape[1], self.lora_B.shape[2])
        return extra_str

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            trunc_normal_(self.lora_A, std=self.lora_A.shape[0] ** -0.5)
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        super(LoRAProjection, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                self.merged_projection = None
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.merged_projection = self.text_projection.unsqueeze(0) + \
                        torch.matmul(self.lora_A.unsqueeze(0), self.lora_B) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            if self.merged:
                return torch.matmul(x.unsqueeze(1), self.merged_projection).squeeze(1)
            else:
                result = torch.matmul(x, self.text_projection) # [B, C_out]
                lora_projection = torch.matmul(self.lora_A.unsqueeze(0), self.lora_B) # [B, C_in, C_out]
                result += torch.matmul(self.lora_dropout(x).unsqueeze(1), lora_projection).squeeze(1) * self.scaling
                return result
        else:
            return torch.matmul(x, self.text_projection)

class GFSS_Model(nn.Module):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 pretrained_clip,
                 base_class_names,
                 img_size,
                 patch_size=16,
                 exclude_key='prompt',
                 criterion=None,
                 is_ft=False,
                 lora_r=-1,
                 **args):
        super(GFSS_Model, self).__init__()
        embed_dim = 512
        n_ctx = 16
        if lora_r < 0: # use default lora_r
            lora_r = 16 if len(base_class_names) > 20 else 8
        self.text_encoder = CLIPTextEncoder(pretrained=pretrained_clip) # frozen
        num_tokens = 100 if len(base_class_names) > 20 else 10
        self.image_encoder = VPTCLIPVisionTransformer(input_resolution=img_size, patch_size=patch_size, pretrained=pretrained_clip, num_tokens=num_tokens) # prompt tuning

        self.base_class_names = base_class_names
        self.n_base = len(base_class_names)
        self.base_prompt_learner = PromptLearner(n_ctx, base_class_names, self.text_encoder, csc=False)
        self.base_projection = LoRAProjection(embed_dim, embed_dim, self.n_base, r=lora_r)
        self.balancer = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, 1)
            )

        self.is_ft = is_ft
        self.gamma = 0.1
        self.n_ctx = n_ctx
        self.lora_r = lora_r
        self.embed_dim = embed_dim
        self.criterion = criterion
        self.init_parameters()

        if self.is_ft:
            self._freeze_stages(self.text_encoder)
            self._freeze_stages(self.image_encoder)
            self._freeze_stages(self.base_prompt_learner)
            self._freeze_stages(self.base_projection)
            self._freeze_stages(self.balancer)
        else:
            self._freeze_stages(self.text_encoder)
            self._freeze_stages(self.image_encoder, exclude_key=exclude_key)

    def set_novel(self, novel_class_names):
        self.novel_class_names = novel_class_names
        self.n_novel = len(novel_class_names)
        self.novel_prompt_learner = PromptLearner(self.n_ctx, novel_class_names, self.text_encoder, csc=False)
        self.novel_projection = LoRAProjection(self.embed_dim, self.embed_dim, self.n_novel, r=self.lora_r)

        self.novel_projection.lora_A.requires_grad = False
        self._freeze_stages(self.novel_prompt_learner)

    def register_base(self):
        with torch.no_grad():
            base_emb = self.prompt_text_embedding(self.base_prompt_learner) # [C_base, C]
        self.register_buffer("base_emb", base_emb)

    def init_novel(self):
        self.novel_projection.text_projection.data.copy_(self.text_encoder.text_projection.data)        
        self.novel_prompt_learner.ctx.data.copy_(self.base_prompt_learner.ctx.data)
        self.novel_projection.lora_A.data.copy_(self.base_projection.lora_A.data)
        with torch.no_grad():
            prompts = self.novel_prompt_learner()
            tokenized_prompts = self.novel_prompt_learner.tokenized_prompts
            text_embeddings = self.text_encoder.forward_prompt(prompts, tokenized_prompts, projection=False)
        self.register_buffer("novel_text_embeddings", text_embeddings) # [C_novel, C]

    def merge_novel(self):
        self.base_projection.lora_B.data = torch.cat([self.base_projection.lora_B.data, self.novel_projection.lora_B.data], dim=0)
        new_base_prompt_learner = PromptLearner(self.n_ctx, self.base_class_names+self.novel_class_names, self.text_encoder, csc=False)
        new_base_prompt_learner.ctx.data.copy_(self.base_prompt_learner.ctx.data)
        self.base_prompt_learner = new_base_prompt_learner
        self._freeze_stages(self.base_prompt_learner)

    def extra_repr(self):
        extra_str = '(gamma): gamma={}'.format(self.gamma)
        return extra_str

    def init_parameters(self):
        self.base_projection.text_projection.data.copy_(self.text_encoder.text_projection.data)

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep some layers freezed."""
        super().train(mode)
        if mode:
            self.text_encoder.eval()
            if self.is_ft:
                self.image_encoder.eval()
            else:
                self.image_encoder.train()

    def params_to_optimize(self, base_lr):
        # parameters to optimize
        names_frozen = list()
        names_learnable = list()
        params_decay = list()
        for name, m in self.named_parameters():
            if m.requires_grad:
                names_learnable.append(name)
                params_decay.append(m)
            else:
                names_frozen.append(name)

        params_to_optimize = [
            {'params': params_decay, 'lr': base_lr}
        ]
        # print('LEARNABLE params: ', names_learnable)
        return params_to_optimize, names_learnable
    
    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def prompt_text_embedding(self, prompt_learner, is_novel=False):
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts
        text_embeddings = self.text_encoder.forward_prompt(prompts, tokenized_prompts, projection=False)
        if is_novel:
            prompt_embeddings = self.novel_projection(text_embeddings)
        else:
            prompt_embeddings = self.base_projection(text_embeddings)
        return prompt_embeddings # [n_cls, C]

    def visual_embedding(self, img):
        """Extract features from images."""
        visual_feat = self.image_encoder(img, norm=False)
        return visual_feat

    @torch.cuda.amp.autocast(enabled=False)
    def orthogonal_decompose(self, feats, bases_b, bases_n=None, return_sim=False):
        '''
            feats: [BxCxN]
            bases_b: [1xKxC]
            bases_n: [1xKxC]
            ---
            out_fg:   [BxKxCxN]
            out_bg:   [Bx1xCxN]
        '''
        B = feats.shape[0]
        f = feats.to(torch.float) # [B, C, N]
        b_base = F.normalize(bases_b.to(torch.float), p=2, dim=-1).expand(B, -1, -1) # [B, Kb, C]

        proj_base = torch.matmul(b_base, f) # [B, Kb, N]
        out_fg_b = proj_base.unsqueeze(2) * b_base.unsqueeze(-1) # [B, Kb, C, N]
        residual = f - out_fg_b.sum(1) # [B, C, N]

        if bases_n is not None:
            b_novel = F.normalize(bases_n.to(torch.float), p=2, dim=-1).expand(B, -1, -1) # [B, Kn, C]
            proj_novel = torch.matmul(b_novel, f) # [B, Kn, N]

            out_fg_n = proj_novel.unsqueeze(2) * b_novel.unsqueeze(-1) # [B, Kn, C, N]
            residual = residual - out_fg_n.sum(1)# [B, C, N]

            b_bg = F.normalize(residual.mean(-1), p=2, dim=-1).unsqueeze(1) # [B, 1, C]
            proj_bg = torch.matmul(b_bg, f) # [B, 1, N]

            b_all = torch.cat([b_bg, b_base, b_novel], dim=1) # [B, 1+Kb+Kn, C]
            proj_all = torch.cat([proj_bg, proj_base, proj_novel], dim=1) # [B, 1+Kb+Kn, N]

            weights_pos = self.balancer(b_all) # [B, 1+Kb+Kn, 1]
            weights_neg = self.balancer(-b_all) # [B, 1+Kb+Kn, 1]
            pred_all = F.relu(proj_all) * weights_pos + F.relu(-proj_all) * weights_neg # [B, 1+Kb+Kn, N]

            if return_sim:
                sim_all = torch.matmul(b_all, b_all.transpose(1,2)) # [B, 1+Kb+Kn, 1+Kb+Kn]
                return pred_all / self.gamma, sim_all
            return pred_all / self.gamma
        else:
            b_bg = F.normalize(residual.mean(-1), p=2, dim=-1).unsqueeze(1) # [B, 1, C]
            proj_bg = torch.matmul(b_bg, f) # [B, 1, N]

            b_all = torch.cat([b_bg, b_base], dim=1) # [B, 1+Kb, C]
            proj_all = torch.cat([proj_bg, proj_base], dim=1) # [B, 1+Kb, N]

            weights_pos = self.balancer(b_all) # [B, 1+Kb, 1]
            weights_neg = self.balancer(-b_all) # [B, 1+Kb, 1]
            pred_all = F.relu(proj_all) * weights_pos + F.relu(-proj_all) * weights_neg # [B, 1+Kb, N]

            return pred_all / self.gamma

    def forward(self, img=None, img_feats=None, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            if img_feats is not None:
                return self.forward_all_feats(img_feats, mask)
            return self.forward_all(img, mask)
        else:
            return self.forward_base(img, mask)

    def forward_all_feats(self, img_feats, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        visual_feat = img_feats
        B, C, h, w = visual_feat.shape

        base_emb = self.base_emb # [C_base, C]
        base_emb = base_emb.unsqueeze(0) # [1, C_base, C]
        novel_emb = self.novel_projection(self.novel_text_embeddings) # [C_novel, C]
        novel_emb = novel_emb.unsqueeze(0) # [1, C_novel, C]

        n_base = base_emb.shape[1] 
        n_novel = novel_emb.shape[1]
        features_full = visual_feat.flatten(2) # [BxCxN]

        with torch.cuda.amp.autocast(enabled=False):
            pred_all = self.orthogonal_decompose(features_full, base_emb, novel_emb)
            pred_all = pred_all.reshape(B, 1+n_base+n_novel, h, w) # [B, 1+Kb+Kn, h, w]
            if self.criterion is not None and mask is not None:                
                base_emb = F.normalize(base_emb.squeeze(0).to(torch.float), p=2, dim=-1)
                novel_emb = F.normalize(novel_emb.squeeze(0).to(torch.float), p=2, dim=-1)
                all_emb = torch.cat([novel_emb, base_emb], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]
                return self.criterion(pred_all, mask, is_ft=True, proto_sim=proto_sim)
            else:
                return pred_all

    def forward_all(self, img, mask=None, return_sim=False):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        visual_feat, visual_cls = self.visual_embedding(img) # [B, C, H, W], [B, C]
        visual_feat = visual_feat[-1]
        B, C, h, w = visual_feat.shape

        base_emb = self.base_emb # [C_base, C]
        base_emb = base_emb.unsqueeze(0) # [1, C_base, C]
        novel_emb = self.novel_projection(self.novel_text_embeddings) # [C_novel, C]
        novel_emb = novel_emb.unsqueeze(0) # [1, C_novel, C]

        n_base = base_emb.shape[1] 
        n_novel = novel_emb.shape[1]
        features_full = visual_feat.flatten(2) # [BxCxN]

        with torch.cuda.amp.autocast(enabled=False):
            pred_all = self.orthogonal_decompose(features_full, base_emb, novel_emb, return_sim=return_sim)
            if return_sim:
                pred_all, proto_sim = pred_all
            pred_all = pred_all.reshape(B, 1+n_base+n_novel, h, w) # [B, 1+Kb+Kn, h, w]
            if self.criterion is not None and mask is not None:                
                base_emb = F.normalize(base_emb.squeeze(0).to(torch.float), p=2, dim=-1)
                novel_emb = F.normalize(novel_emb.squeeze(0).to(torch.float), p=2, dim=-1)
                all_emb = torch.cat([novel_emb, base_emb], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]
                return self.criterion(pred_all, mask, is_ft=True, proto_sim=proto_sim)
            else:
                if return_sim:
                    return pred_all, proto_sim
                return pred_all

    def forward_base(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        B = img.shape[0]
        visual_feat_q, visual_cls_q = self.visual_embedding(img) # [B, C, H, W], [B, C]
        text_feat = self.prompt_text_embedding(self.base_prompt_learner) # [C_base, C]
        query_emb = text_feat.unsqueeze(0) # [1, C_base, C]

        visual_feat_q = visual_feat_q[-1]
        B, C, h, w = visual_feat_q.shape
        n_class = 1 + query_emb.shape[1]
        visual_feat_q = visual_feat_q.flatten(2) # [B, C, N]

        with torch.cuda.amp.autocast(enabled=False):
            pred_all = self.orthogonal_decompose(visual_feat_q, query_emb)
            pred_all = pred_all.reshape(B, n_class, h, w) # [B, 1+K, h, w]
            if self.training:
                text_feat = F.normalize(text_feat.to(torch.float), p=2, dim=-1)
                proto_sim = torch.matmul(text_feat, text_feat.t()) # [KbasexKbase]
                return self.criterion(pred_all, mask, proto_sim=proto_sim)
            else:
                return pred_all