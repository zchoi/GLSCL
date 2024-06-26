import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
allgather = AllGather.apply
allgather2 = AllGather2.apply


def margin_ranking_loss(
        similary_matrix, 
        margin=None, 
        direction= 'both', 
        average_batch = True, 
        ):

    batch_size = similary_matrix.size(0)
    diagonal = similary_matrix.diag().view(batch_size, 1)

    pos_mask = torch.eye(batch_size,batch_size,device=similary_matrix.device).bool()

    # v2c
    if direction == 'both' or direction == 's2n':
        diagonal_1 = diagonal.expand_as(similary_matrix)
        
        cost_cap = (margin + similary_matrix - diagonal_1).clamp(min=0)
        cost_cap = cost_cap.masked_fill(pos_mask, 0)
        if average_batch:
            cost_cap = cost_cap / (batch_size * (batch_size - 1))
            cost_cap = torch.sum(cost_cap)

    # c2v
    if direction == 'both' or direction == 'n2s':
        diagonal_2 = diagonal.t().expand_as(similary_matrix)
        cost_vid = (margin + similary_matrix - diagonal_2).clamp(min=0)
        cost_vid = cost_vid.masked_fill(pos_mask,0)
        if average_batch:
            cost_vid = cost_vid / (batch_size * (batch_size - 1))
            cost_vid = torch.sum(cost_vid)
    
    if direction == 'both':
        return cost_cap + cost_vid
    elif direction == 's2n':
        return cost_cap
    else:
        return cost_vid


class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class DiCoSA(nn.Module):
    def __init__(self, config):
        super(DiCoSA, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        # for n, p in self.clip.named_parameters():
        #     if "clip.visual" in n:
        #         p.requires_grad = False
        
        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
        
        width = int(transformer_width // self.config.center)
        self.weight_fc = nn.Sequential(
                    nn.Linear(2*width, 4*width), nn.ReLU(inplace=True),
                    nn.Linear(4*width, 1))
            
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        self.mse = nn.MSELoss()
        
        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.num_queries = self.config.query_number
        self.noun_queries = nn.Parameter(torch.rand(self.num_queries, transformer_width))
        self.spatial_queries = nn.Parameter(torch.rand(self.num_queries, transformer_width))
        # self.spatial_queries = self.noun_queries.clone()

        self.decoder_layer_noun = nn.TransformerDecoderLayer(d_model=transformer_width, nhead=8)
        self.transformer_decoder_noun = nn.TransformerDecoder(self.decoder_layer_noun, num_layers=self.config.cross_att_layer)

        self.decoder_layer_spatial = nn.TransformerDecoderLayer(d_model=transformer_width, nhead=8)
        self.transformer_decoder_spatial = nn.TransformerDecoder(self.decoder_layer_spatial, num_layers=3)

        self.compact_s = nn.Linear(transformer_width, transformer_width // 8)
        self.compact_n = nn.Linear(transformer_width, transformer_width // 8)
        
        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            M_t2v_logits, M_v2t_logits, Inter_Diversity, Intra_Consistency = self.get_similarity_logits(text_feat, cls, video_feat,
                                                                    text_mask, video_mask, shaped=True)
            
            M_loss_t2v = self.loss_fct(M_t2v_logits * logit_scale)
            M_loss_v2t = self.loss_fct(M_v2t_logits * logit_scale)
            M_loss = (M_loss_t2v + M_loss_v2t) / 2
            
            score2, _, _ = self._score(text_feat, cls, video_feat, text_mask, video_mask)
            
            score2_t2v = self.loss_fct(score2 * logit_scale)
            score2_v2t = self.loss_fct(score2.T * logit_scale)
            
            Score_2_loss = (score2_t2v + score2_v2t) / 2
            
            loss = M_loss + Intra_Consistency + Inter_Diversity
            return loss, M_loss, Inter_Diversity, Intra_Consistency
        else:
            return None
        
    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        M_t2v_logits, M_v2t_logits, Inter_Diversity, Intra_Consistency = self.similarity(text_feat, cls, video_feat, text_mask, video_mask)
        
        return M_t2v_logits, M_v2t_logits, Inter_Diversity, Intra_Consistency
    
    def similarity(self, text_feat, cls, video_feat, text_mask, video_mask):
        '''
        text_feat: torch.Size([128, 32, 512])
        cls: torch.Size([128, 512])
        video_feat: torch.Size([128, 12, 512])
        print(text_feat.shape, cls.shape, video_feat.shape)
        self.noun_queries: torch.rand(self.num_queries, transformer_width)
        self.spatial_queries: torch.rand(self.num_queries, transformer_width)
        video_feat: torch.Size([128, 12, 512])
        print(cls.shape, video_feat.shape)
        '''
        
        v_weight = torch.einsum('ad,bvd->abv', [cls, video_feat]) # bs 512, bs 12 512 -> bs bs 12
        v_weight = torch.softmax(v_weight / self.config.temp, dim=-1)
        v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])  
        video_feat_t_cond = torch.einsum('abv,bvd->abd', [v_weight, video_feat]) # bs bs 12, bs 12 512 -> bs bs 512

        a, b = cls.size(0), video_feat_t_cond.size(1)
        cls, video_feat_t_cond = cls.contiguous(), video_feat_t_cond.contiguous()
        t_feat = cls.view(a, self.config.center, -1)
        v_feat = video_feat_t_cond.view(a, b, self.config.center, -1)
        d = t_feat.size(2)
        
        temp = torch.cat([t_feat.unsqueeze(1).repeat(1, b, 1, 1), v_feat], dim=-1) # bs bs 1 512*2
        weight = self.weight_fc(temp).squeeze(3)  # a b c 2d-> a b c
        
        _t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        _v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        
        # !不要weighting
        retrieve_logits = torch.einsum('acd,abcd->abc', [_t_feat, _v_feat]).squeeze()       # bs 1 512, bs bs 1 512 -> bs bs 1
        # retrieve_logits = torch.einsum('abc,abc->ab', [retrieve_logits, weight])  # bs bs 1 * bs bs 1 -> bs bs 1
        
        # t_feat = cls
        # v_feat = video_feat.mean(1)
        # _t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        # _v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        # retrieve_logits = torch.einsum('ac,bc->ab', [_t_feat, _v_feat])
        
        if self.config.add_query_score_for_eval:
            s, _, _  = self._score(text_feat, cls, video_feat, text_mask, video_mask)
            retrieve_logits += self.config.loss2_weight * s

        if self.training:
            s, spatial_out, noun_out = self._score(text_feat, cls, video_feat, text_mask, video_mask)

            # retrieve_logits += self.config.loss2_weight * s

            # TODO: Inter-Diversity for spatial+Noun query
            Inter_Diversity_s = torch.einsum('abc,adc->abd', [spatial_out, spatial_out])
            Inter_Diversity_n = torch.einsum('abc,adc->abd', [noun_out, noun_out])

            Inter_Diversity_s = Inter_Diversity_s.mean(0)
            Inter_Diversity_n = Inter_Diversity_n.mean(0)
            
            Inter_Diversity_loss1 = margin_ranking_loss(Inter_Diversity_s, margin=0.1, direction='both')
            Inter_Diversity_loss = Inter_Diversity_loss1 * self.config.alpha
            
            # TODO: Inter-Consistency for spatial+Noun query
            # spatial_ = F.log_softmax(spatial_out.mean(0) / video_feat.shape[1], dim=1)
            # noun_ = F.softmax(noun_out.mean(0) / video_feat.shape[1], dim=1)
            # Intra_Consistency_loss = self.KL(spatial_, noun_)
            
            sim_metric = torch.einsum('abc,adc->abd', [spatial_out, noun_out])
            sim_metric = sim_metric.sum(0) / spatial_out.shape[1]
            r = torch.diag(sim_metric).add(-0.75).pow(2).sum()
            
            Intra_Consistency_loss = self.mse(spatial_out.mean(0), noun_out.mean(0)) + r * self.config.beta

        else:   
            Inter_Diversity_loss = 0
            Intra_Consistency_loss = 0

        return retrieve_logits, retrieve_logits.T, Inter_Diversity_loss, Intra_Consistency_loss

    def _score(self, text_feat, cls, video_feat, text_mask, video_mask):
        if self.config.query_share:
            spatial_q = self.spatial_queries.expand(video_feat.size(0),-1,-1)
            noun_q = self.spatial_queries.expand(text_feat.size(0),-1,-1)
        else:
            spatial_q = self.spatial_queries.expand(video_feat.size(0),-1,-1)
            noun_q = self.noun_queries.expand(text_feat.size(0),-1,-1)
        
        tgt = spatial_q.permute(1, 0, 2) # NLD -> LND
        memory = noun_q.permute(1, 0, 2) # NLD -> LND
        
        if self.config.cross_att_share:
            # spatial cross-attention
            spatial_out = self.transformer_decoder_noun(tgt, video_feat.permute(1, 0, 2)).permute(1, 0, 2) # spatial as query
            # noun cross-attention
            noun_out = self.transformer_decoder_noun(memory, text_feat.permute(1, 0, 2)).permute(1, 0, 2) # noun as query
        else:
            # spatial cross-attention
            spatial_out = self.transformer_decoder_spatial(tgt, video_feat.permute(1, 0, 2)).permute(1, 0, 2) # spatial as query
            # noun cross-attention
            noun_out = self.transformer_decoder_noun(memory, text_feat.permute(1, 0, 2)).permute(1, 0, 2) # noun as query
        
        # normalization
        spatial_out = spatial_out / spatial_out.norm(dim=-1, keepdim=True)  # batch x num_query x dim
        noun_out = noun_out / noun_out.norm(dim=-1, keepdim=True) # batch x num_query x dim

        s = torch.matmul(noun_out.permute(1, 0 ,2), spatial_out.permute(1, 2, 0)) # num_query x batch x dim, num_query x dim x batch
        s = s.sum(0) / self.num_queries
        
        return s, spatial_out, noun_out

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, cls = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat, cls

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat


    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()