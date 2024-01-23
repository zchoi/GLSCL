# -*- coding: utf-8 -*-
# ------------------------------------------------------
# -------------------- LOCATE Module -------------------
# ------------------------------------------------------
import torch
import torch.nn as nn
from models.attention import SoftAttention, GumbelAttention

class LOCATE(nn.Module):
    def __init__(self, opt):
        super(LOCATE, self).__init__()
        # spatial soft attention module
        self.spatial_attn = SoftAttention(opt.region_projected_size, opt.hidden_size, opt.hidden_size)

        # temporal soft attention module
        feat_size = opt.region_projected_size + opt.hidden_size * 2
        self.temp_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)

    def forward(self, frame_feats, object_feats, hidden_state):
        """
        :param frame_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: loc_feat: (batch_size, feat_size)
        """
        # spatial attention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # temporal attention
        feat = torch.cat([object_feats_att, frame_feats], dim=-1)
        loc_feat, _ = self.temp_attn(feat, hidden_state)
        return loc_feat


# ------------------------------------------------------
# -------------------- RELATE Module -------------------
# ------------------------------------------------------

class RELATE(nn.Module):
    def __init__(self, opt):
        super(RELATE, self).__init__()

        # spatial soft attention module
        region_feat_size = opt.region_projected_size
        self.spatial_attn = SoftAttention(region_feat_size, opt.hidden_size, opt.hidden_size)

        # temporal soft attention module
        feat_size = region_feat_size + opt.hidden_size * 2
        self.relation_attn = SoftAttention(2*feat_size, opt.hidden_size, opt.hidden_size)

    def forward(self, i3d_feats, object_feats, hidden_state):
        '''
        :param i3d_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: rel_feat
        '''
        # spatial atttention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # generate pair-wise feature
        feat = torch.cat([object_feats_att, i3d_feats], dim=-1)
        feat1 = feat.repeat(1, max_frames, 1)
        feat2 = feat.repeat(1, 1, max_frames).reshape(bsz, max_frames*max_frames, -1)
        pairwise_feat = torch.cat([feat1, feat2], dim=-1)

        # temporal attention
        rel_feat, _ = self.relation_attn(pairwise_feat, hidden_state)
        return rel_feat


# ------------------------------------------------------
# -------------------- FUNC Module ---------------------
# ------------------------------------------------------

class FUNC(nn.Module):
    def __init__(self, opt):
        super(FUNC, self).__init__()
        self.cell_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, cells, hidden_state):
        '''
        :param cells: previous memory states of decoder LSTM
        :param hidden_state: (batch_size, hidden_size)
        :return: func_feat
        '''
        func_feat, _ = self.cell_attn(cells, hidden_state)
        return func_feat



# ------------------------------------------------------
# ------------------- Module Selector ------------------
# ------------------------------------------------------

class ModuleSelection(nn.Module):
    def __init__(self, opt):
        super(ModuleSelection, self).__init__()
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func

        if opt.use_loc:
            loc_feat_size = opt.region_projected_size + opt.hidden_size * 2
            self.loc_fc = nn.Linear(loc_feat_size, opt.hidden_size)
            nn.init.xavier_normal_(self.loc_fc.weight)

        if opt.use_rel:
            rel_feat_size = 2 * (opt.region_projected_size + 2 * opt.hidden_size)
            self.rel_fc = nn.Linear(rel_feat_size, opt.hidden_size)
            nn.init.xavier_normal_(self.rel_fc.weight)

        if opt.use_func:
            func_size = opt.hidden_size
            self.func_fc = nn.Linear(func_size, opt.hidden_size)
            nn.init.xavier_normal_(self.func_fc.weight)

        if opt.use_loc and opt.use_rel and opt.use_func:
            if opt.attention == 'soft':
                self.module_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            elif opt.attention == 'gumbel':
                self.module_attn = GumbelAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, loc_feats, rel_feats, func_feats, hidden_state):
        '''
        soft attention: Weighted sum of three features
        gumbel attention: Choose one of three features
        '''
        loc_feats = self.loc_fc(loc_feats) if self.use_loc else None
        rel_feats = self.rel_fc(rel_feats) if self.use_rel else None
        func_feats = self.func_fc(func_feats) if self.use_func else None

        if self.use_loc and self.use_rel and self.use_func:
            feats = torch.stack([loc_feats, rel_feats, func_feats], dim=1)
            feats, module_weight = self.module_attn(feats, hidden_state)

        elif self.use_loc and not self.use_rel:
            feats = loc_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()
        elif self.use_rel and not self.use_loc:
            feats = rel_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()

        return feats, module_weight