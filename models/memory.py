import torch
from torch import nn
from torch.nn import functional as F



class Memory(nn.Module):
    def __init__(self, radius=16.0, n_slot=96):
        super().__init__()
        self.key = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        nn.init.normal_(self.value, 0, 0.5)
        self.q_embd = nn.Linear(512, 512)
        self.v_embd = nn.Linear(512, 512)
        self.fusion = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.5)
        self.radius = radius
        self.softmax = nn.Softmax(1)

    def forward(self, query, value=None, inference=False):
        query = query.squeeze(2).squeeze(2)
        B, C = query.size()
        add_loss, recon_loss, key_add, value_add, tr_fusion = None, None, None, None, None

        embd_query = self.q_embd(query)
        query_norm = F.normalize(self.key, dim=1)
        key_sim = F.linear(F.normalize(embd_query, dim=1), query_norm)
        key_add = self.softmax(self.radius * key_sim)
        key_add = key_add.cuda()
        vir_lip = torch.matmul(key_add, self.value)

        te_fusion = torch.cat([query, vir_lip], 1)
        te_fusion = self.dropout(te_fusion)
        te_fusion = self.fusion(te_fusion)
        te_fusion = te_fusion.unsqueeze(2).unsqueeze(2)

        # Update
        if not inference:
            value = value.squeeze(2).squeeze(2)
            embd_value = self.v_embd(value)
            value_norm = F.normalize(self.value, dim=1)
            value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm)
            value_add = self.softmax(self.radius * value_sim)
            lip = torch.matmul(value_add, self.value)

            recon_loss = F.mse_loss(lip, embd_value.detach())
            recon_loss = recon_loss.unsqueeze(0)

            tr_fusion = torch.cat([query, lip], 1)
            tr_fusion = self.dropout(tr_fusion)
            tr_fusion = self.fusion(tr_fusion)
            tr_fusion = tr_fusion.unsqueeze(2).unsqueeze(2)

            add_loss = F.kl_div(torch.log(key_add + 1e-13), value_add.detach(), reduction='batchmean')
            add_loss = add_loss.unsqueeze(0)

            return te_fusion, tr_fusion, recon_loss, add_loss, key_add.view(int(B / 5), 5, -1), value_add.view(int(B / 5), 5, -1)
        else:
            return te_fusion, tr_fusion, recon_loss, add_loss, key_add.view(1, -1, 96), value_add