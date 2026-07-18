import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.linear1: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['moe_hidden_dim'], bias=False)
        self.linear2: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['moe_hidden_dim'], bias=False)
        self.linear3: nn.Linear = nn.Linear(in_features=cfg['moe_hidden_dim'], out_features=cfg['emb_dim'], bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = nn.functional.silu(self.linear1(x)) * self.linear2(x)
        return self.linear3(hidden)


class MixtureOfExperts(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        if cfg['n_experts'] <= 0:
            raise ValueError('n_experts must be greater than 0')
        self.n_experts = cfg['n_experts']
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                Expert(cfg=cfg) for _ in range(self.n_experts)
            ]
        )

        self.router: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=self.n_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence_length, emb_dim = x.shape

        # batch, seq_len, num_experts
        router_logits = self.router(x)
        routing_probs = torch.softmax(router_logits, dim=-1)

        # batch, seq_len
        selected_expert = torch.argmax(routing_probs, dim=-1)

        # batch * seq_len, emb_dim
        flat_x = x.reshape(batch * sequence_length, emb_dim)

        # batch * seq_len
        flat_selected_expert = selected_expert.reshape(-1)

        # batch * seq_len, num_experts
        flat_routing_probs = routing_probs.reshape(batch * sequence_length, self.n_experts)

        # batch * seq_len, emb_dim
        flat_output = torch.empty_like(flat_x)

        active_experts = torch.unique(flat_selected_expert)

        for expert_index_tensor in active_experts:
            expert_index = int(expert_index_tensor.item())
            token_positions = torch.where(flat_selected_expert == expert_index)[0]

            # num_selected_tokens, emb_dim
            expert_input = flat_x[token_positions]

            expert_output = self.experts[expert_index](expert_input)

            # num_selected_tokens, 1
            selected_prob = flat_routing_probs[token_positions, expert_index].unsqueeze(-1)
            expert_output = expert_output * selected_prob
            flat_output[token_positions] = expert_output
        
        output = flat_output.reshape(batch, sequence_length, emb_dim)
        return output



if __name__ == '__main__':
    cfg = {'emb_dim': 768, 'moe_hidden_dim': 768//3, 'n_experts': 4}

    experts = nn.ModuleList(
        [
            Expert(cfg=cfg) for _ in range(cfg['n_experts'])
        ]
    )

    router = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['n_experts'], bias=False)

