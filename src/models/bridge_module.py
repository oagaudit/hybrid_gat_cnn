"""
src/models/bridge_module.py
Bridge Module: Attention-based pooling to aggregate pair embeddings into tender-level visual embedding.

Input: For a tender, a set of pair embeddings (each 64-dim) from CNN.
Output: Single visual embedding (128-dim) representing the tender's bidding interaction pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeModule(nn.Module):
    """
    Attention-based pooling module to aggregate multiple pair embeddings into one visual embedding.
    Similar to "Set Transformer" or "Attentional Pooling" but lightweight.
    
    Args:
        pair_embed_dim (int): Dimension of input pair embeddings (default=64)
        visual_embed_dim (int): Output dimension of visual embedding (default=128)
        hidden_dim (int): Hidden dimension for attention network (default=64)
    """
    def __init__(self, pair_embed_dim=64, visual_embed_dim=128, hidden_dim=64):
        super(BridgeModule, self).__init__()
        
        # Query vector (learnable) – can be conditioned on tender context (e.g., screens)
        # For simplicity, we use a learnable parameter. Later we can extend to use screens as context.
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Project pair embeddings to key and value spaces
        self.key_proj = nn.Linear(pair_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(pair_embed_dim, visual_embed_dim)
        
        # Optional: combine with screens? We'll keep separate for now.
        
    def forward(self, pair_embeddings):
        """
        Args:
            pair_embeddings: Tensor of shape (num_pairs, pair_embed_dim) for a single tender.
                             Can also be batched: (batch_size, num_pairs, pair_embed_dim)
        Returns:
            visual_embedding: Tensor of shape (visual_embed_dim) or (batch_size, visual_embed_dim)
        """
        # If input is 2D (single tender), add batch dimension
        if pair_embeddings.dim() == 2:
            pair_embeddings = pair_embeddings.unsqueeze(0)  # (1, num_pairs, dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_pairs, _ = pair_embeddings.shape
        
        # Project keys and values
        keys = self.key_proj(pair_embeddings)   # (batch, num_pairs, hidden_dim)
        values = self.value_proj(pair_embeddings)  # (batch, num_pairs, visual_dim)
        
        # Expand query to match batch
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        
        # Compute attention scores: (batch, 1, num_pairs)
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, num_pairs)
        
        # Weighted sum of values
        visual_embed = torch.bmm(attn_weights, values)  # (batch, 1, visual_dim)
        visual_embed = visual_embed.squeeze(1)          # (batch, visual_dim)
        
        if squeeze_output:
            visual_embed = visual_embed.squeeze(0)      # (visual_dim,)
        
        return visual_embed


class ContextualBridgeModule(nn.Module):
    """
    Extended Bridge Module that also uses statistical screens as context to modulate attention.
    This follows the proposal: visual embedding + screens = full node features.
    
    Args:
        pair_embed_dim (int): 64
        screen_dim (int): 7
        visual_embed_dim (int): 128
        hidden_dim (int): 64
    """
    def __init__(self, pair_embed_dim=64, screen_dim=7, visual_embed_dim=128, hidden_dim=64):
        super(ContextualBridgeModule, self).__init__()
        
        # Project screens to a context vector
        self.screen_proj = nn.Linear(screen_dim, hidden_dim)
        
        # Query is now generated from screens (instead of fixed)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.key_proj = nn.Linear(pair_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(pair_embed_dim, visual_embed_dim)
        
    def forward(self, pair_embeddings, screens):
        """
        Args:
            pair_embeddings: (batch_size, num_pairs, pair_embed_dim)
            screens: (batch_size, screen_dim) – tender-level statistical features
        Returns:
            visual_embedding: (batch_size, visual_embed_dim)
        """
        # Project screens to context
        context = F.relu(self.screen_proj(screens))  # (batch, hidden_dim)
        
        # Generate query from context
        query = self.query_proj(context).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Project keys and values
        keys = self.key_proj(pair_embeddings)   # (batch, num_pairs, hidden_dim)
        values = self.value_proj(pair_embeddings)  # (batch, num_pairs, visual_dim)
        
        # Attention
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        visual_embed = torch.bmm(attn_weights, values).squeeze(1)
        return visual_embed


# Quick test
if __name__ == "__main__":
    # Test BridgeModule
    bridge = BridgeModule()
    dummy_pairs = torch.randn(10, 64)   # 10 pairs in one tender
    out = bridge(dummy_pairs)
    print(f"BridgeModule output shape: {out.shape}")  # Expected (128,)
    
    # Test ContextualBridgeModule
    ctx_bridge = ContextualBridgeModule()
    dummy_pairs_batch = torch.randn(4, 15, 64)   # batch=4 tenders, each with 15 pairs
    dummy_screens = torch.randn(4, 7)
    out_batch = ctx_bridge(dummy_pairs_batch, dummy_screens)
    print(f"ContextualBridgeModule output shape: {out_batch.shape}")  # Expected (4, 128)