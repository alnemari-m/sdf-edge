"""
SDF-Edge: Model definitions for the five DNN architectures.
Each model is defined as an SDFGraph with actors, edges, and timing annotations.
Timing values estimated from CMSIS-NN int8 MAC throughput on Cortex-M7 @ 480MHz.
"""

from sdf_core import SDFGraph, SDFActor, SDFEdge


def build_mcunet():
    """MCUNet-5FPS: NAS-optimized for Cortex-M7, 30 actors, 32 edges, all single-rate."""
    g = SDFGraph("MCUNet-5FPS")

    # Actors: conv blocks, depthwise-pointwise pairs, FC
    # Width mult 0.5, input 64x64x3, int8
    layers = [
        ("conv_stem", 520, "conv"),       # 64x64x3 -> 32x32x8
        ("dw_1", 180, "dw"),              # depthwise 3x3
        ("pw_1", 340, "conv"),            # pointwise 1x1
        ("dw_2", 160, "dw"),
        ("pw_2", 320, "conv"),
        ("res_add_1", 40, "conv"),        # residual
        ("dw_3", 150, "dw"),
        ("pw_3", 310, "conv"),
        ("dw_4", 140, "dw"),
        ("pw_4", 300, "conv"),
        ("res_add_2", 38, "conv"),
        ("dw_5", 135, "dw"),
        ("pw_5", 290, "conv"),
        ("dw_6", 130, "dw"),
        ("pw_6", 280, "conv"),
        ("res_add_3", 36, "conv"),
        ("dw_7", 120, "dw"),
        ("pw_7", 260, "conv"),
        ("dw_8", 110, "dw"),
        ("pw_8", 250, "conv"),
        ("res_add_4", 34, "conv"),
        ("dw_9", 100, "dw"),
        ("pw_9", 240, "conv"),
        ("dw_10", 95, "dw"),
        ("pw_10", 230, "conv"),
        ("res_add_5", 32, "conv"),
        ("dw_11", 90, "dw"),
        ("pw_11", 220, "conv"),
        ("gap", 30, "conv"),              # global average pool
        ("fc", 80, "fc"),                 # fully connected
    ]

    for name, t, lt in layers:
        g.add_actor(SDFActor(name, t, layer_type=lt))

    # Sequential edges (all single-rate p=c=1)
    for i in range(len(layers) - 1):
        src = layers[i][0]
        dst = layers[i + 1][0]
        # Token sizes vary by layer (int8 activations)
        if i < 5:
            tok_size = 32 * 32 * 8  # 8 KB
        elif i < 15:
            tok_size = 16 * 16 * 16  # 4 KB
        else:
            tok_size = 8 * 8 * 32  # 2 KB
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok_size))

    # Skip connection edges (residuals)
    skip_pairs = [
        ("pw_1", "res_add_1", 16 * 16 * 16),
        ("pw_3", "res_add_2", 16 * 16 * 16),
        ("pw_5", "res_add_3", 8 * 8 * 32),
        ("pw_7", "res_add_4", 8 * 8 * 32),
        ("pw_9", "res_add_5", 8 * 8 * 32),
    ]
    for src, dst, tok_size in skip_pairs:
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok_size))

    return g


def build_mobilenetv2():
    """MobileNetV2: Inverted residuals, 28 actors, 30 edges, single-rate."""
    g = SDFGraph("MobileNetV2")

    layers = [
        ("conv_stem", 680, "conv"),
        ("expand_1", 420, "conv"),
        ("dw_1", 240, "dw"),
        ("project_1", 380, "conv"),
        ("expand_2", 400, "conv"),
        ("dw_2", 220, "dw"),
        ("project_2", 360, "conv"),
        ("res_add_1", 45, "conv"),
        ("expand_3", 380, "conv"),
        ("dw_3", 200, "dw"),
        ("project_3", 340, "conv"),
        ("expand_4", 360, "conv"),
        ("dw_4", 180, "dw"),
        ("project_4", 320, "conv"),
        ("res_add_2", 42, "conv"),
        ("expand_5", 340, "conv"),
        ("dw_5", 170, "dw"),
        ("project_5", 300, "conv"),
        ("expand_6", 320, "conv"),
        ("dw_6", 160, "dw"),
        ("project_6", 280, "conv"),
        ("res_add_3", 40, "conv"),
        ("expand_7", 300, "conv"),
        ("dw_7", 150, "dw"),
        ("project_7", 260, "conv"),
        ("conv_head", 200, "conv"),
        ("gap", 35, "conv"),
        ("fc", 90, "fc"),
    ]

    for name, t, lt in layers:
        g.add_actor(SDFActor(name, t, layer_type=lt))

    for i in range(len(layers) - 1):
        src = layers[i][0]
        dst = layers[i + 1][0]
        if i < 3:
            tok_size = 48 * 48 * 16  # 36 KB
        elif i < 10:
            tok_size = 24 * 24 * 24  # 13.5 KB
        elif i < 20:
            tok_size = 12 * 12 * 32  # 4.5 KB
        else:
            tok_size = 6 * 6 * 64  # 2.25 KB
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok_size))

    # Skip connections
    skip_pairs = [
        ("project_1", "res_add_1", 24 * 24 * 24),
        ("project_3", "res_add_2", 12 * 12 * 32),
        ("project_5", "res_add_3", 6 * 6 * 64),
    ]
    for src, dst, tok_size in skip_pairs:
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok_size))

    return g


def build_mobilenetv2_multirate():
    """
    MobileNetV2-MR: Multi-rate variant with tiled execution.
    Stride-2 edges: prod=1, cons=2. Global pool: prod=1, cons=2.
    26 actors, 25 edges, q_max=16.
    """
    g = SDFGraph("MobileNetV2-MultiRate")

    layers = [
        ("input_tile", 50, "conv"),
        ("expand_s2a", 420, "conv"),
        ("stride2a", 380, "conv"),        # stride-2 conv
        ("dw_s2a", 240, "dw"),
        ("pw_s2a", 360, "conv"),
        ("expand_s2b", 400, "conv"),
        ("stride2b", 340, "conv"),        # stride-2 conv
        ("dw_s2b", 200, "dw"),
        ("pw_s2b", 320, "conv"),
        ("expand_1", 380, "conv"),
        ("dw_1", 180, "dw"),
        ("pw_1", 300, "conv"),
        ("res_add_1", 40, "conv"),
        ("expand_2", 360, "conv"),
        ("dw_2", 170, "dw"),
        ("pw_2", 280, "conv"),
        ("res_add_2", 38, "conv"),
        ("expand_s2c", 340, "conv"),
        ("stride2c", 300, "conv"),        # stride-2 conv
        ("dw_s2c", 160, "dw"),
        ("pw_s2c", 260, "conv"),
        ("expand_3", 320, "conv"),
        ("dw_3", 150, "dw"),
        ("pw_3", 240, "conv"),
        ("gap", 30, "conv"),              # global pool: cons=2
        ("fc", 80, "fc"),
    ]

    for name, t, lt in layers:
        g.add_actor(SDFActor(name, t, layer_type=lt))

    # Edges: most are single-rate, stride-2 edges are multi-rate
    tile_size_base = 4 * 4 * 32  # 512 bytes per tile at narrowest

    edges_def = [
        # (src, dst, prod, cons, token_size)
        ("input_tile", "expand_s2a", 1, 1, 8 * 8 * 8),     # 512B
        ("expand_s2a", "stride2a", 1, 1, 8 * 8 * 16),      # 1KB
        ("stride2a", "dw_s2a", 1, 2, 4 * 4 * 16),          # MULTI-RATE stride-2
        ("dw_s2a", "pw_s2a", 1, 1, 4 * 4 * 16),
        ("pw_s2a", "expand_s2b", 1, 1, 4 * 4 * 24),
        ("expand_s2b", "stride2b", 1, 1, 4 * 4 * 48),
        ("stride2b", "dw_s2b", 1, 2, 2 * 2 * 48),          # MULTI-RATE stride-2
        ("dw_s2b", "pw_s2b", 1, 1, 2 * 2 * 48),
        ("pw_s2b", "expand_1", 1, 1, 2 * 2 * 32),
        ("expand_1", "dw_1", 1, 1, 2 * 2 * 64),
        ("dw_1", "pw_1", 1, 1, 2 * 2 * 64),
        ("pw_1", "res_add_1", 1, 1, 2 * 2 * 32),
        ("res_add_1", "expand_2", 1, 1, 2 * 2 * 32),
        ("expand_2", "dw_2", 1, 1, 2 * 2 * 64),
        ("dw_2", "pw_2", 1, 1, 2 * 2 * 64),
        ("pw_2", "res_add_2", 1, 1, 2 * 2 * 32),
        ("res_add_2", "expand_s2c", 1, 1, 2 * 2 * 32),
        ("expand_s2c", "stride2c", 1, 1, 2 * 2 * 64),
        ("stride2c", "dw_s2c", 1, 2, 1 * 1 * 64),          # MULTI-RATE stride-2
        ("dw_s2c", "pw_s2c", 1, 1, 1 * 1 * 64),
        ("pw_s2c", "expand_3", 1, 1, 1 * 1 * 48),
        ("expand_3", "dw_3", 1, 1, 1 * 1 * 96),
        ("dw_3", "pw_3", 1, 1, 1 * 1 * 96),
        ("pw_3", "gap", 1, 1, 1 * 1 * 48),
        ("gap", "fc", 1, 2, 48),                            # MULTI-RATE global pool
    ]

    for src, dst, p, c, tok in edges_def:
        g.add_edge(SDFEdge(src, dst, prod=p, cons=c, token_size_bytes=tok))

    return g


def build_efficientnet_b0():
    """EfficientNet-B0: 76 actors, 84 edges, single-rate (width mult 0.5)."""
    g = SDFGraph("EfficientNet-B0")

    # Simplified: chain of MBConv blocks with squeeze-excite
    block_configs = [
        # (name_prefix, num_layers, exec_time_base, tok_size)
        ("stem", 1, 600, 32 * 32 * 16),
        ("mb1", 4, 350, 32 * 32 * 8),
        ("mb2", 4, 320, 16 * 16 * 12),
        ("mb3", 4, 300, 16 * 16 * 16),
        ("mb4", 6, 280, 8 * 8 * 24),
        ("mb5", 6, 260, 8 * 8 * 40),
        ("mb6", 8, 240, 4 * 4 * 48),
        ("mb7", 4, 220, 4 * 4 * 80),
        ("se", 8, 80, 1 * 1 * 80),       # squeeze-excite blocks
        ("head", 1, 180, 4 * 4 * 80),
    ]

    actor_idx = 0
    all_actors = []
    for prefix, count, t_base, tok in block_configs:
        for i in range(count):
            name = f"{prefix}_{i}"
            # Vary timing slightly per layer
            t = t_base + (i * 15)
            lt = "fc" if "se" in prefix else "conv"
            g.add_actor(SDFActor(name, t, layer_type=lt))
            all_actors.append((name, tok))
            actor_idx += 1

    # Add gap and fc
    g.add_actor(SDFActor("gap", 25, layer_type="conv"))
    all_actors.append(("gap", 64))
    g.add_actor(SDFActor("fc", 85, layer_type="fc"))
    all_actors.append(("fc", 64))

    # Sequential edges
    for i in range(len(all_actors) - 1):
        src, tok = all_actors[i]
        dst, _ = all_actors[i + 1]
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok))

    # Squeeze-excite skip edges (SE reads from MBConv output, writes back)
    se_connections = [
        ("mb4_2", "se_0"), ("mb4_4", "se_1"),
        ("mb5_2", "se_2"), ("mb5_4", "se_3"),
        ("mb6_2", "se_4"), ("mb6_4", "se_5"),
        ("mb6_6", "se_6"), ("mb7_2", "se_7"),
    ]
    for src, dst in se_connections:
        tok_size = 256  # SE reduced channel
        g.add_edge(SDFEdge(src, dst, prod=1, cons=1, token_size_bytes=tok_size))

    return g


def build_vit_tiny():
    """ViT-Tiny: 4-layer transformer, 61 actors, 76 edges, d_model=64."""
    g = SDFGraph("ViT-Tiny")

    # Patch embedding
    g.add_actor(SDFActor("patch_embed", 400, layer_type="conv"))
    g.add_actor(SDFActor("pos_embed_add", 50, layer_type="conv"))

    # 4 transformer layers, each with: layernorm, Q, K, V, attention, proj, add, layernorm2, ffn1, ffn2, add2
    for layer in range(4):
        prefix = f"l{layer}"
        actors = [
            (f"{prefix}_ln1", 60, "conv"),
            (f"{prefix}_q_proj", 150, "fc"),
            (f"{prefix}_k_proj", 150, "fc"),
            (f"{prefix}_v_proj", 150, "fc"),
            (f"{prefix}_attn_scale", 80, "fc"),
            (f"{prefix}_softmax", 100, "fc"),
            (f"{prefix}_attn_v", 120, "fc"),
            (f"{prefix}_out_proj", 150, "fc"),
            (f"{prefix}_res_add1", 40, "conv"),
            (f"{prefix}_ln2", 60, "conv"),
            (f"{prefix}_ffn1", 300, "fc"),
            (f"{prefix}_ffn2", 300, "fc"),
            (f"{prefix}_res_add2", 40, "conv"),
        ]
        for name, t, lt in actors:
            g.add_actor(SDFActor(name, t, layer_type=lt))

    # Classification head
    g.add_actor(SDFActor("ln_final", 60, layer_type="conv"))
    g.add_actor(SDFActor("cls_token_select", 10, layer_type="conv"))
    g.add_actor(SDFActor("fc_head", 80, layer_type="fc"))

    # Token size: 64 tokens * 64 dims * 1 byte = 4096 bytes
    tok = 64 * 64  # 4KB

    # Edges: patch_embed -> pos_embed_add
    g.add_edge(SDFEdge("patch_embed", "pos_embed_add", token_size_bytes=tok))

    prev_output = "pos_embed_add"
    for layer in range(4):
        p = f"l{layer}"
        # Main path
        g.add_edge(SDFEdge(prev_output, f"{p}_ln1", token_size_bytes=tok))
        # Fork: ln1 -> Q, K, V
        g.add_edge(SDFEdge(f"{p}_ln1", f"{p}_q_proj", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_ln1", f"{p}_k_proj", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_ln1", f"{p}_v_proj", token_size_bytes=tok))
        # Q,K -> attn_scale
        g.add_edge(SDFEdge(f"{p}_q_proj", f"{p}_attn_scale", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_k_proj", f"{p}_attn_scale", token_size_bytes=tok))
        # attn_scale -> softmax -> attn_v
        g.add_edge(SDFEdge(f"{p}_attn_scale", f"{p}_softmax", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_softmax", f"{p}_attn_v", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_v_proj", f"{p}_attn_v", token_size_bytes=tok))
        # attn_v -> out_proj -> res_add1
        g.add_edge(SDFEdge(f"{p}_attn_v", f"{p}_out_proj", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_out_proj", f"{p}_res_add1", token_size_bytes=tok))
        # Skip connection
        g.add_edge(SDFEdge(prev_output, f"{p}_res_add1", token_size_bytes=tok))
        # FFN path
        g.add_edge(SDFEdge(f"{p}_res_add1", f"{p}_ln2", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_ln2", f"{p}_ffn1", token_size_bytes=tok))
        g.add_edge(SDFEdge(f"{p}_ffn1", f"{p}_ffn2", token_size_bytes=tok * 4))  # 4x expansion
        g.add_edge(SDFEdge(f"{p}_ffn2", f"{p}_res_add2", token_size_bytes=tok))
        # Skip connection for FFN
        g.add_edge(SDFEdge(f"{p}_res_add1", f"{p}_res_add2", token_size_bytes=tok))

        prev_output = f"{p}_res_add2"

    # Classification head
    g.add_edge(SDFEdge(prev_output, "ln_final", token_size_bytes=tok))
    g.add_edge(SDFEdge("ln_final", "cls_token_select", token_size_bytes=tok))
    g.add_edge(SDFEdge("cls_token_select", "fc_head", token_size_bytes=64))

    return g


ALL_MODELS = {
    "mcunet": build_mcunet,
    "mobilenetv2": build_mobilenetv2,
    "mobilenetv2_mr": build_mobilenetv2_multirate,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_tiny": build_vit_tiny,
}
