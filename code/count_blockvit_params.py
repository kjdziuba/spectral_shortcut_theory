import sys, torch
sys.path.insert(0, '/home/u37314kd/Projects/spectral_tokenization/side_project')
from models.blockvit_v2 import BlockViTv2

def count(m): return sum(p.numel() for p in m.parameters())

for K, label in [(64, 'blockvit_v2 (joint baseline, --reduce_dim 64)'),
                 (128, 'blockvit_v2_pca23 (fair-comparison, --reduce_dim 128)')]:
    m = BlockViTv2(in_channels=3, num_classes=4, num_spectral=314,
                   reduce_dim=K, patch_tok_size=16, hidden_dim=192,
                   num_layers=12, num_heads=12, mlp_ratio=4.0,
                   spatial_size=336, spectral_norm=True)
    Cf = count(m.spectral_reduce)
    total = count(m)
    Cg = total - Cf
    print(f"\n=== {label} ===")
    print(f"  C_f  spectral_reduce      : {Cf:>12,}")
    print(f"       - proj (3*314={3*314} -> {K}) : {count(m.spectral_reduce.proj):>12,}")
    print(f"       - wn_norm BN1d(314)   : {count(m.spectral_reduce.wn_norm):>12,}")
    print(f"       - norm BN2d({K})       : {count(m.spectral_reduce.norm):>12,}")
    print(f"  C_g  spatial (everything else): {Cg:>12,}")
    print(f"       - patch_embed         : {count(m.patch_embed):>12,}")
    print(f"       - pos_embed           : {m.pos_embed.numel():>12,}")
    print(f"       - transformer (12 lyr) : {count(m.transformer):>12,}")
    print(f"       - norm                : {count(m.norm):>12,}")
    print(f"       - upsample            : {count(m.upsample):>12,}")
    print(f"       - seg_head            : {count(m.seg_head):>12,}")
    print(f"  TOTAL                      : {total:>12,}")
    print(f"  >>> C_g / C_f = {Cg/Cf:.1f}")
