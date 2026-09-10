# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 6 -- (B2) channelwise summation / block-diagonality
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/b_bn_crosschannel.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 13579
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/b_bn_crosschannel.py
# ---------------------------------------------------------------------------
"""(B2) supplement: train-mode BN is channel-block-diagonal, so 'for multiple
channels sum these channelwise' is exact.  Seed: 13579"""
import torch
from torch.func import jvp
torch.manual_seed(13579); torch.set_default_dtype(torch.float64)
worst = 0.0
for (Nb, C, H, W, eps) in [(4, 5, 5, 5, 1e-5), (3, 3, 4, 6, 0.0)]:
    bn = torch.nn.BatchNorm2d(C, eps=eps, affine=True, track_running_stats=False)
    with torch.no_grad():
        bn.weight.normal_(); bn.bias.normal_()
    bn.train()
    x = torch.randn(Nb, C, H, W) * 0.8 + 0.3
    for c in range(C):
        dx = torch.zeros_like(x); dx[:, c] = torch.randn(Nb, H, W)
        _, ot = jvp(lambda z: bn(z), (x,), (dx,))
        off = ot.clone(); off[:, c] = 0.0
        worst = max(worst, float(off.abs().max()))
print(f"max |d(output)| in channels OTHER than the perturbed one: {worst:.3e}")
assert worst < 1e-14
print("=> D_BN is block diagonal across channels; channelwise summing in (B2) is exact.")
