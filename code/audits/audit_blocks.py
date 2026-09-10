# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : session scratchpad 887fb000-.../scratchpad/audit_blocks.py
# author  : Claude (session audit of the paper propositions)
# date    : 2026-08-26
# seed    : torch.manual_seed(0), then (1)
# command : venv/bin/python3 code/audits/audit_blocks.py
# ---------------------------------------------------------------------------
"""Hostile-referee numerical audit of cor:attribution and prop:ntk_classprior.

Checks, in the paper's exact normalization (03_setup.tex):
  L = (1/N) sum_n CE(yhat_n, y_n),  r = N^{-1/2}(r_1..r_N),  J = N^{-1/2} dYhat/dp,
  grad_p L = J^T r (claimed EXACT).

(1) grad identity grad_p L == J^T r  (sanity for both blocks)
(2) prop:ntk_classprior head-weight block of J^T w == c hbar^T for a SHARED-W
    1x1-conv head fed by a pixel-MIXING 3x3 conv (BlockViTv2-style), with and
    without bias; and ||J^T w||^2 >= ||hbar||^2.  N_cls = 3 and N_cls = 2.
(3) cor:attribution degenerate case: an exact smooth critical point of the
    joint CE loss with r != 0 (tanh head, W1=W2=b=0, balanced classes) =>
    stationary trajectory on [0,T], L(0)-L(T)=0, A_theta = 0/0 undefined,
    while Assumption ass:residual holds on [0,T] with C = max(1, ||r||(1+mu*T)).
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

Ndata, Hh, Ww = 2, 3, 4
N = Ndata * Hh * Ww          # 24 pixels
S, K, Mh = 6, 5, 7

X = torch.randn(Ndata, S, Hh, Ww)


def run_prop_check(Ncls, bias):
    torch.manual_seed(1)
    Wf = torch.randn(S, K, requires_grad=True)                    # spectral theta
    conv1 = torch.nn.Conv2d(K, Mh, 3, padding=1)                  # mixes pixels
    head = torch.nn.Conv2d(Mh, Ncls, 1, bias=bias)                # shared-W head
    params = [Wf] + list(conv1.parameters()) + list(head.parameters())

    Z = torch.einsum('sk,bshw->bkhw', Wf, X)
    h = torch.relu(conv1(Z))                                      # penultimate
    y = head(h)                                                   # per-pixel logits

    # unit class contrast c ⊥ 1
    c = torch.randn(Ncls); c = c - c.mean(); c = c / c.norm()

    # g := <ytilde, w> = N^{-1} sum_n c^T yhat_n   (w = N^{-1/2}(c,...,c), unit)
    g = torch.einsum('i,bihw->', c, y) / N
    grads = torch.autograd.grad(g, params, retain_graph=True)     # = J^T w

    hbar = h.mean(dim=(0, 2, 3)).detach()                         # (Mh,)
    head_w_grad = grads[-1 if not bias else -2].detach().squeeze(-1).squeeze(-1)
    pred_block = torch.outer(c, hbar)                             # c hbar^T
    block_err = (head_w_grad - pred_block).abs().max().item()

    JTw_sq = sum(gr.pow(2).sum() for gr in grads).item()          # w^T Theta_hat w
    hbar_sq = hbar.pow(2).sum().item()

    # grad identity  grad_p L == J^T r  (exactness claim of eq. (chain))
    labels = torch.arange(N).remainder(Ncls).reshape(Ndata, Hh, Ww)
    L = torch.nn.functional.cross_entropy(y, labels, reduction='mean')
    gL = torch.autograd.grad(L, params, retain_graph=True)
    with torch.no_grad():
        r_pix = torch.softmax(y, dim=1) - torch.nn.functional.one_hot(
            labels, Ncls).permute(0, 3, 1, 2).double()
    inner = torch.einsum('bihw,bihw->', r_pix.detach(), y) / N    # <ytilde, r>
    gJr = torch.autograd.grad(inner, params)
    grad_id_err = max((a - b).abs().max().item() for a, b in zip(gL, gJr))

    print(f"Ncls={Ncls} bias={bias}:")
    print(f"  head-block vs c hbar^T   max|diff| = {block_err:.3e}")
    print(f"  ||J^T w||^2 = {JTw_sq:.6f}  >=  ||hbar||^2 = {hbar_sq:.6f}  "
          f"-> {'OK' if JTw_sq >= hbar_sq - 1e-12 else 'VIOLATED'}")
    print(f"  grad identity grad L = J^T r  max|diff| = {grad_id_err:.3e}")


for ncls in (3, 2):
    for bias in (True, False):
        run_prop_check(ncls, bias)

# ---------------- cor:attribution degenerate case ----------------
print("\ncor:attribution stationary counterexample (tanh, zero init, balanced):")
Ncls = 3
Wf = torch.randn(S, K, requires_grad=True)          # arbitrary nonzero spectral params
W1 = torch.zeros(Mh, K, requires_grad=True)
b1 = torch.zeros(Mh, requires_grad=True)
W2 = torch.zeros(Ncls, Mh, requires_grad=True)
b2 = torch.zeros(Ncls, requires_grad=True)
params = [Wf, W1, b1, W2, b2]

Z = torch.einsum('sk,bshw->bkhw', Wf, X)
h = torch.tanh(torch.einsum('mk,bkhw->bmhw', W1, Z) + b1[None, :, None, None])
y = torch.einsum('im,bmhw->bihw', W2, h) + b2[None, :, None, None]
labels = torch.arange(N).remainder(Ncls).reshape(Ndata, Hh, Ww)  # 8 px/class: balanced
L = torch.nn.functional.cross_entropy(y, labels, reduction='mean')
gL = torch.autograd.grad(L, params)
gnorm = max(g.abs().max().item() for g in gL)
with torch.no_grad():
    r_pix = torch.softmax(y, dim=1) - torch.nn.functional.one_hot(
        labels, Ncls).permute(0, 3, 1, 2).double()
    r_norm = (r_pix.pow(2).sum() / N).sqrt().item()   # RMS per-pixel residual ||r||
print(f"  max|grad| over ALL params = {gnorm:.3e}  (exact critical point)")
print(f"  ||r|| (RMS residual)      = {r_norm:.6f}  (nonzero, constant in t)")
print(f"  => trajectory is stationary: L(0)-L(T) = 0 for every T; "
      f"A_theta(T) = 0/0 undefined.")
print(f"  ass:residual on [0,T] holds with C = max(1, ||r||(1+mu*T)) for any mu>0.")
