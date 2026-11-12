"""Rebuild the DFN3_ll PyTorch model from upstream's stateless ONNX and check parity.

Usage: python rebuild_ll_from_onnx.py <ll_dir>
ll_dir holds config.ini and onnx/{enc,erb_dec,df_dec}.onnx.
Writes <ll_dir>/checkpoints/model_120.ckpt.best for run_export.py --model-dir.
"""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import numpy_helper as nh
from df.enhance import init_df

D = sys.argv[1]
model, df_state, _, _ = init_df(D, epoch="none", log_level="ERROR", config_allow_defaults=True)
model.eval()
sd = model.state_dict()
new = {}


def consts(m):
    c = {i.name: nh.to_array(i) for i in m.graph.initializer}
    for nd in m.graph.node:
        if nd.op_type == "Constant":
            c[nd.output[0]] = nh.to_array(nd.attribute[0].t)
    return c


def put(k, a):
    a = torch.from_numpy(np.ascontiguousarray(a)).to(sd[k].dtype).reshape(sd[k].shape)
    assert k not in new, k
    new[k] = a


for part in ["enc", "erb_dec", "df_dec"]:
    m = onnx.load(os.path.join(D, "onnx", part + ".onnx"))
    c = consts(m)
    nodes = list(m.graph.node)
    blocks = {}
    for nd in nodes:
        if nd.op_type in ("Conv", "ConvTranspose"):
            blk = nd.name.strip("/").split("/")[0]
            blocks.setdefault(blk, []).append(nd)
    for blk, cn in blocks.items():
        pre = f"{part}.{blk}."
        convw = [k for k in sd if k.startswith(pre) and k.endswith(".weight") and sd[k].dim() == 4]
        bn = [k[: -len(".running_var")] for k in sd if k.startswith(pre) and k.endswith(".running_var")]
        assert len(convw) == len(cn) and len(bn) == 1, (blk, convw, [n.name for n in cn], bn)
        for k, nd in zip(convw, cn):
            put(k, c[nd.input[1]])
        last = cn[-1]
        assert len(last.input) == 3, blk
        b = bn[0]
        eps = dict(model.named_modules())[b].eps
        n = sd[b + ".weight"].numel()
        put(b + ".weight", np.ones(n)); put(b + ".bias", c[last.input[2]])
        put(b + ".running_mean", np.zeros(n)); put(b + ".running_var", np.full(n, 1.0 - eps))
        put(b + ".num_batches_tracked", np.zeros(()))
    for nd in nodes:
        if nd.op_type == "Einsum" and nd.input[1] in c:
            put(f"{part}.{nd.input[1]}", c[nd.input[1]])
    grus = [nd for nd in nodes if nd.op_type == "GRU"]
    gname = {"enc": "emb_gru", "erb_dec": "emb_gru", "df_dec": "df_gru"}[part]
    for l, nd in enumerate(grus):
        W, R, B = c[nd.input[1]][0], c[nd.input[2]][0], c[nd.input[3]][0]
        H = W.shape[0] // 3
        zrh = lambda a: np.concatenate([a[H:2 * H], a[:H], a[2 * H:]])  # ONNX z,r,h -> torch r,z,n
        p = f"{part}.{gname}.gru."
        put(p + f"weight_ih_l{l}", zrh(W)); put(p + f"weight_hh_l{l}", zrh(R))
        put(p + f"bias_ih_l{l}", zrh(B[:3 * H])); put(p + f"bias_hh_l{l}", zrh(B[3 * H:]))
    for nd in nodes:
        if nd.op_type == "MatMul" and nd.input[1] in c:
            fc = {"enc": "lsnr_fc", "df_dec": "df_fc_a"}[part]
            put(f"{part}.{fc}.0.weight", c[nd.input[1]].T)
            add = [a for a in nodes if a.op_type == "Add" and nd.output[0] in a.input][0]
            put(f"{part}.{fc}.0.bias", c[[i for i in add.input if i != nd.output[0]][0]])

missing = [k for k in sd if k not in new and not k.endswith(("erb_fb", "erb_inv_fb"))]
assert not missing, missing
for k in ("erb_fb", "mask.erb_inv_fb"):
    new[k] = sd[k]
model.load_state_dict(new)
model.eval()

# Parity: full-sequence torch vs upstream ONNX on random features.
torch.manual_seed(0)
S = 200
feat_erb = torch.randn(1, 1, S, 32) * 0.3 - 0.5
feat_spec = torch.randn(1, 2, S, 96) * 0.1
so = lambda p: ort.InferenceSession(os.path.join(D, "onnx", p + ".onnx"), providers=["CPUExecutionProvider"])
with torch.no_grad():
    e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
    m = model.erb_dec(emb, e3, e2, e1, e0)
    coefs = model.df_dec(emb, c0)
o_enc = so("enc").run(None, {"feat_erb": feat_erb.numpy(), "feat_spec": feat_spec.numpy()})
o_erb = so("erb_dec").run(None, {"emb": o_enc[4], "e3": o_enc[3], "e2": o_enc[2], "e1": o_enc[1], "e0": o_enc[0]})
o_df = so("df_dec").run(None, {"emb": o_enc[4], "c0": o_enc[5]})
coefs_t = coefs[0] if isinstance(coefs, (tuple, list)) else coefs
for name, a, b in [("emb", emb, o_enc[4]), ("c0", c0, o_enc[5]), ("lsnr", lsnr, o_enc[6]),
                   ("erb mask", m, o_erb[0]), ("df coefs", coefs_t, o_df[0])]:
    a = a.numpy(); print(f"{name:9s} max|diff| {np.abs(a - b).max():.2e}  (max|ref| {np.abs(b).max():.2e})")

os.makedirs(os.path.join(D, "checkpoints"), exist_ok=True)
torch.save(new, os.path.join(D, "checkpoints", "model_120.ckpt.best"))
print("saved")

# Reload through init_df to prove the checkpoint round-trips.
m2, _, _, ep = init_df(D, log_level="ERROR", config_allow_defaults=True)
sd2 = m2.state_dict()
print("reload epoch", ep, "max weight diff", max(float((sd2[k].float() - new[k].float()).abs().max()) for k in new))
