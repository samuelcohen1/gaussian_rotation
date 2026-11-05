import json
from pathlib import Path

nb = json.loads(Path("ddim_affine_sign_registration_n5.ipynb").read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    lines = src.splitlines() or ["empty"]
    print(f"=== cell {i} ({c['cell_type']}) lines={len(lines)} ===")
    print(lines[0][:120])
    if c["cell_type"] == "code":
        # print key def lines
        for line in lines:
            if line.startswith("def ") or line.startswith("class ") or line.startswith("# "):
                print(" ", line[:100])
