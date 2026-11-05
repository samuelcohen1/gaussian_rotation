import json
from pathlib import Path

nb = json.loads(Path("ddim_affine_sign_registration_n5.ipynb").read_text(encoding="utf-8"))
out = Path("_tmp_ddim_n5_cells")
out.mkdir(exist_ok=True)
for i in [0, 2, 4, 5, 7, 9, 11, 13, 15]:
    src = "".join(nb["cells"][i]["source"])
    (out / f"cell_{i:02d}.py").write_text(src, encoding="utf-8")
    print(f"wrote cell {i}, {len(src)} chars")
