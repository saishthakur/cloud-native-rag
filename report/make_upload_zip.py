"""Build a self-contained LaTeX upload zip (Prism, Overleaf, or any TeX Live service).

    python report/make_upload_zip.py [output.zip]

Inlines every \\input{tables/...} into main.tex so compilation does not depend on the folder
layout surviving the upload, and lets graphics resolve from figures/ or the project root.
"""

import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def inline_inputs(tex: str) -> str:
    def repl(m):
        path = os.path.join(HERE, m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        with open(path, encoding="utf-8") as f:
            return f"% --- {m.group(1)} ---\n" + f.read().rstrip("\n")
    return re.sub(r"\\input\{([^}]+)\}", repl, tex)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "CloudRAG_paper_upload.zip")
    with open(os.path.join(HERE, "main.tex"), encoding="utf-8") as f:
        tex = inline_inputs(f.read())
    tex = tex.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{figures/}{./}}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        z.write(os.path.join(HERE, "references.bib"), "references.bib")
        for name in sorted(os.listdir(os.path.join(HERE, "figures"))):
            z.write(os.path.join(HERE, "figures", name), f"figures/{name}")
    standalone = os.path.splitext(out)[0] + "_main.tex"  # the same single file, for pasting into an editor
    with open(standalone, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print("wrote", os.path.abspath(out))
    print("wrote", os.path.abspath(standalone))
    for info in zipfile.ZipFile(out).infolist():
        print(f"{info.file_size:>9}  {info.filename}")


if __name__ == "__main__":
    main()
