from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
ASSETS = OUT / "assets"
OUT.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)


def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


stage32 = load_json("logs/stage3_2_repeated_seed_sensitivity.json")
stage31 = load_json("logs/stage3_1_buffered_blocks.json")
initial = load_json("logs/initial_salinas_experiment.json")


def font(size: int = 11, bold: bool = False, italic: bool = False, color: str = "222222"):
    f = FontSpec(size, bold, italic, color)
    return f


class FontSpec:
    def __init__(self, size, bold=False, italic=False, color="222222"):
        self.size = size
        self.bold = bold
        self.italic = italic
        self.color = color


def set_cell_shading(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color="D9D9D9", size="6"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def prevent_row_split(row):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trPr.append(cant)


def set_cell_text(cell, text, *, bold=False, color="222222", size=9, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def add_table(doc, headers, rows, widths=None, font_size=8.7):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header = table.rows[0]
    repeat_header(header)
    for i, h in enumerate(headers):
        if widths:
            header.cells[i].width = Inches(widths[i])
        set_cell_text(header.cells[i], h, bold=True, color="FFFFFF", size=font_size, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(header.cells[i], "1F4E5F")
        set_cell_borders(header.cells[i])
    for row_index, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for i, value in enumerate(values):
            if widths:
                row.cells[i].width = Inches(widths[i])
            align = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or all(ch not in str(value) for ch in [" ", "/"]) else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(row.cells[i], value, size=font_size, align=align)
            set_cell_shading(row.cells[i], "F3F7F8" if row_index % 2 else "FFFFFF")
            set_cell_borders(row.cells[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_hyperlink(paragraph, text, url, color="1F4E5F", underline=True):
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), color)
    rPr.append(c)
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)
    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def add_link_paragraph(doc, label, url):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(label + ": ")
    r.font.size = Pt(10.5)
    r.font.name = "Arial"
    add_hyperlink(p, url, url)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("555555")
    r.font.name = "Arial"
    return p


def add_body(doc, text, *, bold_lead=None):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead and text.startswith(bold_lead):
        a, b = bold_lead, text[len(bold_lead):]
        r = p.add_run(a)
        r.bold = True
        r.font.name = "Arial"
        p.add_run(b)
    else:
        p.add_run(text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.05
    p.add_run(text)
    return p


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.right_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.name = "Consolas"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
    r.font.size = Pt(9)
    return p


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def get_font(size, bold=False, italic=False):
    candidates = [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/Arial.ttf")]
    if bold:
        candidates = [Path("C:/Windows/Fonts/arialbd.ttf"), Path("C:/Windows/Fonts/Arialbd.ttf")] + candidates
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_convergence():
    hist = stage31["local_results"]["JSLLDA"]["history"]
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = get_font(34, bold=True)
    label = get_font(23)
    small = get_font(18)
    d.text((70, 35), "JSLLDA convergence under the buffered protocol", fill="#222222", font=title)
    left, right, top, bottom = 120, 1510, 115, 760
    d.line((left, top, left, bottom), fill="#555555", width=3)
    d.line((left, bottom, right, bottom), fill="#555555", width=3)
    n = len(hist)
    obj = [x["objective"] for x in hist]
    lo, hi = min(obj), max(obj)
    y0, y1 = 0.0, hi * 1.01
    points = []
    for i, value in enumerate(obj):
        x = left + (right-left) * i / max(1, n-1)
        y = bottom - (bottom-top) * (value-y0) / (y1-y0)
        points.append((x, y))
    d.line(points, fill="#1F4E5F", width=5)
    d.text((left+25, top+15), "Objective", fill="#1F4E5F", font=label)
    d.text((right-220, bottom+25), "Iteration", fill="#444444", font=label)
    d.text((20, top-10), "104.2", fill="#444444", font=small)
    d.text((20, bottom-10), "0", fill="#444444", font=small)
    d.text((left-10, bottom+25), "1", fill="#444444", font=small)
    d.text((right-25, bottom+25), str(n), fill="#444444", font=small)
    inset = (930, 150, 1460, 395)
    d.rectangle(inset, outline="#BBBBBB", width=2)
    rvals = [x["primal_residual"] for x in hist]
    rmax = max(rvals)
    rmin = min(rvals)
    rp = []
    for i, value in enumerate(rvals):
        x = inset[0] + (inset[2]-inset[0]) * i / max(1, n-1)
        y = inset[3] - (inset[3]-inset[1]) * (math.log10(value)-math.log10(rmin*0.8)) / (math.log10(rmax)-math.log10(rmin*0.8))
        rp.append((x, y))
    d.line(rp, fill="#C45A2A", width=4)
    d.text((inset[0]+15, inset[1]+12), "Primal residual (log scale)", fill="#C45A2A", font=small)
    path = ASSETS / "convergence.png"
    img.save(path)
    return path


def draw_results():
    summary = stage32["repeated_seed_summary"]
    names = ["JSLLDA", "Classical LDA"]
    oa = [summary["JSLLDA"]["metrics"]["overall_accuracy"]["mean"], summary["classical_LDA"]["metrics"]["overall_accuracy"]["mean"]]
    sd = [summary["JSLLDA"]["metrics"]["overall_accuracy"]["std"], summary["classical_LDA"]["metrics"]["overall_accuracy"]["std"]]
    w, h = 1400, 800
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = get_font(32, bold=True)
    label = get_font(24)
    small = get_font(18)
    d.text((60, 35), "Five-seed buffered spatial comparison", fill="#222222", font=title)
    left, right, top, bottom = 150, 1320, 120, 680
    d.line((left, top, left, bottom), fill="#555555", width=3)
    d.line((left, bottom, right, bottom), fill="#555555", width=3)
    ymin, ymax = 0.88, 0.92
    colors = ["#1F4E5F", "#C45A2A"]
    for i, (name, value, err, color) in enumerate(zip(names, oa, sd, colors)):
        x = left + 340 + i*500
        y = bottom - (bottom-top)*(value-ymin)/(ymax-ymin)
        yerr = (bottom-top)*err/(ymax-ymin)
        d.rectangle((x-115, y, x+115, bottom), fill=color)
        d.line((x, y-yerr, x, y+yerr), fill="#222222", width=4)
        d.line((x-18, y-yerr, x+18, y-yerr), fill="#222222", width=4)
        d.line((x-18, y+yerr, x+18, y+yerr), fill="#222222", width=4)
        d.text((x-115, bottom+25), name, fill="#222222", font=label)
        d.text((x-95, y-45), f"{value:.4f}", fill="#222222", font=label)
    for val in [0.88, 0.90, 0.92]:
        y = bottom - (bottom-top)*(val-ymin)/(ymax-ymin)
        d.line((left-8, y, right, y), fill="#E5E5E5", width=2)
        d.text((55, y-12), f"{val:.2f}", fill="#555555", font=small)
    d.text((right-180, top-45), "OA with SD error bars", fill="#555555", font=small)
    path = ASSETS / "results.png"
    img.save(path)
    return path


def setup_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string("222222")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size, color in [("Title", 24, "000000"), ("Heading 1", 15, "000000"), ("Heading 2", 11.5, "1F4E5F"), ("Heading 3", 10.5, "1F4E5F")]:
        st = styles[name]
        st.font.name = "Arial"
        st._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        st._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        st.paragraph_format.space_after = Pt(5)
        st.paragraph_format.keep_with_next = True
    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.text = "CISC3024 AI Assignment 3  |  JSLLDA"
    fp.style = styles["Normal"]
    fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fp.runs[0].font.size = Pt(8)
    fp.runs[0].font.color.rgb = RGBColor.from_string("666666")
    add_page_number(footer.add_paragraph())
    return doc


def add_title(doc):
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(4)
    p.add_run("CISC3024 AI Assignment 3")
    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(12)
    r = p2.add_run("Independent implementation and evaluation of JSLLDA for Salinas hyperspectral image classification")
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor.from_string("1F4E5F")
    r.italic = True
    add_body(doc, "This report documents how an AI agent searched for and selected a recent PCA/LDA-based method, clarified ambiguities in the source paper, implemented an independent CPU solver, and evaluated it against classical LDA. The main conclusion is that JSLLDA consistently outperformed the matched LDA baseline in the five-seed buffered spatial experiment, but the result is a reduced reproduction rather than an exact replication of the paper.")
    meta = add_table(doc, ["Item", "Value"], [
        ["Course", "CISC3024 Pattern Recognition"],
        ["Selected algorithm", "Joint Sparse Local Linear Discriminant Analysis (JSLLDA)"],
        ["Primary dataset", "Salinas hyperspectral scene"],
        ["Primary feature dimension", "15"],
        ["Implementation", "Independent Python implementation using NumPy, SciPy and scikit-learn"],
    ], widths=[1.55, 5.75], font_size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def section_heading(doc, text):
    p = doc.add_paragraph(text, style="Heading 1")
    return p


def subheading(doc, text):
    return doc.add_paragraph(text, style="Heading 2")


def build():
    convergence_path = draw_convergence()
    results_path = draw_results()
    doc = setup_doc()
    add_title(doc)

    section_heading(doc, "1. How AI finds the algorithm")
    add_body(doc, "The AI agent first searched for recent PCA and LDA methods published mainly between 2023 and 2026, then checked whether the dimensionality-reduction method was a core contribution rather than a minor preprocessing step. The candidate set included ATAM-PCA for image recognition, KPCA-CAM for CNN explainability, and JSLLDA for hyperspectral image classification.")
    add_body(doc, "JSLLDA was selected because it provided the strongest balance of topical fit and reproducibility. It is explicitly an LDA-based feature-reduction algorithm for a computer-vision application, its 2024 Remote Sensing paper is openly accessible, the objective and ADMM updates are described in enough detail to implement independently, and Salinas is a public benchmark that can be audited locally. ATAM-PCA had excellent image-recognition relevance but its detailed datasets and implementation evidence were not sufficiently accessible. KPCA-CAM had a clear public preprint, but full evaluation depends on CNN models and ILSVRC-scale data and is less directly a stand-alone recognition method.")
    subheading(doc, "Evidence and selection criteria")
    for t in [
        "Original publication metadata, author list, venue and DOI were checked against the publisher or primary record.",
        "PCA or LDA had to be central to the proposed method, not only a baseline.",
        "A student-scale CPU experiment had to be plausible with public data and no pretrained model requirement.",
        "Official author code was treated as unverified unless an author-linked repository could be confirmed.",
    ]:
        add_bullet(doc, t)
    add_body(doc, "The selected paper is Cao, Li, Deng, Ren, Liu and Zhu, Joint Sparse Local Linear Discriminant Analysis for Feature Dimensionality Reduction of Hyperspectral Images, Remote Sensing 16(22), 4287, 2024. No official author repository was verified; the code in this project is independent.")

    section_heading(doc, "2. Algorithm description")
    add_body(doc, "Let X in R^(b x n) contain n training spectra as columns, Y in R^(c x n) be one-hot class targets, P in R^(b x k) be the projection, W in R^(c x k) be the label-embedding factor, and E in R^(c x n) be the residual. The JSLLDA objective used in the implementation is:")
    add_code(doc, "min  Tr(P^T (Sw - mu Sb) P) + lambda1 Tr(P^T X L X^T P)\n     + lambda2 ||E||_(2,1) + lambda3 ||P||_(2,1)\nsubject to  Y = W P^T X + E,       W^T W = I")
    add_body(doc, "The four additive terms have different roles. The first is the LDA discriminant term: it reduces within-class scatter Sw and rewards between-class scatter Sb through mu. The second is a locality term: a graph Laplacian L penalizes distances between mapped neighboring spectra. The third is a robust residual penalty that permits sample-level reconstruction errors in the label embedding. The fourth encourages joint sparsity in P so entire spectral-band rows can be suppressed across all output directions.")
    subheading(doc, "Locality graph")
    add_body(doc, "The paper describes a nearest-neighbor weighted graph and L = D - A, but does not specify the neighbor count, distance metric, kernel, bandwidth, symmetrization, or whether held-out samples enter the graph. The implementation therefore uses an explicit decision: a training-only spectral graph with Euclidean distance, 10 nearest neighbors, median positive-distance RBF bandwidth, zero diagonal, and max/union symmetrization. This is an implementation assumption, not a paper fact.")
    subheading(doc, "ADMM and alternating updates")
    add_body(doc, "The equality constraint is handled with an unscaled multiplier Lambda and penalty beta. With M = Y - E + Lambda/beta, the P step solves:")
    add_code(doc, "[2(Sw - mu Sb) + 2 lambda1 X L X^T + lambda3 H + beta X X^T] P\n    = beta X M^T W,       H_ii = 1/(2 ||P[i,:]||_2)")
    add_body(doc, "The factor 2 on lambda1 follows differentiation of the stated symmetric graph objective; the printed paper update appears to omit it. The W step is an orthogonal Procrustes update: if M X^T P = U Sigma V^T, W = U V^T. The E step uses sample-wise column groups, E[:,j] = max(1 - lambda2/(beta ||R[:,j]||_2), 0) R[:,j], with R = Y - W P^T X + Lambda/beta. Finally, Lambda is updated by Lambda <- Lambda + beta(Y - W P^T X - E), beta is increased by rho, and convergence is checked.")
    add_body(doc, "The paper's norm definition writes row-wise and column-wise sums as if they were equal. The implementation resolves this inconsistency explicitly: P uses row groups for band sparsity, while E uses column groups for one residual vector per sample, matching the cited vector shrinkage reference. The paper also states W^T W = I, which is infeasible when k exceeds the number of classes; the study therefore fixes k=15 for Salinas with 16 classes.")

    section_heading(doc, "3. How AI implements the algorithm")
    add_body(doc, "The AI agent converted the clarified equations into an independent Python implementation. The data layout, matrix dimensions, update order and regularizer orientations were written down before coding. NumPy and SciPy provide array operations and stable symmetric linear solves; scikit-learn provides nearest neighbors, classical LDA, SVM and metrics. The graph is sparse so the solver never constructs an all-pixel dense similarity matrix.")
    subheading(doc, "Implementation and debugging process")
    for t in [
        "A synthetic test first checked P, W and E shapes, W.T @ W orthogonality, finite objective values and finite equality residuals.",
        "Salinas acquisition was audited with file sizes, MD5 checksums, MATLAB shapes, class counts and the corrected-product band convention.",
        "The original randomized row split was diagnosed as insufficiently separated spatially; it was replaced by contiguous 8-row blocks with one 8-row buffer block between partitions.",
        "The initial 30-iteration result was not called converged. Primal and dual residuals were added, and the unchanged tolerance was required for both.",
        "A 100/200-iteration diagnostic confirmed stabilization at iteration 90 under the old geometry. The primary buffered experiment converged at iteration 77.",
    ]:
        add_bullet(doc, t)
    subheading(doc, "Leakage controls")
    add_body(doc, "The buffered spatial geometry is fixed across all repeated seeds. Only the 100 spectra per class selected from the training region change. Graph construction, scatter matrices, JSLLDA, classical LDA and the feature scaler use training data only. Validation labels select the SVM C value; the selected SVM is then refit on train plus validation and evaluated once on the untouched test partition. Test labels are never used for model fitting, parameter selection or stopping.")
    subheading(doc, "Convergence evidence")
    doc.add_picture(str(convergence_path), width=Inches(6.55))
    add_caption(doc, "Figure 1. Objective and primal-residual history from the converged buffered spatial JSLLDA run.")
    add_body(doc, "At the final buffered run, the objective was 100.1732624, the primal residual was 9.8878e-6, the dual residual was 3.2992e-7, and the W orthogonality error was 5.22e-15. These values are recorded in the experiment log and are not paper-reported quantities.")

    section_heading(doc, "4. Experiment settings and results")
    subheading(doc, "Dataset and split")
    add_body(doc, "The Salinas scene is a 512 x 217 AVIRIS hyperspectral image. The corrected public product contains 204 retained bands after the standard water-absorption removals [108-112], [154-167] and 224 in the original 224-band scene. The ground truth contains 54,129 labeled pixels in 16 classes. Local files were downloaded from the public danaroth/salinas Hugging Face mirror and matched the MD5 values listed by the Zenodo record 10.5281/zenodo.15771735.")
    add_body(doc, "The primary split uses contiguous 8-row tiles. Tile residues 0, 2 and 4 modulo 6 form train, validation and test; residues 1, 3 and 5 are buffer tiles. The resulting labeled counts are 8,942 train, 9,518 validation and 8,540 test, with all 16 classes present and a minimum 9-row gap between partitions. The buffer excludes 27,129 labeled pixels. This is a strict row-direction separation, not a full two-dimensional region holdout.")
    add_table(doc, ["Setting", "Value"], [
        ["Feature dimension", "15 for JSLLDA and classical LDA"],
        ["Training subset", "100 spectra per class; 1,600 total"],
        ["Graph", "Training-only spectral kNN, k=10, Euclidean, median RBF, union/max"],
        ["JSLLDA base parameters", "lambda1=0.1, lambda2=0.1, lambda3=0.001, mu=1e-5"],
        ["ADMM", "beta0=0.1, rho=1.01, tol=1e-5, max_iter=200"],
        ["SVM", "RBF SVC; C in {1, 10, 100}; gamma=scale; validation selection"],
        ["Metrics", "OA, AA, Cohen kappa, confusion matrix, per-class recall and runtime"],
    ], widths=[1.75, 5.55], font_size=8.8)
    subheading(doc, "Paper results versus local results")
    add_body(doc, "The original paper reports a different protocol: random class sampling, feature dimensions varying up to 30, incompletely specified SVM settings and graph details, and MATLAB execution. Its main Salinas table reports LDA OA 84.61 +/- 0.63 and JSLLDA OA 91.15 +/- 0.26, with a 6.54 percentage-point difference. Those are author-reported values and are not reproduced here.")
    subheading(doc, "Five-seed base comparison")
    add_body(doc, "The following results use the paper's base regularizers and the fixed buffered geometry. Mean and SD are sample statistics over five training-subset seeds; the same selected spectra are used by both methods in each seed.")
    summary = stage32["repeated_seed_summary"]
    add_table(doc, ["Method", "OA mean +/- SD", "AA mean +/- SD", "Kappa mean +/- SD", "Runtime mean +/- SD (s)", "Convergence"], [
        ["JSLLDA", "0.907588 +/- 0.002338", "0.961639 +/- 0.001058", "0.896418 +/- 0.002622", "4.4616 +/- 0.3728", "5/5; 75.8 +/- 13.0 iter."],
        ["Classical LDA", "0.895948 +/- 0.004248", "0.951054 +/- 0.002305", "0.883379 +/- 0.004763", "4.1831 +/- 0.0989", "Not applicable"],
    ], widths=[1.25, 1.18, 1.18, 1.25, 1.42, 1.02], font_size=7.8)
    doc.add_picture(str(results_path), width=Inches(6.2))
    add_caption(doc, "Figure 2. Five-seed mean test OA with sample-standard-deviation error bars under the buffered spatial protocol.")
    add_body(doc, "JSLLDA's paired OA improvements over LDA were positive for every seed: +0.00796, +0.01792, +0.01557, +0.00925 and +0.00749. Kappa improvements were also positive for all five seeds. The variation is therefore real but modest; it reflects which training spectra are selected from a spatially heterogeneous training region and the stochastic SVM/training subset interaction.")
    subheading(doc, "Parameter sensitivity")
    add_body(doc, "Seven one-at-a-time settings were tested. The selector used only mean validation OA across the five seeds. Test metrics below are descriptive after selection and were not used to choose the setting.")
    sens = stage32["sensitivity_summary"]
    rows = []
    for name in ["base", "lambda1_low", "lambda1_high", "lambda2_low", "lambda2_high", "lambda3_low", "lambda3_high"]:
        s = sens[name]
        p = s["parameters"]
        rows.append([
            name,
            f"({p['lambda1']}, {p['lambda2']}, {p['lambda3']})",
            f"{s['validation_accuracy']['mean']:.6f} +/- {s['validation_accuracy']['std']:.6f}",
            f"{s['test_overall_accuracy']['mean']:.6f} +/- {s['test_overall_accuracy']['std']:.6f}",
            f"{s['test_kappa']['mean']:.6f} +/- {s['test_kappa']['std']:.6f}",
            f"{s['convergence_rate']:.1f}",
        ])
    add_table(doc, ["Setting", "(lambda1, lambda2, lambda3)", "Validation OA", "Test OA", "Test Kappa", "Conv. rate"], rows, widths=[1.0, 1.65, 1.35, 1.25, 1.25, 0.8], font_size=7.4)
    add_body(doc, "Validation selected lambda1=0.2, lambda2=0.1, lambda3=0.001. The higher lambda1 setting gave the strongest validation average and also the highest descriptive test averages. Lower lambda2 was close. Higher lambda2 converged in only four of five runs, while lambda3 changes had negligible effect in this reduced experiment.")
    subheading(doc, "Limitations and interpretation")
    for t in [
        "The experiment is a student-scale study with 100 training spectra per class, not the full training region.",
        "The buffered split separates rows but not every two-dimensional neighborhood direction and discards labeled buffer pixels.",
        "The graph, RBF SVM settings and stopping tolerance are documented implementation choices where the paper is silent.",
        "The selected-parameter test summary is cautious evidence because the same five held-out regions are summarized after validation-based selection.",
        "No claim of exact reproduction is made, and no official author code or pretrained model was used.",
    ]:
        add_bullet(doc, t)

    section_heading(doc, "5. What I have learnt from this AI assignment")
    add_body(doc, "I learned that selecting an algorithm is not only a search problem. A recent method can be a poor project choice if its data, code or evaluation protocol cannot be verified. JSLLDA became suitable because its paper, public dataset and mathematical structure could be checked together, while the more recent alternatives had weaker reproducibility evidence for this assignment.")
    add_body(doc, "The most important technical lesson was that equations require dimensional and optimization checks. The paper's L2,1 definition incorrectly presents row and column groupings as equal, and its printed P update appears inconsistent with the derivative of the stated graph term. The residual grouping, the factor of two, and the infeasible W constraint for k greater than the number of classes were resolved before trusting an experiment. These are mathematical issues, not cosmetic coding details.")
    add_body(doc, "The experiments also showed why convergence and leakage must be tested separately from accuracy. The first 30-iteration JSLLDA result had a plausible objective and high accuracy but a primal residual of 0.01556, so it was not final. Increasing the iteration limit and logging primal and dual residuals showed stable convergence. Replacing an interleaved row split with buffered spatial blocks reduced the apparent accuracy, but made the spatial evaluation more defensible.")
    add_body(doc, "Finally, I learned how an AI agent can support a complete research workflow: discover candidates, verify primary sources, derive updates, write and test an implementation, audit data, detect inconsistencies, run controlled experiments and preserve evidence in logs. The agent is useful only when its outputs are constrained by checksums, dimensions, leakage rules, convergence diagnostics and explicit uncertainty. An AI-generated result still needs human approval before publication or a public code release.")

    section_heading(doc, "6. Webpage link of source codes")
    add_body(doc, "No verified public source-code webpage exists for this project at the time of submission. The independent implementation remains local and has not been uploaded. Therefore, the required source-code link is pending and should not be replaced with an unrelated repository or an unverified JSLLDA fork.")
    add_body(doc, "After approval, a suitable repository could use the following structure:")
    add_code(doc, "cisc3024-jsll-da/\n  README.md\n  requirements.txt\n  src/jsll_da/core.py\n  src/jsll_da/data.py\n  src/jsll_da/baseline.py\n  src/jsll_da/metrics.py\n  src/run_salinas_experiment.py\n  src/run_stage3_2.py\n  data/audit/        # checksums, split metadata, no large raw data by default\n  logs/              # reproducible JSON experiment logs\n  reports/           # final report and figures")
    add_body(doc, "The repository should be published only after removing local paths, deciding how the dataset license and download instructions will be handled, and checking that no raw dataset or private metadata is committed. The current verified local source files and logs are the evidence used for this report.")
    subheading(doc, "Verified references")
    refs = [
        ("[1]", "Cao, C.-Y., Li, M.-T., Deng, Y.-J., Ren, L., Liu, Y. and Zhu, X. (2024). Joint Sparse Local Linear Discriminant Analysis for Feature Dimensionality Reduction of Hyperspectral Images. Remote Sensing, 16(22), 4287.", "https://doi.org/10.3390/rs16224287"),
        ("[2]", "Yang, J., Yin, W., Zhang, Y. and Wang, Y. (2009). A Fast Algorithm for Edge-Preserving Variational Multichannel Image Restoration. SIAM Journal on Imaging Sciences, 2(2), 569-592.", "https://doi.org/10.1137/080730421"),
        ("[3]", "University of the Basque Country. Hyperspectral Remote Sensing Scenes: Salinas scene and ground truth.", "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"),
        ("[4]", "NASA/JPL AVIRIS Salinas Hyperspectral Dataset. Zenodo record 15771735, including file checksums used in the audit.", "https://doi.org/10.5281/zenodo.15771735"),
        ("[5]", "danaroth. Salinas dataset mirror used for local acquisition; provenance description identifies the EHU/NASA AVIRIS scene.", "https://huggingface.co/datasets/danaroth/salinas"),
        ("[6]", "NumPy, SciPy and scikit-learn documentation for the numerical and baseline components.", "https://scikit-learn.org"),
    ]
    for label, text, url in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.1)
        p.paragraph_format.first_line_indent = Inches(-0.1)
        p.paragraph_format.space_after = Pt(4)
        p.add_run(label + " " + text + " ")
        add_hyperlink(p, url, url)
    add_body(doc, "The original paper's reported metrics, the local experiment metrics, and the implementation assumptions are kept separate throughout this report. The local values come from the project JSON logs and are not substituted for the paper's values.")

    output_path = OUT / "final_assignment3_report.docx"
    doc.save(output_path)
    print(output_path)


if __name__ == "__main__":
    build()
