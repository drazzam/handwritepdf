# HandwritePDF Filler — Streamlit App (PyMuPDF + Drawable Canvas)
# -----------------------------------------------------------------
# Two‑page PDF form filler that:
#  • Loads your EMPTY form PDF and HANDWRITING font from repo assets by default.
#  • Lets you CALIBRATE exact rectangles visually (draw boxes on page preview) or by numbers.
#  • Uses your EXACT text sizes (e.g., 16.08, 14, 24) and chosen color.
#  • Wraps/clips text INSIDE each rectangle. Never spills outside.
#  • Embeds the font and exports a print‑ready 2‑page PDF.
#
# Files expected in repo (can be overridden by uploads at runtime):
#   assets/forms/empty_form.pdf
#   assets/fonts/MyHand.ttf
#   templates/template.json   (field rectangles + default sizes/color)
#
# How to run locally:
#   pip install streamlit pymupdf Pillow streamlit-drawable-canvas numpy
#   streamlit run app.py
#
# Note: Coordinates are in PDF points (1 pt = 1/72 inch). Canvas shows a raster
# preview; rectangles drawn on canvas are converted back to PDF points.

import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Utilities
# -------------------------------

def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    s = hex_color.strip().lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    return (int(s[0:2], 16)/255.0, int(s[2:4], 16)/255.0, int(s[4:6], 16)/255.0)

@dataclass
class FieldRect:
    page: int
    x: float
    y: float
    w: float
    h: float
    def as_rect(self) -> fitz.Rect:
        return fitz.Rect(self.x, self.y, self.x + self.w, self.y + self.h)

@dataclass
class TrainerRow:
    epa: FieldRect
    rubric: FieldRect
    strength: FieldRect
    improve: FieldRect

@dataclass
class Template:
    name: str
    page_sizes: List[Tuple[float, float]]  # [(w, h), ...] in pt
    date: FieldRect
    age_gender: FieldRect
    main_theme: FieldRect
    case_summary: FieldRect
    self_reflect_do: FieldRect
    self_reflect_dev: FieldRect
    epa_boxes: Optional[List[FieldRect]] = None
    trainer_rows: Optional[List[TrainerRow]] = None
    sizes: Optional[Dict[str, float]] = None
    color_hex: Optional[str] = None

    def to_json(self) -> str:
        def enc(obj):
            if isinstance(obj, FieldRect):
                return asdict(obj)
            if isinstance(obj, TrainerRow):
                return {
                    'epa': asdict(obj.epa),
                    'rubric': asdict(obj.rubric),
                    'strength': asdict(obj.strength),
                    'improve': asdict(obj.improve),
                }
            return obj
        d = asdict(self)
        if self.epa_boxes is not None:
            d['epa_boxes'] = [asdict(fr) for fr in self.epa_boxes]
        if self.trainer_rows is not None:
            d['trainer_rows'] = [enc(tr) for tr in self.trainer_rows]
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "Template":
        d = json.loads(s)
        def FR(x):
            return FieldRect(**x)
        def TR(x):
            return TrainerRow(epa=FR(x['epa']), rubric=FR(x['rubric']), strength=FR(x['strength']), improve=FR(x['improve']))
        epa_boxes = [FR(x) for x in d.get('epa_boxes', [])] if d.get('epa_boxes') else None
        trainer_rows = [TR(x) for x in d.get('trainer_rows', [])] if d.get('trainer_rows') else None
        return Template(
            name=d['name'],
            page_sizes=[tuple(x) for x in d.get('page_sizes', [])],
            date=FR(d['date']),
            age_gender=FR(d['age_gender']),
            main_theme=FR(d['main_theme']),
            case_summary=FR(d['case_summary']),
            self_reflect_do=FR(d['self_reflect_do']),
            self_reflect_dev=FR(d['self_reflect_dev']),
            epa_boxes=epa_boxes,
            trainer_rows=trainer_rows,
            sizes=d.get('sizes'),
            color_hex=d.get('color_hex'),
        )

# -------------------------------
# PDF operations
# -------------------------------

def register_font(doc: fitz.Document, font_bytes: bytes, alias: str = "HandFont") -> str:
    try:
        doc.insert_font(alias, file=io.BytesIO(font_bytes))
        return alias
    except TypeError:
        doc.insert_font(fontname=alias, fontfile=io.BytesIO(font_bytes))
        return alias


def draw_textbox(page: fitz.Page, rect: FieldRect, text: str, font_alias: str,
                 fontsize: float, color: Tuple[float, float, float], align: int = 0,
                 lineheight: Optional[float] = None):
    R = rect.as_rect()
    kwargs = {
        'fontsize': float(fontsize),
        'fontname': font_alias,
        'color': color,
        'align': align,
        'render_mode': 0,
    }
    if lineheight is not None:
        kwargs['lineheight'] = float(lineheight)
    page.insert_textbox(R, text, **kwargs)


def draw_centered_mark(page: fitz.Page, rect: FieldRect, mark: str, font_alias: str,
                       fontsize: float, color: Tuple[float, float, float]):
    R = rect.as_rect()
    tw = page.get_text_length(mark, fontname=font_alias, fontsize=float(fontsize))
    x = R.x0 + (R.width - tw) / 2.0
    y = R.y0 + (R.height - float(fontsize)) / 2.0 + float(fontsize) * 0.8
    page.insert_text(fitz.Point(x, y), mark, fontsize=float(fontsize), fontname=font_alias, color=color)


def render_page_image(doc: fitz.Document, page_index: int, scale: float = 2.0) -> Tuple[bytes, Tuple[int,int]]:
    page = doc[page_index]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), (pix.width, pix.height)


# -------------------------------
# Filling engine
# -------------------------------

def fill_pdf(template: Template, pdf_bytes: bytes, font_bytes: bytes,
             color_hex: str, sizes: Dict[str, float],
             data: Dict[str, str],
             epa_numbers: Optional[List[int]] = None) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    font_alias = register_font(doc, font_bytes, alias="HandFont")
    color = hex_to_rgb01(color_hex)

    # Page 1
    p0 = doc[0]
    draw_textbox(p0, template.date, data.get('date', ''), font_alias, sizes['date'], color)
    draw_textbox(p0, template.age_gender, data.get('age_gender', ''), font_alias, sizes['age_gender'], color)
    draw_textbox(p0, template.main_theme, data.get('main_theme', ''), font_alias, sizes['main_theme'], color)
    draw_textbox(p0, template.case_summary, data.get('case_summary', ''), font_alias, sizes['case_summary'], color)
    draw_textbox(p0, template.self_reflect_do, data.get('self_do', ''), font_alias, sizes['self_reflection'], color)
    draw_textbox(p0, template.self_reflect_dev, data.get('self_dev', ''), font_alias, sizes['self_reflection'], color)

    if template.epa_boxes and epa_numbers:
        for idx, fr in enumerate(template.epa_boxes, start=1):
            if idx in epa_numbers:
                draw_centered_mark(p0, fr, 'x', font_alias, sizes.get('epa_checkbox', 14.0), color)

    # Page 2
    if template.trainer_rows and len(doc) > 1:
        p1 = doc[1]
        for i, row in enumerate(template.trainer_rows, start=1):
            epa_val = data.get(f'row{i}_epa', '')
            rub_val = data.get(f'row{i}_rubric', '')
            str_val = data.get(f'row{i}_strength', '')
            imp_val = data.get(f'row{i}_improve', '')
            if epa_val:
                draw_textbox(p1, row.epa, epa_val, font_alias, sizes['epa_col'], color, align=1)
            if rub_val:
                draw_textbox(p1, row.rubric, rub_val, font_alias, sizes['rubric_col'], color, align=1)
            if str_val:
                draw_textbox(p1, row.strength, str_val, font_alias, sizes['strength_col'], color)
            if imp_val:
                draw_textbox(p1, row.improve, imp_val, font_alias, sizes['improve_col'], color)

    out = io.BytesIO()
    doc.save(out, deflate=True)
    return out.getvalue()


# -------------------------------
# Canvas calibration helpers
# -------------------------------

def last_rect_from_canvas_json(json_data: dict) -> Optional[Tuple[float,float,float,float]]:
    """Return (left, top, width, height) in canvas pixels from the LAST drawn object."""
    if not json_data or 'objects' not in json_data or not json_data['objects']:
        return None
    obj = json_data['objects'][-1]
    left = float(obj.get('left', 0))
    top = float(obj.get('top', 0))
    width = float(obj.get('width', 0)) * float(obj.get('scaleX', 1))
    height = float(obj.get('height', 0)) * float(obj.get('scaleY', 1))
    return (left, top, width, height)


def canvas_rect_to_pdf_field(fr_canvas: Tuple[float,float,float,float], page_index: int,
                             page_pt_size: Tuple[float,float], img_px_size: Tuple[int,int]) -> FieldRect:
    """Map canvas rectangle (pixels) back to PDF points for the given page."""
    page_w_pt, page_h_pt = page_pt_size
    img_w_px, img_h_px = img_px_size
    sx = img_w_px / page_w_pt
    sy = img_h_px / page_h_pt
    # We render with uniform scale, but use independent just in case.
    x_pt = fr_canvas[0] / sx
    y_pt = fr_canvas[1] / sy
    w_pt = fr_canvas[2] / sx
    h_pt = fr_canvas[3] / sy
    return FieldRect(page=page_index, x=x_pt, y=y_pt, w=w_pt, h=h_pt)


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="🖊️ HandwritePDF Filler", layout="wide")
st.title("🖊️ HandwritePDF — Two‑Page Form Filler")

# Load defaults from repo assets if present
DEFAULT_FORM = Path("assets/forms/empty_form.pdf")
DEFAULT_FONT = Path("assets/fonts/MyHand.ttf")
DEFAULT_TEMPLATE = Path("templates/template.json")

with st.sidebar:
    st.header("Assets")
    up_pdf = st.file_uploader("Empty 2‑page form PDF", type=["pdf"], key="pdf")
    up_font = st.file_uploader("Handwriting font (TTF/OTF)", type=["ttf", "otf"], key="font")
    up_tmpl = st.file_uploader("Load template (JSON)", type=["json"], key="tmpl")

# Read assets
pdf_bytes = up_pdf.read() if up_pdf else (DEFAULT_FORM.read_bytes() if DEFAULT_FORM.exists() else None)
font_bytes = up_font.read() if up_font else (DEFAULT_FONT.read_bytes() if DEFAULT_FONT.exists() else None)

# Prepare doc for previews & sizes
doc_ref = fitz.open(stream=pdf_bytes, filetype='pdf') if pdf_bytes else None
page_sizes_pt: Optional[List[Tuple[float,float]]] = None
if doc_ref:
    page_sizes_pt = [(p.rect.width, p.rect.height) for p in doc_ref]

# Session template
if 'template' not in st.session_state:
    if up_tmpl is not None:
        st.session_state.template = Template.from_json(up_tmpl.read().decode('utf-8'))
    elif DEFAULT_TEMPLATE.exists():
        st.session_state.template = Template.from_json(DEFAULT_TEMPLATE.read_text())
    else:
        st.session_state.template = None

# Build default template if needed
if (st.session_state.template is None) and page_sizes_pt:
    w0, h0 = page_sizes_pt[0]
    w1, h1 = page_sizes_pt[1] if len(page_sizes_pt) > 1 else (w0, h0)
    st.session_state.template = Template(
        name="DefaultTemplate",
        page_sizes=[(w0, h0), (w1, h1)],
        date=FieldRect(0, w0*0.13, h0*0.17, w0*0.18, 24),
        age_gender=FieldRect(0, w0*0.56, h0*0.17, w0*0.20, 24),
        main_theme=FieldRect(0, w0*0.23, h0*0.23, w0*0.66, 24),
        case_summary=FieldRect(0, w0*0.06, h0*0.30, w0*0.88, h0*0.17),
        self_reflect_do=FieldRect(0, w0*0.06, h0*0.49, w0*0.88, h0*0.10),
        self_reflect_dev=FieldRect(0, w0*0.06, h0*0.61, w0*0.88, h0*0.10),
        epa_boxes=None,
        trainer_rows=[
            TrainerRow(FieldRect(1, w1*0.06, h1*0.24, w1*0.10, 28), FieldRect(1, w1*0.17, h1*0.24, w1*0.10, 28), FieldRect(1, w1*0.28, h1*0.24, w1*0.32, h1*0.05), FieldRect(1, w1*0.61, h1*0.24, w1*0.33, h1*0.05)),
            TrainerRow(FieldRect(1, w1*0.06, h1*0.31, w1*0.10, 28), FieldRect(1, w1*0.17, h1*0.31, w1*0.10, 28), FieldRect(1, w1*0.28, h1*0.31, w1*0.32, h1*0.05), FieldRect(1, w1*0.61, h1*0.31, w1*0.33, h1*0.05)),
            TrainerRow(FieldRect(1, w1*0.06, h1*0.38, w1*0.10, 28), FieldRect(1, w1*0.17, h1*0.38, w1*0.10, 28), FieldRect(1, w1*0.28, h1*0.38, w1*0.32, h1*0.05), FieldRect(1, w1*0.61, h1*0.38, w1*0.33, h1*0.05)),
        ],
        sizes={
            'date': 16.08, 'age_gender': 16.08, 'main_theme': 16.08,
            'case_summary': 14.0, 'self_reflection': 14.0,
            'epa_col': 24.0, 'rubric_col': 24.0, 'strength_col': 14.0, 'improve_col': 14.0,
            'epa_checkbox': 14.0,
        },
        color_hex="#1F4FB2",
    )

# Sidebar: sizes & color
with st.sidebar:
    st.header("Appearance")
    tmpl = st.session_state.template
    default_color = tmpl.color_hex or "#1F4FB2"
    color_hex = st.text_input("Handwriting color (hex)", value=default_color)

    sizes = tmpl.sizes or {}
    def _sz(key, val, step=0.01):
        return st.number_input(key, value=float(sizes.get(key, val)), step=step, format="%.2f")

    sizes = {
        'date': _sz('Date size', 16.08),
        'age_gender': _sz('Age & gender size', 16.08),
        'main_theme': _sz('Main theme size', 16.08),
        'case_summary': _sz('Case summary size', 14.0),
        'self_reflection': _sz('Self‑reflection size', 14.0),
        'epa_col': st.number_input('EPA column size (p2)', value=float(tmpl.sizes.get('epa_col', 24.0) if tmpl.sizes else 24.0), step=0.1, format='%.1f'),
        'rubric_col': st.number_input('Rubric column size (p2)', value=float(tmpl.sizes.get('rubric_col', 24.0) if tmpl.sizes else 24.0), step=0.1, format='%.1f'),
        'strength_col': st.number_input('Strength column size (p2)', value=float(tmpl.sizes.get('strength_col', 14.0) if tmpl.sizes else 14.0), step=0.1, format='%.1f'),
        'improve_col': st.number_input('Improvement column size (p2)', value=float(tmpl.sizes.get('improve_col', 14.0) if tmpl.sizes else 14.0), step=0.1, format='%.1f'),
        'epa_checkbox': st.number_input("EPA checkbox 'x' size (p1)", value=float(tmpl.sizes.get('epa_checkbox', 14.0) if tmpl.sizes else 14.0), step=0.1, format='%.1f'),
    }

    if st.button("💾 Save Template JSON"):
        tmpl.sizes = sizes
        tmpl.color_hex = color_hex
        s = tmpl.to_json()
        st.download_button("Download template.json", data=s, file_name="template.json", mime="application/json")

# Tabs
if not (pdf_bytes and font_bytes and doc_ref):
    st.warning("Please provide the empty 2‑page form PDF and the handwriting font (or place them in assets/).")
    st.stop()

TAB_CAL, TAB_FILL = st.tabs(["Calibrate", "Data Entry & Generate"]) 

# -------------------------------
# Calibrate Tab — visual drawing + numeric edit
# -------------------------------
with TAB_CAL:
    st.subheader("Calibrate exact rectangles — draw on the page preview, then assign to a field.")

    tmpl = st.session_state.template
    # Keep template's page sizes in sync with the loaded PDF
    if tmpl and page_sizes_pt and (len(tmpl.page_sizes) != len(page_sizes_pt) or any(abs(tmpl.page_sizes[i][0]-page_sizes_pt[i][0])>1 or abs(tmpl.page_sizes[i][1]-page_sizes_pt[i][1])>1 for i in range(len(page_sizes_pt)))):
        tmpl.page_sizes = page_sizes_pt

    # Page selector for calibration
    page_choice = st.radio("Choose page to calibrate", options=[0,1], format_func=lambda i: f"Page {i+1}", horizontal=True)

    # Render selected page at a comfortable scale for drawing
    render_scale = 1.5  # canvas image scale factor (px per pt)
    img_bytes, img_size = render_page_image(doc_ref, page_choice, scale=render_scale)
    img_w, img_h = img_size

    # Select field to assign
    field_options = [
        ('date','Date'), ('age_gender','Age & gender'), ('main_theme','Main theme'),
        ('case_summary','Case summary'), ('self_reflect_do','Self‑reflection: do right'), ('self_reflect_dev','Self‑reflection: develop'),
    ]

    # Trainer rows fields
    if tmpl.trainer_rows:
        for i in range(len(tmpl.trainer_rows)):
            idx = i+1
            field_options += [(f'row{idx}.epa', f'P2 Row {idx} — EPA'),
                              (f'row{idx}.rubric', f'P2 Row {idx} — Rubric'),
                              (f'row{idx}.strength', f'P2 Row {idx} — Strength'),
                              (f'row{idx}.improve', f'P2 Row {idx} — Improvement')]

    # Optional EPA boxes on page 1
    if tmpl.epa_boxes:
        for i, _ in enumerate(tmpl.epa_boxes, start=1):
            field_options.append((f'epa_box.{i}', f'P1 — EPA box #{i}'))

    st.markdown("**1) Select a field, 2) Draw a rectangle, 3) Click Assign.**")
    field_key, field_label = st.selectbox("Field to assign", options=field_options, format_func=lambda x: x[1], index=0)

    # Draw canvas
    c = st_canvas(
        fill_color="#00000000",  # transparent
        stroke_width=2,
        stroke_color="#1F4FB2",
        background_image=Image.open(io.BytesIO(img_bytes)),
        update_streamlit=True,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        key=f"canvas_page_{page_choice}",
    )

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Assign last drawn rectangle → selected field"):
            fr_can = last_rect_from_canvas_json(c.json_data)
            if fr_can is None:
                st.error("Draw a rectangle first.")
            else:
                # Only allow assignment if the field belongs to this page
                target_page = 0 if field_key in [k for k,_ in field_options[:6]] or field_key.startswith('epa_box') else 1
                if target_page != page_choice:
                    st.error(f"Selected field is on Page {target_page+1}. Switch page above.")
                else:
                    pdf_fr = canvas_rect_to_pdf_field(fr_can, page_choice, page_sizes_pt[page_choice], img_size)
                    # Assign
                    if field_key == 'date': tmpl.date = pdf_fr
                    elif field_key == 'age_gender': tmpl.age_gender = pdf_fr
                    elif field_key == 'main_theme': tmpl.main_theme = pdf_fr
                    elif field_key == 'case_summary': tmpl.case_summary = pdf_fr
                    elif field_key == 'self_reflect_do': tmpl.self_reflect_do = pdf_fr
                    elif field_key == 'self_reflect_dev': tmpl.self_reflect_dev = pdf_fr
                    elif field_key.startswith('row'):
                        # rowN.field
                        parts = field_key.split('.')
                        row_idx = int(parts[0][3:]) - 1
                        fld = parts[1]
                        row = tmpl.trainer_rows[row_idx]
                        if fld == 'epa': row.epa = pdf_fr
                        elif fld == 'rubric': row.rubric = pdf_fr
                        elif fld == 'strength': row.strength = pdf_fr
                        elif fld == 'improve': row.improve = pdf_fr
                    elif field_key.startswith('epa_box'):
                        box_idx = int(field_key.split('.')[-1]) - 1
                        tmpl.epa_boxes[box_idx] = pdf_fr
                    st.success(f"Assigned {field_label}.")

    with colB:
        if st.button("Quick test preview with placeholder text"):
            # Minimal preview to verify placements
            tmp_bytes = fill_pdf(
                template=tmpl,
                pdf_bytes=pdf_bytes,
                font_bytes=font_bytes,
                color_hex=color_hex,
                sizes={k: float(v) for k,v in sizes.items()},
                data={
                    'date': '2025-09-01',
                    'age_gender': '65y/M',
                    'main_theme': 'Hypertension FU',
                    'case_summary': 'BP 155/92. No CP/SOB. Counselled on meds and lifestyle. Plan: adjust dose.',
                    'self_do': 'Followed clinic protocol; safety‑netting given.',
                    'self_dev': 'Time management in OPD; reinforce salt restriction counselling.',
                },
                epa_numbers=[1,2],
            )
            doc_prev = fitz.open(stream=tmp_bytes, filetype='pdf')
            img0, _ = render_page_image(doc_prev, 0, scale=1.5)
            st.image(img0, caption="Page 1 preview (placeholder)")
            if len(doc_prev) > 1:
                img1, _ = render_page_image(doc_prev, 1, scale=1.5)
                st.image(img1, caption="Page 2 preview (placeholder)")
            doc_prev.close()

    st.divider()
    st.caption("Tip: You can also fine‑tune by numbers below.")

    def edit_rect(label: str, fr: FieldRect):
        c1, c2, c3, c4 = st.columns(4)
        fr.x = c1.number_input(f"{label}.x", value=float(fr.x), step=0.5)
        fr.y = c2.number_input(f"{label}.y", value=float(fr.y), step=0.5)
        fr.w = c3.number_input(f"{label}.w", value=float(fr.w), step=0.5)
        fr.h = c4.number_input(f"{label}.h", value=float(fr.h), step=0.5)

    st.markdown("### Page 1 boxes")
    edit_rect("Date", tmpl.date)
    edit_rect("AgeGender", tmpl.age_gender)
    edit_rect("MainTheme", tmpl.main_theme)
    edit_rect("CaseSummary", tmpl.case_summary)
    edit_rect("SelfDo", tmpl.self_reflect_do)
    edit_rect("SelfDev", tmpl.self_reflect_dev)

    if tmpl.trainer_rows:
        st.markdown("### Page 2 rows")
        for i, row in enumerate(tmpl.trainer_rows, start=1):
            st.write(f"Row {i}")
            edit_rect(f"r{i}.EPA", row.epa)
            edit_rect(f"r{i}.Rubric", row.rubric)
            edit_rect(f"r{i}.Strength", row.strength)
            edit_rect(f"r{i}.Improve", row.improve)

# -------------------------------
# Data Entry & Generate
# -------------------------------
with TAB_FILL:
    st.subheader("Data Entry — your exact content (no auto changes)")

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        date = st.text_input("Date", value="")
        age_gender = st.text_input("Age & Gender", value="")
    with col3:
        main_theme = st.text_input("Main theme of the case", value="")

    case_summary = st.text_area("Case summary", height=140)
    self_do = st.text_area("Self‑reflection — What did I do right?", height=110)
    self_dev = st.text_area("Self‑reflection — What needs development?", height=110)

    epa_str = st.text_input("EPA tested (comma‑separated numbers for page‑1 boxes; optional)", value="")
    epa_numbers = []
    if epa_str.strip():
        try:
            epa_numbers = [int(x.strip()) for x in epa_str.split(',') if x.strip()]
        except Exception:
            st.warning("EPA must be a comma‑separated list of integers.")

    rows_data = {}
    if st.session_state.template.trainer_rows:
        st.markdown("### Trainer page rows (leave blank if trainer fills)")
        for i in range(len(st.session_state.template.trainer_rows)):
            with st.expander(f"Row {i+1}"):
                c1, c2, c3, c4 = st.columns([1,1,3,3])
                rows_data[f'row{i+1}_epa'] = c1.text_input(f"Row {i+1} — EPA")
                rows_data[f'row{i+1}_rubric'] = c2.text_input(f"Row {i+1} — Rubric")
                rows_data[f'row{i+1}_strength'] = c3.text_area(f"Row {i+1} — Strength", height=80)
                rows_data[f'row{i+1}_improve'] = c4.text_area(f"Row {i+1} — Improvement", height=80)

    do_gen = st.button("🧾 Generate PDF")

    if do_gen:
        if not (pdf_bytes and font_bytes):
            st.error("Upload/Provide both the empty form PDF and the handwriting font.")
        else:
            data = {'date': date, 'age_gender': age_gender, 'main_theme': main_theme,
                    'case_summary': case_summary, 'self_do': self_do, 'self_dev': self_dev}
            data.update(rows_data)
            out_bytes = fill_pdf(
                template=st.session_state.template,
                pdf_bytes=pdf_bytes,
                font_bytes=font_bytes,
                color_hex=color_hex,
                sizes={k: float(v) for k,v in sizes.items()},
                data=data,
                epa_numbers=epa_numbers,
            )
            st.success("PDF generated.")

            # In‑app preview of the generated PDF pages
            doc_prev = fitz.open(stream=out_bytes, filetype='pdf')
            img0, _ = render_page_image(doc_prev, 0, scale=1.5)
            st.image(img0, caption="Page 1 — final")
            if len(doc_prev) > 1:
                img1, _ = render_page_image(doc_prev, 1, scale=1.5)
                st.image(img1, caption="Page 2 — final")
            doc_prev.close()

            st.download_button("⬇️ Download filled_case.pdf", data=out_bytes, file_name="filled_case.pdf", mime="application/pdf")

st.caption("Calibrate once, then reuse — your input is placed verbatim with the exact sizes you specified, and clipped inside each box.")
