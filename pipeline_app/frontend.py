from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_CRACK_LABELS, DEFAULT_SEG_CLASSES
from .pipeline import DiseasePipeline

st.set_page_config(page_title="Bridge Disease Pipeline", layout="wide")

st.title("桥梁病害一体化检测与分割")
st.write("使用训练好的 YOLOv11 与 DeeplabV3+ 权重，对病害进行检测、分割与尺寸量化。")

resolution = st.number_input("R（像素/mm）", min_value=0.0001, value=5.0, step=0.1)
seg_classes = st.text_input("分割类别（按索引顺序，逗号分隔）", ",".join(DEFAULT_SEG_CLASSES)).split(",")
seg_classes = [name.strip() for name in seg_classes if name.strip()]
crack_labels = st.text_input("裂缝标签（逗号分隔）", ",".join(DEFAULT_CRACK_LABELS)).split(",")
crack_labels = [c.strip() for c in crack_labels if c.strip()]

@st.cache_resource
def load_pipeline(seg_classes, crack_labels):
    return DiseasePipeline(
        pixels_per_mm=resolution,
        seg_classes=seg_classes,
        crack_labels=crack_labels,
    )

uploaded = st.file_uploader("上传待检测图片（支持多张）", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)
run = st.button("开始处理")

if run and uploaded:
    pipeline = load_pipeline(tuple(seg_classes), tuple(crack_labels))
    pipeline.pixels_per_mm = resolution
    output_dir = DEFAULT_OUTPUT_DIR / "frontend"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for file in uploaded:
            image = Image.open(file).convert("RGB")
            temp_path = tmp_dir / file.name
            image.save(temp_path)
            result = pipeline.process_image(temp_path, output_dir)

            st.subheader(file.name)
            cols = st.columns(3)
            cols[0].image(str(result.detection_image), caption="检测框")
            cols[1].image(str(result.mask_image), caption="分割掩码")
            cols[2].image(str(result.overlay_image), caption="检测 + 掩码叠加")

            if result.measurements:
                st.write("尺寸测量（单位：mm / mm²）：")
                st.table(result.measurements)
            else:
                st.info("未在该图片中找到前景掩码。")
elif run:
    st.warning("请先上传至少一张图片。")

st.markdown(
    """
**使用提示**
- R（像素/mm）用于将掩码像素尺寸转换为实际尺寸；裂缝输出长度和宽度，其他病害输出面积。
- 若分割类别数量与权重不匹配，可在输入框中调整类别列表或在命令行中使用 `--seg-classes` 进行覆盖。
- 处理结果与 `results.json` 会写入 `outputs/frontend/` 目录，便于下载和比对。
"""
)
