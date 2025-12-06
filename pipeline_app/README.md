# Bridge disease pipeline

该模块基于仓库中的 `yolov11.pt` 与 `deeplabv3.pth` 权重，提供一体化的目标检测、语义分割与尺寸量化能力。

## 安装依赖

```bash
pip install -r pipeline_app/requirements.txt
```

> 提示：`opencv-python-headless` 已写入依赖，用于避免缺少 `libGL.so.1` 等系统库时的导入问题。

## 命令行使用

```bash
python -m pipeline_app.cli \
  --input path/to/image_or_dir \
  --resolution 5.0 \
  --output outputs/run1 \
  --seg-classes background crack exposed_rebar spalling rust efflorescence \
  --crack-labels crack 裂缝
```

- `--resolution`：R（像素/mm），用于将裂缝长度/宽度与其他病害面积换算为实际尺寸。
- 输出包括三张图片（检测框、分割掩码、检测+掩码叠加）与 `results.json` 尺寸统计，支持单张或目录批量处理。

## 前端界面

```bash
streamlit run -m pipeline_app.frontend
```

- 页面支持多图上传，填入 R（像素/mm） 后即可查看检测框、分割掩码与叠加效果，表格展示裂缝长度/宽度及其他病害面积。
- 运行结果会写入 `outputs/frontend/` 目录，便于下载与复用。
