# CS31 Rhinoplasty Outcome Prediction - Handoff Guide

这份文件夹是给组内同学和课堂展示使用的整理版交付包。

## Package Contents

- `CS31Preview_MacApp.zip`：已经打包好的 macOS App，用于课堂展示。
- `CS31_Source_Code.zip`：完整源码压缩包，包含桌面端、模型代码、后端、旧 Web 端、测试、报告材料和最终 LoRA 权重。
- `CS31_Source_Code/`：源码解压后的同内容文件夹，方便直接查看。

## App Demo Steps

1. 解压 `CS31Preview_MacApp.zip`。
2. 把 `CS31Preview.app` 拖到 `/Applications`，或者直接在解压后的文件夹里打开。
3. 因为没有 Apple Developer ID 签名，第一次打开需要右键/Control-click `CS31Preview.app`，选择 `Open`，弹窗里再点 `Open`。
4. 第一次运行会下载 Stable Diffusion Inpainting 基础模型，约 4GB。建议展示前提前打开一次，让模型下载完成。
5. 打开 App 后上传一张清晰侧脸照片，点击 `Generate Prediction`。
6. 生成完成后，界面会左边显示 `Original`，右边显示 `Predicted Result`。
7. 点击 `Save Result` 可以保存右侧生成图片。

## Source Code Map

- `desktop/`：最终交付的 Mac App 源码，包含 PyQt6 界面、模型下载、图片校验、推理线程、左右对比展示。
- `desktop/core/image_geometry.py`：解决图片比例问题的关键代码，负责把原图等比例放入 512x512 画布，再把生成结果还原为原图比例。
- `ml/`：数据处理、训练、评估、模型定义相关代码。
- `backend/`：FastAPI 推理接口代码，主要作为历史/调试接口保留。
- `frontend/`：早期 React Web 端源码，当前最终交付以 Mac App 为准。
- `tests/`：自动测试，包括数据处理和图片比例修复测试。
- `reports/`：报告、阶段文档、展示文档。
- `models/outcome_v3_512/sd_inpaint_nose_v6/step_10000/`：最终选择的 V6 `step_10000` LoRA 权重。
- `artifacts/eval_sd_inpaint_v6/`：V6 评估指标和定性结果图。

## Run From Source

建议使用 Python 3.9。

```bash
cd CS31_Source_Code
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r desktop/requirements-desktop.txt
python -m pytest -q
python -m desktop
```

注意：从源码运行桌面端时，也需要下载 4GB Stable Diffusion Inpainting 基础模型。打包好的 App 会自动引导下载。

## Not Included

为了减小体积和保护医疗图片隐私，这个交接包没有包含：

- 原始患者图片数据集 `CS31_Rhioplasty_Outcome_Prediction/`
- `.venv/`、`.nodeenv/`、`node_modules/`
- `build/` 缓存
- 大量训练中间 checkpoint 和数据集派生产物
- 4GB Stable Diffusion 基础模型

如果老师需要看数据来源，可以在展示时说明：原始数据用于本地训练和评估，但不随交接包分发；交接包只提供源码、最终 App、最终 LoRA 和评估/报告材料。

## Current Final Version

- 最终模型：V6 Stable Diffusion Inpainting LoRA `step_10000`
- 最终交付形式：macOS App `CS31Preview`
- 当前展示方式：生成完成后左右两图对比，不再使用拖拽分割线
- 当前比例修复：保存图和展示图都会保持上传原图的宽高比例
