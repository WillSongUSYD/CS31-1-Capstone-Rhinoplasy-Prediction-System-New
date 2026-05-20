# CS31 源码说明

这是 CS31 rhinoplasty outcome prediction 项目的源码整理版。

## 核心目录

- `desktop/`：最终 Mac App 源码。
- `ml/`：数据索引、数据准备、模型训练、模型评估代码。
- `backend/`：FastAPI 后端接口，主要保留为调试和历史实现。
- `frontend/`：早期 React Web 端，最终展示以 Mac App 为准。
- `tests/`：自动化测试。
- `reports/`：项目报告、展示文档和阶段文档。
- `models/outcome_v3_512/sd_inpaint_nose_v6/step_10000/`：最终 LoRA 权重。
- `artifacts/eval_sd_inpaint_v6/`：最终 V6 评估结果。

## 运行测试

```bash
python -m pytest -q
```

## 运行 Mac App 源码

```bash
python -m desktop
```

第一次运行会触发 4GB Stable Diffusion Inpainting 基础模型下载。

## 重要说明

这个源码包不包含原始患者图片、虚拟环境、node_modules、训练缓存和 4GB Stable Diffusion 基础模型。最终可展示 App 在交接包根目录的 `CS31-1-Rhinoplasty-Prediction-Studio_MacApp.zip` 里。
