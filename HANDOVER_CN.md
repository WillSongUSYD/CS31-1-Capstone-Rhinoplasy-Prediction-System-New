# CS31-1 鼻整形预测工作室 — 项目交接文档

**文档版本：** 1.0
**最后更新：** 2026-05-20
**适用应用版本：** v1.1.0

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码仓库](#2-代码仓库)
3. [第一部分 — 最终用户指南](#第一部分--最终用户指南)
   - [A.1 应用功能简介](#a1-应用功能简介)
   - [A.2 系统要求](#a2-系统要求)
   - [A.3 安装](#a3-安装)
   - [A.4 首次启动 — 模型下载](#a4-首次启动--模型下载)
   - [A.5 使用应用](#a5-使用应用)
   - [A.6 照片要求与建议](#a6-照片要求与建议)
   - [A.7 文件位置](#a7-文件位置)
   - [A.8 故障排查](#a8-故障排查)
4. [第二部分 — 开发者指南](#第二部分--开发者指南)
   - [B.1 仓库结构](#b1-仓库结构)
   - [B.2 三种部署形态](#b2-三种部署形态)
   - [B.3 开发环境搭建](#b3-开发环境搭建)
   - [B.4 机器学习流水线](#b4-机器学习流水线)
   - [B.5 后端 API](#b5-后端-api)
   - [B.6 前端](#b6-前端)
   - [B.7 测试](#b7-测试)
   - [B.8 构建桌面应用](#b8-构建桌面应用)
   - [B.9 架构说明](#b9-架构说明)
   - [B.10 数据与患者隐私](#b10-数据与患者隐私)
5. [第三部分 — 附录](#第三部分--附录)

---

## 1. 项目概述

**CS31-1 鼻整形预测工作室**是一个研究原型，可根据一张术前侧脸照片预测鼻整形
（隆鼻手术）的术后效果。给定患者的"术前"侧脸照片，系统会生成一张预测的"术后"
图像，并附带一段对鼻部几何变化的简短描述。

本项目有三种部署形态：

| 形态 | 说明 | 主要使用者 |
|---|---|---|
| **桌面应用** | 自带图形界面、可独立运行的 macOS 与 Windows 应用 | 临床医生 / 最终用户 |
| **Python 机器学习流水线** | 数据集索引、模型训练、评估 | 研究人员 / 开发者 |
| **Web 应用** | FastAPI 后端 + React 前端，用于浏览器端推理 | 开发者 / 演示 |

本交接文档分为**第一部分**（面向桌面应用的最终用户）与**第二部分**（面向将要
维护或扩展源代码的开发者）。

> **状态说明：** 本项目为研究原型。预测结果是 AI 生成的近似效果，用于辅助沟通
> 与可视化，**并非**对手术结果的医学保证。

---

## 2. 代码仓库

**仓库地址：**

```
https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New
```

- **默认分支：** `main`
- **大文件：** 训练好的 LoRA 模型权重通过 **Git LFS** 存储。克隆前必须先安装
  Git LFS，否则权重文件会变成一个几百字节的文本指针，而不是真实的约 28 MB 模型。

**克隆仓库：**

```bash
# 每台机器只需安装一次 Git LFS
git lfs install

# 克隆仓库
git clone https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New.git
cd CS31-1-Capstone-Rhinoplasy-Prediction-System-New

# 确认 LFS 文件已拉取（应约为 28 MB，而非几百字节）
git lfs pull
```

所有应用源代码位于 `CS31_Source_Code_TuohengZhang/` 目录下。除非另有说明，
第二部分中的所有命令都在该目录内执行。

---

# 第一部分 — 最终用户指南

本部分面向将要**安装并使用**桌面应用的人员，无需任何编程知识。

## A.1 应用功能简介

1. 你提供一张患者术前的**侧脸照片**。
2. 应用分析鼻部区域并生成一张**预测的术后图像**。
3. 应用将原图与预测图**并排显示**以便对比。
4. 应用生成一段对鼻部预测变化的简短文字说明。
5. 你可以将预测图像**保存**到电脑。

## A.2 系统要求

| | macOS | Windows |
|---|---|---|
| 操作系统 | macOS 12 (Monterey) 或更新版本，**Apple 芯片**（M1/M2/M3/M4） | Windows 10 或 11，64 位 |
| 内存 (RAM) | 最低 8 GB | 推荐 16 GB（最低 8 GB） |
| 可用磁盘空间 | **至少 8 GB 可用空间** | **至少 8 GB 可用空间** |
| 网络 | 首次启动时需联网**一次**，以下载 AI 模型（约 4 GB） | 首次启动时需联网**一次**，以下载 AI 模型（约 4 GB） |

> **⚠ 磁盘空间 —— 请务必阅读。** 本应用需要相当数量的可用磁盘空间。开始前请
> 确保**至少有 8 GB 可用空间**：
>
> - **解压后的应用文件夹**在 Windows 上约为 **2 GB**（下载的 `.zip` 较小——
>   解压后会变大）。
> - **首次启动下载的 AI 模型**还需额外约 **4 GB**，且它保存在与应用文件夹
>   **不同**的位置（见 [A.7 文件位置](#a7-文件位置)）——该磁盘/位置同样必须有
>   足够的可用空间。
> - 还需为下载的 `.zip` 本身及运行所需空间预留余量。
>
> 若任一位置空间不足，应用会出现解压失败、模型下载失败或无法生成预测的情况。

## A.3 安装

应用以压缩 `.zip` 文件形式分发，可从仓库的 **Releases（发布）** 页面获取
（见第 2 节）。

### macOS

1. 下载 macOS 版 `.zip` 并双击解压。
2. 将 **`CS31-1-Rhinoplasty-Prediction-Studio.app`** 拖入 **应用程序
   (Applications)** 文件夹。
3. 由于该应用未使用付费的 Apple 开发者证书签名，首次打开时 macOS 会拦截它。
   批准方式如下：
   - **右键点击**（或按住 Control 点击）应用图标，选择 **打开**。
   - 出现无法验证开发者的警告，点击 **完成 / 取消**。
   - 打开 **系统设置 → 隐私与安全性**。
   - 向下滚动到"安全性"部分，会看到类似
     *"已阻止使用 CS31-1-Rhinoplasty-Prediction-Studio..."* 的提示，
     点击 **仍要打开**。
   - 再次确认，按提示输入 Mac 密码。
4. 应用现在即可打开。此批准操作仅需进行**一次**。

### Windows

1. 开始前，请确认有**至少 2 GB 的可用磁盘空间**用于存放解压后的文件夹
   （另需约 4 GB 用于模型——见 A.4）。
2. 下载 Windows 版 `.zip` 并解压到任意位置（例如桌面）。
3. 打开解压后的文件夹，其中包含：

   | 项目 | 用途 |
   |---|---|
   | `CS31-1-Rhinoplasty-Prediction-Studio.exe` | 应用程序，双击运行。 |
   | `download_sd_model_v3.bat` | 备用 AI 模型下载器（见故障排查）。 |
   | `README.txt` | 快速参考。 |
   | `FIRST_LAUNCH.txt` | 首次启动与下载说明。 |
   | `_internal\` | 程序文件。**请勿删除、移动或重命名。** |

4. 请将以上**所有**项目保持在同一文件夹内。若缺少 `_internal` 文件夹，
   应用将无法启动。
5. 双击 **`CS31-1-Rhinoplasty-Prediction-Studio.exe`** 运行。首次运行时
   Windows SmartScreen 可能弹出警告 —— 点击 **更多信息 → 仍要运行**。

## A.4 首次启动 — 模型下载

应用首次启动时，必须下载一个**约 4 GB 的 AI 模型**
（Stable Diffusion 1.5 Inpainting）。此过程**仅进行一次**，之后每次启动都会
立即完成。

> **⚠ 磁盘空间：** 模型在其保存位置（见 [A.7](#a7-文件位置)）需要约
> **4 GB 可用空间**——该位置与应用文件夹**相互独立**。若该磁盘空间不足，
> 下载将会失败。

1. 确保网络连接**稳定**，并有足够的可用磁盘空间（模型约需 4 GB）。
2. 启动应用。
3. 会出现一个引导窗口并开始下载。
4. 在下载完成前请**保持窗口打开**。视网络情况，通常需要 **10–30 分钟**。
5. 下载完成后，应用即可使用。

若应用内下载失败，请参见 [A.8 故障排查](#a8-故障排查)。

## A.5 使用应用

1. **载入照片** —— 将一张术前侧脸照片拖入应用窗口，或使用文件选择器选取。
2. **自动开始预测** —— 一旦载入有效照片，应用即开始生成预测，并在处理过程中
   显示进度提示。
3. **查看结果** —— 原图与预测的术后图像并排显示，并附带一段简短的文字说明
   预测的变化。
4. **保存** —— 使用"保存"按钮将预测图像写入电脑。在 macOS 上，默认保存位置为
   `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/`。

## A.6 照片要求与建议

预测质量在很大程度上取决于输入照片，请遵循以下指引。

**必须满足：**

- 必须是**真正的侧脸**——从正面旋转 90° 的角度拍摄。
- 照片中**仅有一个人**。
- 文件格式：**JPEG、PNG 或 WEBP**。
- 最小尺寸：**512 × 512 像素**。

**获得最佳效果的建议：**

- **从肩部以上取景。** 不要包含胸部、全身或大量衣物。多余的身体区域会把模型的
  注意力从面部分散开，导致**预测图像模糊**。
- **使用干净、无杂物的背景**（纯色墙面效果很好）。杂乱的背景会引入视觉噪声，
  降低预测质量。
- **尽量使用分辨率最高的照片。** 预测图像以与输入相同的分辨率生成，因此越清晰
  的照片，预测结果越清晰。1024 像素或更大明显优于 512 像素的最低要求。
- 确保**鼻部清晰可见**且光照良好。

## A.7 文件位置

| 项目 | macOS | Windows |
|---|---|---|
| 下载的 AI 模型 | `~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/models/sd_base/inpaint/` | `%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\models\sd_base\inpaint\` |
| 日志文件 | `~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/cs31-rhinoplasty-prediction-studio.log` | `%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\cs31-rhinoplasty-prediction-studio.log` |
| 已保存的预测图 | `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/` | `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/` |

> 在 Windows 上，将路径（包含 `%APPDATA%`）粘贴到文件资源管理器的地址栏，
> 即可直接打开该文件夹。

## A.8 故障排查

| 问题 | 原因与解决方法 |
|---|---|
| **macOS：提示"无法打开应用" / "身份不明的开发者"** | 未签名应用的正常现象。按 [A.3](#a3-安装) 中的"右键 → 打开"批准步骤操作，仅需一次。 |
| **Windows：SmartScreen 警告** | 未签名应用的正常现象。点击 **更多信息 → 仍要运行**。 |
| **Windows：应用完全无法启动** | 很可能是 `_internal` 文件夹缺失或与 `.exe` 分离。请重新解压原始下载文件，并保持所有项目在一起。 |
| **首次启动模型下载失败或卡住（Windows）** | 关闭应用，双击应用文件夹中的 **`download_sd_model_v3.bat`**。它会查找或安装兼容的 Python，创建隔离环境，并直接下载模型。启动时命令窗口可能看起来卡住长达 60 秒——这是正常现象。当显示 *"Download complete"* 后，关闭窗口并重启应用。 |
| **首次启动模型下载失败（macOS）** | 确认网络连接稳定，然后退出并重新打开应用以重试下载。查看日志文件（见 [A.7](#a7-文件位置)）以获取具体错误信息。 |
| **预测结果模糊** | 几乎都是照片质量问题。请按 [A.6](#a6-照片要求与建议) 重新拍摄：紧凑取景于面部、使用纯色背景、并使用高分辨率图片。 |
| **应用运行缓慢** | 推理计算量很大。在 Windows 上使用 CPU 运行；在 Apple 芯片 Mac 上使用 GPU（MPS）。请关闭其他占用资源的程序，并为每次预测预留更多时间。 |
| **出现其他问题** | 打开**日志文件**（见 [A.7](#a7-文件位置)）——其中记录了详细的错误信息。报告问题时请附上该文件。 |

---

# 第二部分 — 开发者指南

本部分面向将要维护、重新构建或扩展本项目的开发者。

## B.1 仓库结构

所有源代码位于 `CS31_Source_Code_TuohengZhang/` 下：

```
CS31_Source_Code_TuohengZhang/
├── ml/              机器学习流水线：数据集索引、训练、评估、模型
├── backend/         FastAPI API、推理调度、SQLite 历史记录
├── frontend/        React + Vite Web 应用
├── desktop/         PyQt6 桌面应用 + 构建脚本（macOS 与 Windows）
├── data/            清单、数据划分、标注模板（不含患者图像）
├── models/          训练好的检查点与 LoRA 权重
├── artifacts/        准备好的图像对与预测输出（已被 gitignore）
├── reports/         报告草稿与评审模板
├── tests/           pytest 测试套件
└── requirements.txt Python 依赖
```

`.github/workflows/` 目录（位于仓库根目录）存放自动化的 Windows 构建流水线。

## B.2 三种部署形态

1. **Python 机器学习流水线**（`ml/`）—— 构建数据集、训练模型、评估模型，
   仅命令行操作。
2. **Web 应用**（`backend/` + `frontend/`）—— 一个执行推理的 FastAPI 服务器，
   配 React 浏览器界面。
3. **桌面应用**（`desktop/`）—— 一个可独立运行的 PyQt6 应用，打包为 macOS
   `.app`（通过 py2app）和 Windows 文件夹/`.exe`（通过 PyInstaller）。

三者共用 `ml/` 与 `backend/` 中相同的模型和推理代码。

## B.3 开发环境搭建

所有命令都在 `CS31_Source_Code_TuohengZhang/` 目录内执行。

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows

# 安装 Python 依赖
pip install -r requirements.txt

# 鼻部区域检测需要 InsightFace，但它不在 requirements.txt 中，
# 需要单独安装
pip install insightface
```

**Python 版本：** 支持 3.12（两个桌面构建均使用该版本）。

Stable Diffusion 训练/评估代码还需要 Hugging Face 相关库
（`diffusers`、`transformers`、`peft`、`safetensors`、`accelerate`），
它们**不在** `requirements.txt` 中——需要时请单独安装。

## B.4 机器学习流水线

```bash
# 从源图像构建 manifest.csv
python -m ml.index_dataset

# 生成 256 像素图像对与鼻部 ROI 裁剪
python -m ml.prepare_pairs

# 训练模型（全脸 256 像素模式）
python -m ml.train_outcome --model pix2pix --epochs 30

# 快速冒烟测试
python -m ml.train_outcome --model pix2pix --epochs 1 --limit 32

# 评估
python -m ml.evaluate_outcome --model pix2pix --limit 16

# Stable Diffusion + LoRA 训练 / 评估
python -m ml.train_sd_inpaint --base <sd_base_dir> --out <lora_out> --steps 5000
python -m ml.evaluate_sd_inpaint --base <sd_base_dir> --lora <lora_dir> --out <out>
```

在模型名后追加 `_nose`（例如 `pix2pix_nose`）即可使用鼻部 ROI 模式
（128 × 128）训练，而非全脸模式（256 × 256）。

## B.5 后端 API

```bash
python -m backend.serve     # FastAPI 运行于 http://127.0.0.1:8000
```

CORS 来源默认为 `localhost:5173`；可通过 `CS31_CORS_ORIGINS` 环境变量覆盖。

## B.6 前端

```bash
cd frontend
npm install
npm run dev      # Vite 开发服务器，运行于 http://localhost:5173
npm run build    # 生产构建，输出到 frontend/dist/（由 FastAPI 在 / 提供服务）
```

## B.7 测试

```bash
pytest tests/                                  # 运行全部测试
pytest tests/test_dataset_tools.py -v          # 运行单个文件
```

## B.8 构建桌面应用

### macOS（py2app）

```bash
pip install -r desktop/requirements-desktop.txt

# 推荐：构建脚本会处理 rpath 修补、代码签名与打包
bash desktop/scripts/build_app.sh        # 在 CS31_Source_Code_TuohengZhang/ 下运行

# 不打包直接运行（用于开发）
python -m desktop
```

构建前 V6 LoRA 权重文件必须存在；`build_app.sh` 会检查该文件，若缺失则提前
报错退出。

### Windows（PyInstaller）

Windows 应用通常由 GitHub Actions **自动构建**，也可在 Windows 机器上本地构建：

```bat
desktop\scripts\build_windows.bat
```

该脚本会安装依赖、对 `desktop/CS31_windows.spec` 运行 PyInstaller、组装分发
文件、校验产物，并生成 `.zip`。

**自动构建（推荐）：**

1. 在 GitHub 上打开仓库 → **Actions** 标签页。
2. 选择 **Build Windows App** 工作流。
3. 点击 **Run workflow** → **Run workflow**。
4. 完成后（约 12 分钟），从该次运行的 **Artifacts** 部分下载 `.zip`。

当推送版本标签（`v*`）时，该工作流也会自动触发。

**Windows 分发布局：** 构建会保持 zip 顶层整洁——只显示 `.exe`、
`download_sd_model_v3.bat`、`README.txt` 与 `FIRST_LAUNCH.txt`；所有运行时
文件（DLL、Python、模型数据）都位于 `_internal/` 文件夹内。

## B.9 架构说明

**机器学习模型**（`ml/models/`）—— 五种模型类型共用 `ml/runtime.py` 中的统一
调度器：`autoencoder`、`pix2pix`、`cyclegan`、`diffusion` 与
`sd_inpaint_nose`（Stable Diffusion 1.5 Inpainting + LoRA）。生产环境的桌面
应用使用 `sd_inpaint_nose` 路径。

**设备选择** —— `ml/runtime.py:get_device` 按以下优先级返回首个可用设备：
**MPS**（Apple 芯片 GPU）→ **CUDA** → **CPU**。检查点始终保存为 CPU 格式，
以便在任何平台上加载。

**推理流程** —— `backend/inference.py:run_prediction` 拆分成对图像、加载模型
（带 LRU 缓存）、运行推理、为 `_nose` 模型将鼻部区域贴回，并计算误差指标。
Stable Diffusion 预测会分派到 `backend/inference_sd.py`。

**鼻部 ROI** —— `ml/nose_roi.py` 使用 InsightFace（buffalo_l ONNX）进行
关键点检测，并配有 CLAHE 增强与比例启发式两级回退。鼻部掩膜是一个以双眼中点
与鼻尖为锚点的倾斜椭圆。

**桌面运行时路径** —— `desktop/core/paths.py` 通过 `sys._MEIPASS`
（PyInstaller）或 `.app` 的 Resources 目录（py2app）解析只读的捆绑资源，
并将用户可写状态放在各平台的应用数据目录下。`desktop/app.py` 会在任何
backend/ml 导入之前调用 `desktop.core.config.install_environment()`，因为
这些模块在导入时即读取环境变量。

**Stable Diffusion 基础模型** —— 约 4 GB 的 SD 1.5 Inpainting 基础权重
**不会**提交到仓库。桌面应用在首次启动时下载它们；开发时可将其放在
`models/sd_base/inpaint/`，并通过 `CS31_SD_BASE_DIR` 环境变量指向该路径。
体积小得多的 V6 LoRA 权重**会**随应用捆绑（通过仓库中的 Git LFS）。

## B.10 数据与患者隐私

**患者图像及可识别身份的元数据绝不能提交到 Git。**
以下内容通过 `.gitignore` 排除，它们包含患者面部图像或个人可识别信息（PII）：

- `CS31_Rhioplasty_Outcome_Prediction/` —— 原始数据集图像
- `data/manifest.csv` —— 文件名元数据
- `artifacts/dataset/` —— 派生的图像产物

交接项目时，请通过安全、私密的渠道转移患者数据集——**不要**通过公开仓库传递。

---

# 第三部分 — 附录

## C.1 快速参考 — 关键位置

| 内容 | 位置 |
|---|---|
| 代码仓库 | `https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New` |
| 源代码根目录 | `CS31_Source_Code_TuohengZhang/` |
| Windows 构建流水线 | `.github/workflows/build-windows.yml` |
| macOS 构建脚本 | `desktop/scripts/build_app.sh` |
| Windows 构建脚本 | `desktop/scripts/build_windows.bat` |
| Windows PyInstaller 规格文件 | `desktop/CS31_windows.spec` |
| 随应用分发的用户文档 | `desktop/dist_files/README.txt`、`desktop/dist_files/FIRST_LAUNCH.txt` |

## C.2 版本历史

| 版本 | 主要内容 |
|---|---|
| v1.1.0 | 首个正式 Windows 版本；界面重新设计；修复 LoRA 崩溃；修复模型下载服务器；按平台区分数据路径。 |
| v1.0.0 | 首个 macOS 版本（旧称 "CS31Preview"）。 |

## C.3 已知限制

- 预测结果是 AI 生成的近似效果，并非医学保证。
- Windows 推理在 CPU 上运行；每次预测耗时会比 Apple 芯片 Mac 更长。
- 首次启动的模型下载需要稳定的网络连接（约 4 GB）。
- 桌面应用未使用付费开发者证书签名，因此操作系统会在首次启动时显示一次性
  安全警告。

## C.4 交接备注

- GitHub 仓库归 **WillSongUSYD** 账户所有。要推送更改，维护者需要拥有该仓库的
  写入权限。
- 训练好的 LoRA 权重通过 **Git LFS** 跟踪——克隆或推送前请确保已安装 Git LFS。
- `main` 分支是唯一可信来源。Windows 构建流水线从 `main` 运行。
