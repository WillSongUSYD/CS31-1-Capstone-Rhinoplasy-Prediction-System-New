# CS31Preview Windows — 开发者问题报告与修复说明

**日期：** 2026-05-19
**分支 / 提交：** `main` — `d749b58`、`d8a542a`
**需要执行的操作：** 拉取最新提交，重新编译 `CS31Preview.exe`

---

## 背景说明

Windows 可执行文件 `CS31Preview.exe` 是从一个以 macOS 为主的代码库编译而来的。在 Windows 上首次运行时发现了两个 Bug。这两个 Bug 均已在源代码中修复（提交记录 `d749b58` 和 `d8a542a`，位于 `main` 分支）。**开发者只需拉取最新提交并重新编译 Windows exe 即可。**

---

## Bug 1 — 即使模型已经下载完成，应用每次启动仍然弹出"下载模型"对话框

### 根本原因

`desktop/core/paths.py` 文件中的 `user_support_dir()` 函数，在**所有平台**上都硬编码返回 macOS 路径：

```python
# 旧代码 — 在 Windows 上路径错误
d = Path.home() / "Library" / "Application Support" / _APP_NAME
```

在 Windows 上，`Path.home()` 返回 `C:\Users\<用户名>`，所以旧代码计算出的路径为：

```
C:\Users\<用户名>\Library\Application Support\CS31Preview\
```

这个路径在标准 Windows 系统中根本不存在。exe 实际写入数据的位置是 Windows 标准数据目录：

```
C:\Users\<用户名>\AppData\Roaming\CS31Preview\    （即 %APPDATA%\CS31Preview\）
```

由于模型存在性检查函数 `is_sd_base_present()` 一直在错误的文件夹中查找模型，结果始终返回 `False`，因此每次启动都会弹出下载对话框，即便模型早已完整下载。

### 已应用的修复 — `desktop/core/paths.py`

```python
import platform
import os

def user_support_dir() -> Path:
    """各平台对应的可写应用数据目录，首次访问时自动创建。

    - Windows: %APPDATA%\\CS31Preview\\
    - macOS:   ~/Library/Application Support/CS31Preview/
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        d = Path(appdata) / _APP_NAME if appdata else Path.home() / _APP_NAME
    else:
        d = Path.home() / "Library" / "Application Support" / _APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
```

### 重新编译后的影响

- **新用户：** 从首次启动起，数据目录将正确指向 `%APPDATA%\CS31Preview\`，无需任何额外操作。
- **已将模型下载到错误路径的老用户**（路径为 `C:\Users\<用户名>\Library\Application Support\CS31Preview\models\sd_base\inpaint\`）需要手动将模型文件夹移至正确位置：`%APPDATA%\CS31Preview\models\sd_base\inpaint\`。

---

## Bug 2 — 应用内下载进度卡在 0%，无法启动下载

### 根本原因

`huggingface_hub >= 0.25.0` 版本引入了一项 CDN URL 验证机制。当下载端点设置为 `https://hf-mirror.com`（一个中国镜像站）时，LFS 二进制文件通过**非** `*.huggingface.co` 的 CDN 域名提供服务，新的验证机制会拒绝这些 URL，并报如下错误：

```
FileMetadataError: Distant resource does not seem to be on huggingface.co.
It is possible that a configuration issue prevents you from downloading
resources from https://huggingface.co.
```

应用内置的下载器（`desktop/core/downloader.py`）在运行时会将 `HF_ENDPOINT` 设置为 `https://hf-mirror.com`。如果编译环境安装了 `huggingface_hub >= 0.25`，PyInstaller 会将该版本打包进 exe，导致下载在 0% 处静默失败。

### 已应用的修复 — `desktop/requirements-desktop.txt`

```
# >=0.25 引入了 CDN URL 验证，会拒绝 hf-mirror.com 提供的 LFS 文件。
# 固定在该版本以下；0.21.x 是与本项目 mirror 端点及 diffusers/transformers 版本兼容的最后一个版本。
huggingface_hub>=0.19,<0.25
```

### 给开发者的关键说明

在配置编译环境以重新编译 exe 时，必须运行：

```bash
pip install -r desktop/requirements-desktop.txt
```

这将把 `huggingface_hub` 安装为 **0.19.x – 0.24.x** 版本（**不能是 0.25 或更高版本**）。请通过以下命令验证：

```bash
pip show huggingface-hub
# 预期输出：Version: 0.2x.x（其中次版本号小于 25）
```

如果编译机器已全局安装 `huggingface_hub >= 0.25`，项目 venv 中的固定版本要求会将其降级。**请勿忽略或覆盖此版本限制。**

---

## 其他文件（非 exe 文件，已在仓库中）

以下文件已完整，无需进一步修改代码。只需在 Windows 发行版 zip 包中与 `CS31Preview.exe` 一起包含即可：

| 文件 | 用途 |
|---|---|
| `desktop/download_sd_model.bat` | 用户首次启动前用于下载约 4 GB 模型的一键脚本。通过 `py` 启动器自动检测 Python 3.9–3.13；若未找到兼容版本则通过 `winget` 自动安装 Python 3.12；在 `%APPDATA%\CS31Preview\download_env` 创建独立 venv；直接从 `huggingface.co` 下载（不使用镜像） |
| `desktop/FIRST-LAUNCH.txt` | 面向用户的安装说明文件；顶部新增 Windows 专属指引，说明首次启动前必须先完成模型下载步骤 |

---

## 总结：开发者需要执行的操作

| 步骤 | 命令 / 操作 |
|---|---|
| 1. 拉取最新源代码 | `git pull` — 获取提交 `d749b58` 和 `d8a542a` |
| 2. 配置编译 venv | `pip install -r desktop/requirements-desktop.txt` |
| 3. 验证 huggingface_hub 版本 | `pip show huggingface-hub` → 版本必须 `< 0.25` |
| 4. 重新编译 exe | 使用 PyInstaller 从更新后的源代码编译 |
| 5. 打包发行版 | 将 `CS31Preview.exe`、`download_sd_model.bat` 和 `FIRST-LAUNCH.txt` 一并放入 zip 包 |

---

*生成日期：2026-05-19。所有源代码变更已提交至 `main` 分支。*
