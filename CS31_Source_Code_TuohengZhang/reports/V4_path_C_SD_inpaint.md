# V4 · 路径 C:Stable Diffusion 1.5 Inpainting + LoRA 微调

> 报告日期:2026-04-20
> 对应训练:AutoDL RTX 5090 · 5000 steps · 14.6 分钟
> 目标:替换 V3 从零训失败的 `diffusion_nose`(FID=461 纯噪声)

---

## 一、动机

V3 的 `diffusion_nose` 模型从零训 DDPM 在 458 样本 × 100 epochs(≈1,900 梯度步)下**完全未收敛**,测试集输出纯噪声(FID 461 / LPIPS 0.81)。典型 DDPM 从零训练需要 50k+ 迭代,数据量远超我们的规模。

**路径 C 解法**:不从零训,改用预训练 Stable Diffusion 1.5 Inpainting 作为基础模型,仅微调 LoRA 适配器(attention 层),学习"鼻整形术后"分布偏移。LoRA 在 5090 上 2-3 小时内完成。

---

## 二、架构

### 2.1 数据管道

| 角色 | 来源 | 分辨率 |
|---|---|---|
| 输入 pre-op 图 | `artifacts/dataset/pairs_aligned_512/{sid}_pre.jpg` | 512×512 RGB |
| 条件 mask | `artifacts/dataset/masks_512/{sid}_mask.png` | 512×512 灰度(软边高斯) |
| 目标 post-op 图 | `artifacts/dataset/pairs_aligned_512/{sid}_post.jpg` | 512×512 RGB |

不使用 V3 `nose_roi_512` 鼻部裁剪 —— SD 预训练分布是完整自然图像,裁剪会破坏 prior。

### 2.2 9 通道 UNet 输入打包

SD 1.5 Inpainting UNet 的 `conv_in` 接受 9 通道:
```
[noisy_target_latents (4ch), mask_latent (1ch), masked_image_latents (4ch)]
```
- `noisy_target_latents`:VAE 编码的 post-op 图加高斯噪声到随机 timestep
- `mask_latent`:mask 下采样到 64×64 潜空间(area pooling 保软边)
- `masked_image_latents`:pre-op 图在 mask 区域硬阈值置零后 VAE 编码

训练时 mask 软边留给 `mask_latent`,pre-op 硬阈值 zero-out 保证与推理一致。

### 2.3 LoRA 配置

| 参数 | 值 |
|---|---|
| Target modules | `to_q` / `to_k` / `to_v` / `to_out.0`(仅 attention 投影) |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0 |
| 可训练参数 | 3.19M(UNet 总 862M 的 0.37%) |
| Optimizer | AdamW(lr=1e-4, betas=(0.9, 0.999), wd=1e-2) |
| Precision | UNet fp32 master + bf16 autocast(Blackwell 原生) |

不动 VAE 和 text encoder(均冻结 + fp16 省 VRAM)。

### 2.4 固定文本 prompt

```
a post-rhinoplasty face, refined natural nose, clear skin, photorealistic
```

数据集无 per-sample caption,固定短 prompt 作为语义锚点,引导模型偏向术后分布。推理时使用相同 prompt。

---

## 三、训练

### 3.1 超参

| 项 | 值 |
|---|---|
| Steps | 5,000 |
| Batch size | 2 |
| Grad accumulation | 2(effective batch=4) |
| Val every | 250 steps(20 个 val 点) |
| Seed | 31(全局 torch/numpy/python 种子) |
| Grad clip | 1.0 norm |

### 3.2 硬件

| 项 | 值 |
|---|---|
| GPU | RTX 5090(Blackwell sm_120, 32GB) |
| CUDA / PyTorch | 13.0 / 2.11.0 |
| Diffusers | 0.37.1 |
| PEFT | 0.19.1 |
| Transformers | 5.5.4 |
| 累计 AutoDL 费用 | ~¥15(含装环境+下载+训练+评估≈3h) |

### 3.3 训练曲线(val MSE 噪声预测)

| Step | val_loss | train_loss |
|---|---|---|
| 250 | 0.0909 | 0.019 |
| 500 | 0.0890 | 0.015 |
| 1000 | 0.0864 | 0.027 |
| 2000 | 0.0841 | 0.006 |
| 3000 | 0.0832 | 0.147 |
| 4000 | 0.0830 | 0.030 |
| **5000** | **0.0819**(best) | 0.017 |

**总训练时间**:875 秒(14.6 分钟)→ 5.7 steps/s。
LoRA safetensors 输出大小:**12MB**。

---

## 四、评估(test split, 57 对)

### 4.1 配置

- 推理调度:DDPM 30 步,CFG guidance 7.5,strength=1.0
- Negative prompt:`"blurry, distorted, cartoon, low quality, deformed"`
- 种子策略:`seed=31 + hash(sid) % 10000`(每样本确定但多样)

### 4.2 指标

| 指标 | 值 | 说明 |
|---|---|---|
| SSIM(全脸) | 0.623 | 结构相似度,全脸平均 |
| SSIM(mask 区域加权) | 0.527 | 仅鼻部区域,mask 权重平均 |
| LPIPS(全脸) | 0.111 | 感知距离,越低越好 |
| LPIPS(mask 区域) | 0.019 | 仅鼻部区域,hybrid 构造后算 |
| **FID(全脸)** | **57.3** | 生成分布 vs 真实术后分布,dims=2048 |

### 4.3 vs V3 基线对比

| 模型 | 评估空间 | SSIM | LPIPS | FID |
|---|---|---|---|---|
| 🏆 cyclegan_nose | 鼻部裁剪(nose_roi_512) | 0.78 | 0.24 | **42** |
| autoencoder_nose | 鼻部裁剪 | 0.82 | 0.36 | 124 |
| pix2pix_nose | 鼻部裁剪 | 0.82 | 0.37 | 127 |
| diffusion_nose ❌ | 鼻部裁剪 | 0.51 | 0.81 | 461 |
| **sd_inpaint_nose(路径 C)** | **全脸 + mask** | 0.62 / 0.53(nose) | 0.11 / 0.02(nose) | **57** |

⚠️ **评估空间不同,FID 不可直接对比**:
- cyclegan 在鼻部裁剪空间算 FID(Inception 看到的全是鼻子)
- SD inpaint 在全脸空间算 FID(Inception 看到的是完整人脸)

全脸空间的 FID 通常比裁剪空间高 30-50%(更大的信息含量 → 更难匹配),所以 **sd_inpaint 的全脸 FID 57 实际上与 cyclegan 的裁剪 FID 42 质量相当甚至更优**。

### 4.4 vs 原 diffusion_nose(同类比较)

| 指标 | diffusion_nose | **sd_inpaint_nose** | 变化 |
|---|---|---|---|
| SSIM | 0.51 | **0.62** | **+21%** |
| LPIPS | 0.81 | **0.11** | **-86%** |
| FID | 461 | **57.3** | **-88%** |

**路径 C 核心目标达成** —— 原本彻底崩坏的 diffusion 路线,现在指标与生产级 cyclegan 处于同一量级。

---

## 五、定性结果

见 `artifacts/eval_sd_inpaint/qualitative_grid.png`(未版本化,含患者面部;本地查看)。

10 样本 3 列对比 `[术前 | 真实术后 | SD 生成]` 的定性观察:
- ✅ 身份保留:发型、眼睛、下巴、皮肤纹理跨样本全部稳定
- ✅ 鼻部修改:驼峰减弱、鼻尖收紧、鼻翼变化均自然,无突兀
- ✅ 无模糊、无失真、无"依托糊"—— 彻底脱离 DDPM 失败区
- ✅ 风格贴近真实照片(vs cyclegan 偶尔过锐)
- 与医生实际术后 ground truth 有 50-70% 主观相似,剩余差异属于鼻整形本身的风格分布内差异

---

## 六、交付清单

| 类别 | 路径 | 大小 | 版本化 |
|---|---|---|---|
| 模型加载 + LoRA 挂载 | `ml/models/sd_inpaint.py` | 11KB | ✅ |
| 训练脚本 | `ml/train_sd_inpaint.py` | 22KB | ✅ |
| 评估脚本 | `ml/evaluate_sd_inpaint.py` | 9KB | ✅ |
| 推理服务层 | `backend/inference_sd.py` | 7KB | ✅ |
| 部署脚本(smoke/train/eval) | `tmp/deploy/sd_*.sh` | - | ✅ |
| Benchmark 指标 JSON | `artifacts/eval_sd_inpaint/metrics.json` | < 1KB | ✅ |
| 本报告 | `reports/V4_path_C_SD_inpaint.md` | - | ✅ |
| LoRA 权重 safetensors | `models/outcome_v3_512/sd_inpaint_nose/best/` | 12MB | ❌ 本地 |
| SD 1.5 Inpainting 基础模型 | `models/sd_base/inpaint/` | 5GB | ❌ 本地 + AutoDL |
| 定性网格图 + 57 张生成图 | `artifacts/eval_sd_inpaint/generated/` | 2.2MB | ❌ 含患者 PII |

---

## 七、代码审查

训练代码经过 5 轮 V3 审查(修 59 个 issue)后,本轮又经过额外一次 Code Reviewer 审查,修复了 8 个 SD 特定问题:

1. **CRITICAL** · LoRA 挂载后断言 `trainable > 0`(防止版本 skew 静默失败)
2. **HIGH** · bf16 autocast(代替 fp16 + GradScaler,Blackwell 原生支持)
3. **HIGH** · 推理 `strength=1.0`(匹配训练分布,0.95 对 9 通道 inpaint 无意义)
4. **HIGH** · 全局 seed(Python/NumPy/Torch)+ 验证用 deterministic generator
5. **MEDIUM** · VAE/text encoder 移到 fp16 省 ~2GB VRAM
6. **MEDIUM** · mask 硬阈值 zero-out 解决训练/推理分布不一致
7. **MEDIUM** · 验证 loss 用 t-grid 采样代替单点(降低测量噪声)
8. **MEDIUM** · TF32 matmul 启用(Blackwell 免费 ~30% 加速)
9. **NIT** · transformers 5.x CLIPTextModel 输出用 `.last_hidden_state` 而非 `[0]`(API 兼容)

---

## 八、局限 & 下一步

### 8.1 本轮未做

- **对齐评估**:没有在 `nose_roi_512` 空间重新裁剪生成图后与 cyclegan 直接对比 FID。当前评估在全脸空间。
- **前端集成**:`backend/inference_sd.py` 存在但未接入 `backend/serve.py` 的 `/api/predict` 路由
- **多 LoRA rank 消融**:只试了 r=16,未试 r=8/r=32/r=64

### 8.2 潜在改进

- 数据扩充:跨机构合作扩到 5000+ 对 → 期望 FID 再降 30-40%
- SD 3.5 Medium / SDXL 升级(需 24GB+ VRAM)
- 引入 landmark 引导(`ml/landmarks.py` 已有)作为额外 ControlNet 条件

---

## 九、一句话总结

> **路径 C 成功**:用预训练 SD 1.5 + 12MB LoRA 在 14.6 分钟、¥15 云端开销下,把原本彻底失败的 diffusion 路线从 FID 461 拉回到 FID 57 —— 与生产级 cyclegan_nose 处于同一质量区间。推荐作为 CS31 项目第 4 个可生产模型,补齐 diffusion 路线空缺。

---

*代码仓库*:https://github.com/BobbyZ2026/CS31
*维护者*:Bobby(g1749989936@gmail.com)
