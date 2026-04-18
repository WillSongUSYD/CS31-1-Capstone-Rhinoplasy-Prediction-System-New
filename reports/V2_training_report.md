# V2 Nose-only Training Report

## 数据

| 项 | V1(旧) | **V2(本次)** |
|---|---|---|
| 训练样本数 | 582 | **572**(-10) |
| 正脸(已排除) | 251 | 252 |
| 面部朝向 | 部分左朝向 | **全部统一右朝向**(翻转 11 张) |
| 坏样本 | 9 张混入 | **全部剔除** |
| Mask 算法 | MediaPipe 478 点(漏检 40%) | **InsightFace 5 点 + 鼻轴倾斜椭圆**(漏检 <1%) |
| Nose ROI 裁剪 | MediaPipe bbox | **基于新 mask bbox 重裁** |

splits.csv: train 458 / val 57 / test 57

## 训练配置

| 超参 | 值 |
|---|---|
| epochs | 50 |
| batch_size | 8 |
| image_size | 128 × 128 |
| optimizer | Adam |
| lr | 2e-4 |
| β (GAN) | (0.5, 0.999) |
| β (diffusion) | (0.9, 0.999) |
| device | MPS |
| 数据增强 | hflip + ColorJitter |

## 训练结果(validation L1)

| 模型 | Best val L1 | Last val L1 | 收敛 |
|---|---|---|---|
| `autoencoder_nose` | **0.0760** | 0.0875 | ✅ |
| `pix2pix_nose` | 0.0771 | 0.0783 | ✅ |
| `cyclegan_nose` | 0.0763 | 0.0855 | ✅ |
| `diffusion_nose` | 0.5479 | 0.6286 | ⚠️ 未收敛(57 样本 × 50 epochs 对 DDPM 不够) |

## 测试集评估(test split,57 对)

| 模型 | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|
| **`autoencoder_nose`** | **0.7383** | 0.2536 | 140.84 |
| `pix2pix_nose` | 0.6942 | **0.1917** | 180.79 |
| `cyclegan_nose` | 0.6424 | 0.1935 | 189.14 |
| `diffusion_nose` | 0.1818 | 0.7349 | 406.76 |

**结论**:
- **autoencoder_nose 最佳** — SSIM 和 FID 均领先,生成的鼻部最接近真实术后
- **pix2pix_nose** LPIPS 感知相似度最好,细节保真度略胜
- **cyclegan_nose** 与 pix2pix 相近,但 SSIM 略低
- **diffusion_nose** 未收敛,产出噪声,后续若要用需加到 200+ epochs 或改为预训练模型 fine-tune

## 定性效果

### autoencoder_nose

![autoencoder_nose qualitative grid](../artifacts/eval/autoencoder_nose/qualitative_grid.png)

生成的鼻部结构清晰、纹理自然,和真实术后高度相似。

### pix2pix_nose

![pix2pix_nose](../artifacts/eval/pix2pix_nose/qualitative_grid.png)

视觉上与 autoencoder 相近,判别器损失让边缘更锐利。

### cyclegan_nose

![cyclegan_nose](../artifacts/eval/cyclegan_nose/qualitative_grid.png)

循环一致性保留了身份特征,但在 57 样本规模下优势不明显。

### diffusion_nose

![diffusion_nose](../artifacts/eval/diffusion_nose/qualitative_grid.png)

产出仍为噪声,未学到像素分布。

## 交付物

- 训练好的 checkpoint:`/Applications/CS31/models/outcome/{autoencoder,pix2pix,cyclegan,diffusion}_nose/best.pt`
- V1 备份:`/Applications/CS31/models/outcome_v1_backup/`
- 评测:`/Applications/CS31/artifacts/eval/benchmark.csv`
- V1 评测对照:`/Applications/CS31/artifacts/eval/benchmark_v1.csv`
- 推理代码(已更新):`ml/nose_roi.py` → InsightFace + 倾斜椭圆
