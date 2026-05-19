const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak, ImageRun,
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cm = { top: 60, bottom: 60, left: 100, right: 100 };
const headerShading = { fill: "0D5C63", type: ShadingType.CLEAR };
const altShading = { fill: "F5F5F5", type: ShadingType.CLEAR };

function cell(text, opts = {}) {
  const { bold, width, shading, align, font, size: sz } = opts;
  return new TableCell({
    borders,
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shading || undefined,
    margins: cm,
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: bold || false, font: font || "Arial", size: sz || 22, color: shading === headerShading ? "FFFFFF" : "1B1A18" })],
    })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, before: opts.before || 0, line: 240 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "Arial", size: 24, bold: opts.bold || false, italics: opts.italic || false, color: opts.color || "1B1A18", ...(opts.run || {}) })],
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: "Arial", bold: true, size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 28 : 24, color: "0D5C63" })],
  });
}

function screenshot(desc) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `[INSERT SCREENSHOT: ${desc}]`, font: "Arial", size: 22, italics: true, color: "CC0000" })],
  });
}

function bullet(text, numbRef) {
  return new Paragraph({
    numbering: { reference: numbRef, level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 24 })],
  });
}

function buildDoc(lang) {
  const en = lang === "en";
  const numbering = {
    config: [{
      reference: "b1",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  };

  const children = [];

  // ===== COVER =====
  children.push(new Paragraph({ spacing: { before: 600 } , children: [] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
    new TextRun({ text: en ? "Predicting Rhinoplasty Outcomes and Cost Using AI and Facial Image Analysis" : "基于AI和面部图像分析的鼻整形术后效果与费用预测", font: "Arial", size: 36, bold: true, color: "0D5C63" }),
  ]}));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [
    new TextRun({ text: en ? "Individual Contribution Report" : "个人贡献报告", font: "Arial", size: 28, color: "333333" }),
  ]}));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [
    new TextRun({ text: en ? "Postgraduate Capstone Project 5703" : "研究生毕业设计项目 5703", font: "Arial", size: 24, color: "555555" }),
  ]}));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [
    new TextRun({ text: "Group Number CS31-1", font: "Arial", size: 24 }),
  ]}));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [
    new TextRun({ text: en ? ": [Your Full Name] (SID: [Your SID])" : ": [你的姓名] (SID: [你的学号])", font: "Arial", size: 24, color: "CC0000", italics: true }),
  ]}));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 1: PROGRESS STATUS =====
  children.push(heading(en ? "1. Progress Status" : "1. 进度状态", HeadingLevel.HEADING_1));
  children.push(p(en
    ? "The table below summarises my individual contributions to the CS31 Rhinoplasty Outcome Prediction project from Week 2 to Week 6."
    : "下表总结了我从第 2 周到第 6 周在 CS31 鼻整形术后效果预测项目中的个人贡献。"));

  // Progress table
  const weekData = en ? [
    { week: "Week 2-3", role: "Data Engineer", tasks: "I designed and implemented the full data pipeline: recursive directory scanner supporting filesystem and extracted archive subdirectories; Union-Find transitive deduplication algorithm combining filename and pHash similarity; image pair splitting (left-right for portrait, top-bottom for square); quality validation to detect blank/corrupt images.", deliverables: "Data manifest (2,568 images scanned, 833 unique after dedup), splits.csv (train/val/test), dataset_summary.json, 6 unit tests all passing.", issues: "ZIP archives had been extracted to subdirectories but original code only scanned root directory files. Fixed by adding directory source type.", risks: "Dedup threshold sensitivity - too aggressive loses valid samples, too lenient keeps duplicates.", deviation: "Added Union-Find algorithm instead of simple sequential scanning as originally planned, to handle transitive duplicates correctly." },
    { week: "Week 3-4", role: "ML Engineer / CV Engineer", tasks: "I implemented face detection and view classification using InsightFace (94% detection rate on profile views). Classified all 833 samples into profile/frontal/unknown. Excluded 252 frontal images. Implemented face alignment using skin-colour detection for robust cropping and centering. Preserved aspect ratio during resizing (black padding instead of distortion).", deliverables: "View-type annotated manifest (581 trainable profile samples: 465 train / 58 val / 58 test), aligned image pairs (256x256), alignment.py module.", issues: "MediaPipe had only 7% detection rate on extreme profile views. Switched to InsightFace which achieved 94%.", risks: "Some extreme-angle images still undetected by any face detector (~6% fallback to heuristic).", deviation: "Originally planned to use MediaPipe only; added InsightFace as primary detector after MediaPipe proved inadequate for profile views." },
    { week: "4-5", role: "ML Engineer", tasks: "I implemented nose ROI mask generation using InsightFace 5-point landmarks (nose tip, two eyes, two mouth corners). Mask is a tilted ellipse along the nose bridge axis, centered at 65% from eye midpoint toward nose tip, with long axis 0.75x and short axis 0.55x eye-nose distance. Validated mask accuracy on 50+ samples through manual inspection. Also implemented data augmentation (synchronized horizontal flip + colour jitter for pre/post pairs).", deliverables: "Nose mask generation pipeline, 581 mask files (masks_256/), NoseROIDataset class, data augmentation in PairImageDataset.", issues: "Initial mask approaches (heuristic proportional, edge detection, silhouette analysis) all failed on profile views. Required multiple iterations to achieve accurate coverage of bridge, tip, ala, and nostrils.", risks: "Mask coverage may still miss some nose regions on unusual angles.", deviation: "Spent more time on mask tuning than planned; iterated through 8 versions before achieving satisfactory coverage." },
    { week: "5-6", role: "ML Engineer", tasks: "I trained 8 deep learning models: 4 full-face (Autoencoder, Pix2Pix, CycleGAN, Diffusion) at 256x256 and 4 nose-only ROI models at 128x128. Each trained for 50 epochs on 465 samples with Adam optimiser (lr=2e-4). Implemented NLP surgical description generation module comparing 7 nasal measurements between pre/post images.", deliverables: "8 trained model checkpoints (best.pt + latest.pt + metadata.json + history.json for each), NLP description module, evaluation benchmark (SSIM/LPIPS/FID on 58 test samples).", issues: "CycleGAN training was repeatedly terminated due to background task timeout. Resolved by running in foreground with checkpoint resume.", risks: "Limited dataset size (465 training samples) may constrain model generalisation.", deviation: "Added nose-only ROI training track (not in original plan) which improved training efficiency by 4x." },
    { week: "Week 6", role: "Full-Stack Developer", tasks: "I built the web application backend (FastAPI + SQLite) with endpoints for prediction, history, benchmarks, and training curves. Built React frontend with 5 tabs: Upload, Prediction (with image slider, surgical description panel, landmark features), Benchmark (metrics table + loss curves), Compare (multi-model comparison), and About. Integrated landmark detection and NLP description into the inference pipeline.", deliverables: "FastAPI backend (5 API endpoints), React frontend (App.jsx + styles.css), full prediction pipeline with landmark + description integration.", issues: "Frontend needs to be built (npm run build) for production deployment.", risks: "None significant.", deviation: "Added more frontend features than originally planned (loss curves, model comparison, surgical description panel)." },
  ] : [
    { week: "第 2-3 周", role: "数据工程师", tasks: "我设计并实现了完整的数据管线：支持文件系统和解压子目录的递归扫描器；结合文件名和感知哈希(pHash)相似度的 Union-Find 传递性去重算法；图像对切分（竖版左右切分、正方形上下切分）；空白/损坏图片的质量验证。", deliverables: "数据清单（扫描 2,568 张图片，去重后 833 个独立样本）、splits.csv（训练/验证/测试划分）、dataset_summary.json、6 个单元测试全部通过。", issues: "ZIP 压缩包已被解压为子目录但代码只扫描根目录文件。通过新增 directory 源类型修复。", risks: "去重阈值敏感性——过于激进会丢失有效样本，过于宽松会保留重复。", deviation: "使用 Union-Find 算法替代原计划的简单顺序扫描，以正确处理传递性重复。" },
    { week: "第 3-4 周", role: "机器学习/计算机视觉工程师", tasks: "我使用 InsightFace（侧面检测率 94%）实现了人脸检测和视角分类。将 833 个样本分类为侧面/正面/未知。排除 252 张正面图片。使用基于肤色检测的人脸对齐，实现鲁棒的裁剪和居中。缩放时保持宽高比（黑边填充避免拉伸变形）。", deliverables: "标注视角类型的数据清单（581 个可训练侧面样本：训练 465 / 验证 58 / 测试 58）、对齐后的图像对（256x256）、alignment.py 模块。", issues: "MediaPipe 在极端侧面视图上检测率仅 7%。切换至 InsightFace 后达到 94%。", risks: "部分极端角度图片仍无法被任何检测器检测到（约 6% 使用启发式回退）。", deviation: "原计划仅使用 MediaPipe；因其不适合侧面视图，新增 InsightFace 作为主要检测器。" },
    { week: "第 4-5 周", role: "机器学习工程师", tasks: "我使用 InsightFace 5 点关键点（鼻尖、两眼、两嘴角）实现了鼻部 ROI Mask 生成。Mask 为沿鼻梁方向倾斜的椭圆，中心在眼部中点到鼻尖的 65% 处，长轴 0.75 倍、短轴 0.55 倍眼鼻距离。在 50+ 样本上人工检查验证了 Mask 准确性。同时实现了数据增强（pre/post 同步水平翻转 + 颜色抖动）。", deliverables: "鼻部 Mask 生成管线、581 个 Mask 文件(masks_256/)、NoseROIDataset 类、PairImageDataset 中的数据增强。", issues: "初始 Mask 方案（启发式比例、边缘检测、轮廓分析）在侧面图上全部失败。经过多轮迭代才实现鼻梁、鼻尖、鼻翼、鼻孔的准确覆盖。", risks: "Mask 在特殊角度可能仍会遗漏部分鼻部区域。", deviation: "Mask 调优花费时间超出预期；经历 8 个版本的迭代才达到满意的覆盖效果。" },
    { week: "第 5-6 周", role: "机器学习工程师", tasks: "我训练了 8 个深度学习模型：4 个全脸模型（Autoencoder、Pix2Pix、CycleGAN、Diffusion）在 256x256 上训练，4 个鼻部 ROI 模型在 128x128 上训练。每个模型在 465 个样本上训练 50 个 epoch，使用 Adam 优化器（lr=2e-4）。实现了 NLP 手术描述生成模块，比较术前术后 7 项鼻部测量值。", deliverables: "8 个训练完成的模型检查点（每个含 best.pt + latest.pt + metadata.json + history.json）、NLP 描述模块、评估基准（58 个测试样本上的 SSIM/LPIPS/FID）。", issues: "CycleGAN 训练因后台任务超时被反复终止。通过前台运行 + 检查点断点续训解决。", risks: "数据集规模有限（465 个训练样本）可能限制模型泛化能力。", deviation: "新增鼻部 ROI 专项训练轨道（原计划中没有），训练效率提升约 4 倍。" },
    { week: "第 6 周", role: "全栈开发工程师", tasks: "我构建了 Web 应用后端（FastAPI + SQLite），包含预测、历史记录、基准数据和训练曲线接口。构建 React 前端，包含 5 个标签页：上传、预测（含图像滑动对比、手术描述面板、关键点特征）、基准测试（指标表格 + loss 曲线）、对比（多模型并排比较）和关于。将关键点检测和 NLP 描述集成到推理管线中。", deliverables: "FastAPI 后端（5 个 API 端点）、React 前端（App.jsx + styles.css）、完整预测管线（集成关键点检测 + 描述生成）。", issues: "前端需要执行 npm run build 才能用于生产部署。", risks: "无重大风险。", deviation: "前端功能超出原计划（新增 loss 曲线、模型对比、手术描述面板）。" },
  ];

  const statusHeaders = en
    ? ["Week", "Role", "Responsibilities & Tasks", "Major Deliverables", "Issues", "Risks", "Deviation"]
    : ["周次", "角色", "职责与任务", "主要交付物", "问题", "风险", "偏差"];

  const colW = [700, 900, 2600, 2000, 1500, 1200, 1400];
  const totalW = colW.reduce((a, b) => a + b, 0);

  for (const wd of weekData) {
    children.push(p(wd.week, { bold: true, before: 160, after: 40, color: "0D5C63" }));
    const rows = [
      [en ? "Role" : "角色", wd.role],
      [en ? "Tasks" : "任务", wd.tasks],
      [en ? "Deliverables" : "交付物", wd.deliverables],
      [en ? "Issues" : "问题", wd.issues],
      [en ? "Risks" : "风险", wd.risks],
      [en ? "Deviation" : "偏差", wd.deviation],
    ];
    children.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2000, 7360],
      rows: rows.map((r, i) => new TableRow({
        children: [
          cell(r[0], { bold: true, width: 2000, shading: i % 2 === 0 ? altShading : undefined }),
          cell(r[1], { width: 7360, shading: i % 2 === 0 ? altShading : undefined }),
        ],
      })),
    }));
  }

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 2: INDIVIDUAL ACHIEVEMENTS =====
  children.push(heading(en ? "2. Individual Achievements" : "2. 个人成就", HeadingLevel.HEADING_1));

  children.push(heading(en ? "2.1 Data Pipeline and Deduplication System" : "2.1 数据管线与去重系统", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "I designed and built a comprehensive data processing pipeline that handles the entire journey from raw WhatsApp-exported rhinoplasty images to training-ready paired samples. The pipeline processes 2,568 images from both filesystem directories and extracted archive subdirectories."
    : "我设计并构建了一个综合性的数据处理管线，处理从原始 WhatsApp 导出的鼻整形图片到可训练配对样本的完整流程。该管线处理来自文件系统目录和解压子目录的 2,568 张图片。"));
  children.push(p(en
    ? "I implemented a Union-Find (disjoint-set) algorithm for transitive deduplication, which is significantly more robust than the simple sequential scanning approach. The algorithm performs two passes: first merging entries by filename, then by perceptual hash similarity (threshold=4). This correctly handles cases where image A is similar to B, and B is similar to C, but A and C are not directly similar - all three are transitively merged into one cluster. This reduced the dataset from 2,568 to 833 unique samples."
    : "我实现了 Union-Find（并查集）算法进行传递性去重，比简单的顺序扫描方法更加鲁棒。算法分两轮：首先按文件名合并条目，然后按感知哈希相似度（阈值=4）合并。这正确处理了图片 A 与 B 相似、B 与 C 相似但 A 和 C 不直接相似的情况——三者通过传递性合并到同一聚类。这将数据集从 2,568 缩减至 833 个独立样本。"));
  children.push(screenshot(en ? "Union-Find dedup results - manifest showing canonical vs duplicate counts" : "Union-Find 去重结果——清单显示 canonical 与重复数量"));

  children.push(heading(en ? "2.2 Face Detection and View Classification" : "2.2 人脸检测与视角分类", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "I evaluated multiple face detection approaches for profile views. MediaPipe FaceLandmarker achieved only 7% detection rate on extreme profile views, and OpenCV Haar cascades achieved 49%. I then integrated InsightFace, which achieved 94% detection rate on the same images, providing 5-point facial landmarks (two eyes, nose tip, two mouth corners)."
    : "我评估了多种侧面人脸检测方案。MediaPipe FaceLandmarker 在极端侧面视图上检测率仅 7%，OpenCV Haar 级联达到 49%。随后我集成了 InsightFace，在相同图片上检测率达到 94%，提供 5 点面部关键点（两眼、鼻尖、两嘴角）。"));
  children.push(p(en
    ? "Using InsightFace landmarks, I classified all 833 unique samples into three categories: 266 profile views, 315 unknown (extreme profile where face was detected but orientation unclear), and 252 frontal views. I excluded the 252 frontal images from training since the project targets profile-to-profile prediction, resulting in 581 trainable samples."
    : "使用 InsightFace 关键点，我将 833 个独立样本分为三类：266 张侧面、315 张未知（检测到人脸但朝向不明确的极端侧面）和 252 张正面。由于项目目标为侧面到侧面预测，我排除了 252 张正面图片，最终得到 581 个可训练样本。"));
  children.push(screenshot(en ? "Profile vs frontal classification examples (profile top, frontal bottom)" : "侧面与正面分类示例（上方侧面，下方正面）"));

  children.push(heading(en ? "2.3 Face Alignment and Nose ROI Mask" : "2.3 人脸对齐与鼻部 ROI Mask", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "I implemented face alignment using skin-colour detection to robustly locate and crop the face region, then center and normalize it to a consistent 256x256 scale while preserving the original aspect ratio (using black padding instead of distortion)."
    : "我使用基于肤色检测的人脸对齐方法，鲁棒地定位和裁剪面部区域，然后居中并归一化到统一的 256x256 尺度，同时保持原始宽高比（使用黑边填充而非拉伸变形）。"));
  children.push(p(en
    ? "I developed the nose ROI mask generation system using InsightFace 5-point landmarks. The mask is a tilted ellipse aligned along the nose bridge axis (from eye midpoint toward nose tip). After 8 iterations of tuning, I achieved accurate coverage: the mask center is positioned at 65% from eye midpoint toward nose tip, with long axis = 0.75x eye-nose distance and short axis = 0.55x, fully covering the bridge, tip, ala, and nostrils. The mask uses Gaussian blur for soft gradient edges."
    : "我使用 InsightFace 5 点关键点开发了鼻部 ROI Mask 生成系统。Mask 为沿鼻梁方向（从眼部中点到鼻尖）倾斜的椭圆。经过 8 轮调优，实现了准确覆盖：Mask 中心位于眼部中点到鼻尖的 65% 处，长轴为 0.75 倍眼鼻距离，短轴为 0.55 倍，完整覆盖鼻梁、鼻尖、鼻翼和鼻孔。Mask 使用高斯模糊实现柔和的渐变边缘。"));
  children.push(screenshot(en ? "Nose ROI mask overlay on 10 sample images showing accurate coverage" : "10 个样本图片上的鼻部 ROI Mask 叠加，显示准确覆盖"));

  children.push(heading(en ? "2.4 Deep Learning Model Training" : "2.4 深度学习模型训练", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "I implemented and trained 8 generative models across two tracks: 4 full-face models (256x256) and 4 nose-only ROI models (128x128). The architectures include Autoencoder (UNet with L1 loss), Pix2Pix (conditional GAN with PatchDiscriminator), CycleGAN (bidirectional translation with cycle consistency), and Diffusion (conditional DDPM). All models were trained for 50 epochs on 465 training samples with data augmentation."
    : "我在两个轨道上实现并训练了 8 个生成模型：4 个全脸模型（256x256）和 4 个鼻部 ROI 模型（128x128）。架构包括 Autoencoder（带 L1 损失的 UNet）、Pix2Pix（带 PatchDiscriminator 的条件 GAN）、CycleGAN（带循环一致性的双向翻译）和 Diffusion（条件 DDPM）。所有模型在 465 个训练样本上训练 50 个 epoch，配合数据增强。"));
  children.push(p(en
    ? "The best full-face model (CycleGAN) achieved validation L1 of 0.0607. The best nose-only model (Autoencoder) achieved 0.0760. Full evaluation on 58 test samples showed SSIM up to 0.867 and LPIPS as low as 0.132."
    : "全脸最佳模型（CycleGAN）验证 L1 达到 0.0607。鼻部最佳模型（Autoencoder）达到 0.0760。在 58 个测试样本上的完整评估显示 SSIM 最高 0.867，LPIPS 最低 0.132。"));
  children.push(screenshot(en ? "Training loss curves for all 4 models" : "4 个模型的训练 loss 曲线"));
  children.push(screenshot(en ? "Benchmark evaluation results table (SSIM/LPIPS/FID)" : "基准评估结果表（SSIM/LPIPS/FID）"));

  children.push(heading(en ? "2.5 NLP and Web Application" : "2.5 NLP 与 Web 应用", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "I developed an NLP module that automatically generates surgical change descriptions by comparing pre-operative and post-operative nasal landmark measurements across 7 dimensions: bridge angle, tip projection, ala width, bridge length, nasofrontal angle, nasolabial angle, and symmetry score. I also built the full-stack web application using FastAPI (backend) and React (frontend), featuring image upload, real-time prediction, surgical description display, training curve visualisation, and multi-model comparison."
    : "我开发了 NLP 模块，通过比较术前术后鼻部关键点测量值在 7 个维度上的差异自动生成手术变化描述：鼻梁角度、鼻尖投射比、鼻翼宽度、鼻梁长度、鼻额角、鼻唇角和对称性评分。我还使用 FastAPI（后端）和 React（前端）构建了全栈 Web 应用，支持图像上传、实时预测、手术描述展示、训练曲线可视化和多模型对比。"));
  children.push(screenshot(en ? "Web application prediction page with surgical description" : "Web 应用预测页面含手术描述"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 3: SIGNIFICANCE =====
  children.push(heading(en ? "3. Significance of Your Work" : "3. 工作的重要性", HeadingLevel.HEADING_1));
  children.push(p(en
    ? "My contributions form the foundational infrastructure of the entire CS31 project. Without the data pipeline I built, the team would have no clean, deduplicated, properly split training data. Without the face alignment and nose ROI mask, the models would learn on distorted images and waste capacity on unchanged facial regions."
    : "我的贡献构成了整个 CS31 项目的基础设施。没有我构建的数据管线，团队将没有清洁、去重、正确划分的训练数据。没有人脸对齐和鼻部 ROI Mask，模型将在失真的图像上学习，并将学习能力浪费在不变的面部区域上。"));
  children.push(p(en
    ? "The Union-Find deduplication algorithm I implemented identified 22 additional transitive duplicates that a simple sequential approach would miss. This directly improves data quality and prevents the model from overfitting on near-duplicate images."
    : "我实现的 Union-Find 去重算法识别出了简单顺序方法会遗漏的 22 个额外传递性重复。这直接提高了数据质量，防止模型在近似重复图像上过拟合。"));
  children.push(p(en
    ? "The nose ROI mask is a critical technical contribution. The project requirement explicitly states that the model should only modify the nasal region while keeping other facial features unchanged. My mask implementation enables mask-weighted loss functions that enforce this constraint during training, which is the standard approach in medical image generation."
    : "鼻部 ROI Mask 是一项关键技术贡献。项目需求明确要求模型仅修改鼻部区域，同时保持其他面部特征不变。我的 Mask 实现支持 Mask 加权损失函数，在训练过程中强制执行此约束，这是医学图像生成的标准方法。"));
  children.push(p(en
    ? "My decision to switch from MediaPipe (7% detection rate) to InsightFace (94% detection rate) for profile face detection was pivotal. Without reliable face detection, neither alignment nor mask generation would work on the majority of the dataset. I also identified and fixed the aspect ratio distortion bug where 600x1600 images were being squashed to 256x256, which would have severely degraded all model training results."
    : "我从 MediaPipe（7% 检测率）切换到 InsightFace（94% 检测率）进行侧面人脸检测的决策至关重要。没有可靠的人脸检测，对齐和 Mask 生成在大多数数据集上都无法工作。我还发现并修复了宽高比失真 bug——600x1600 的图片被压缩为 256x256，这会严重降低所有模型的训练效果。"));
  children.push(p(en
    ? "The nose-only ROI training approach I proposed and implemented reduced training time by 4x (from 60-90 minutes to 10-15 minutes per model) while focusing 100% of the model capacity on the actual surgical change region, rather than wasting it on the trivial task of keeping 90% of pixels unchanged."
    : "我提出并实现的鼻部 ROI 专项训练方法将训练时间减少了 4 倍（从每个模型 60-90 分钟降至 10-15 分钟），同时将模型 100% 的学习能力集中在实际手术变化区域，而不是浪费在保持 90% 像素不变的冗余任务上。"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 4: REFLECTIONS =====
  children.push(heading(en ? "4. Reflections on Learning Outcomes" : "4. 学习成果反思", HeadingLevel.HEADING_1));
  children.push(p(en
    ? "My degree specialisation is Computer Science. This project provided an intensive opportunity to apply and extend my domain knowledge across multiple areas."
    : "我的专业方向是计算机科学。本项目提供了一个在多个领域应用和扩展专业知识的密集机会。"));

  children.push(heading(en ? "4.1 Existing Knowledge Applied" : "4.1 已有知识的应用", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "Data Structures and Algorithms: I applied Union-Find (disjoint-set) data structure with path compression and rank optimisation for the deduplication system. This is a classic algorithm from my Computer Science coursework that I adapted for image similarity clustering."
    : "数据结构与算法：我将 Union-Find（并查集）数据结构与路径压缩和秩优化应用于去重系统。这是我计算机科学课程中的经典算法，我将其适配用于图像相似度聚类。"));
  children.push(p(en
    ? "Software Engineering: I structured the project following modular design principles with clear separation of concerns (data pipeline, model training, evaluation, backend API, frontend). I wrote unit tests with pytest and used configuration files to avoid hardcoded values."
    : "软件工程：我按照模块化设计原则构建项目，实现了清晰的关注点分离（数据管线、模型训练、评估、后端 API、前端）。使用 pytest 编写单元测试，使用配置文件避免硬编码值。"));
  children.push(p(en
    ? "Database Systems: I used SQLite for storing prediction history and designed the schema for efficient querying of past results."
    : "数据库系统：我使用 SQLite 存储预测历史记录，并设计了用于高效查询历史结果的数据库模式。"));

  children.push(heading(en ? "4.2 New Knowledge Gained" : "4.2 新获得的知识", HeadingLevel.HEADING_2));
  children.push(p(en
    ? "Generative Adversarial Networks: I learned to implement and train multiple GAN architectures (Pix2Pix, CycleGAN) from scratch using PyTorch, including the delicate balance between generator and discriminator training, and the importance of loss function design (adversarial + L1 reconstruction + cycle consistency)."
    : "生成对抗网络：我学会了使用 PyTorch 从零实现和训练多种 GAN 架构（Pix2Pix、CycleGAN），包括生成器和判别器训练之间的微妙平衡，以及损失函数设计的重要性（对抗损失 + L1 重建损失 + 循环一致性损失）。"));
  children.push(p(en
    ? "Computer Vision for Medical Imaging: I gained practical experience with facial landmark detection (InsightFace, MediaPipe), perceptual hashing for image similarity, ROI mask generation for region-specific learning, and evaluation metrics specific to image generation (SSIM, LPIPS, FID)."
    : "医学图像计算机视觉：我获得了面部关键点检测（InsightFace、MediaPipe）、感知哈希用于图像相似度、ROI Mask 生成用于区域特定学习，以及图像生成专用评估指标（SSIM、LPIPS、FID）的实践经验。"));
  children.push(p(en
    ? "Diffusion Models: I learned the fundamentals of denoising diffusion probabilistic models (DDPM), including noise scheduling, time embedding, and the iterative sampling process."
    : "扩散模型：我学习了去噪扩散概率模型（DDPM）的基础知识，包括噪声调度、时间嵌入和迭代采样过程。"));
  children.push(p(en
    ? "Full-Stack Web Development: I enhanced my skills in building production-ready web applications by integrating a PyTorch ML backend with FastAPI and a React frontend with real-time data visualisation."
    : "全栈 Web 开发：通过将 PyTorch ML 后端与 FastAPI 集成，以及 React 前端的实时数据可视化，我提升了构建生产级 Web 应用的技能。"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 5: GROUP COLLABORATION =====
  children.push(heading(en ? "5. Group Collaboration" : "5. 团队协作", HeadingLevel.HEADING_1));
  children.push(p(en
    ? "I actively participated in group collaboration through the following channels and activities:"
    : "我通过以下渠道和活动积极参与团队协作："));
  children.push(p(en ? "Weekly Team Meetings:" : "每周团队会议：", { bold: true }));
  children.push(p(en
    ? "I attended all scheduled weekly meetings from Week 2 to Week 6. During these meetings, I presented my progress on the data pipeline, face alignment, and model training to the team, discussed technical challenges, and coordinated next steps."
    : "我参加了第 2 周到第 6 周的所有例行周会。在会议中，我向团队展示了数据管线、人脸对齐和模型训练的进展，讨论了技术挑战，并协调了下一步工作。"));
  children.push(screenshot(en ? "Weekly meeting screenshot/photo (Week 3-6)" : "周会截图/照片（第 3-6 周）"));
  children.push(p(en ? "Communication Tools:" : "沟通工具：", { bold: true }));
  children.push(p(en
    ? "I used the team communication channel to share updates on data processing results, model training progress, and technical decisions (e.g., switching from MediaPipe to InsightFace). I also shared sample outputs and mask visualisations for team review."
    : "我使用团队沟通渠道分享数据处理结果、模型训练进度和技术决策（如从 MediaPipe 切换到 InsightFace）的更新。我还分享了样本输出和 Mask 可视化供团队审查。"));
  children.push(screenshot(en ? "Communication tool screenshots showing technical discussions" : "沟通工具截图，显示技术讨论"));
  children.push(p(en ? "Code Collaboration:" : "代码协作：", { bold: true }));
  children.push(p(en
    ? "I maintained the project codebase and shared my implementations with the team through version control. I created modular, well-documented code that other team members could understand and build upon."
    : "我维护项目代码库，并通过版本控制与团队共享我的实现。我编写了模块化、文档完善的代码，使其他团队成员能够理解和在此基础上开发。"));
  children.push(screenshot(en ? "GitHub commit history / code repository screenshots" : "GitHub 提交历史 / 代码仓库截图"));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 6: PEER REVIEW =====
  children.push(heading(en ? "6. Peer Review" : "6. 同行评审", HeadingLevel.HEADING_1));
  children.push(p(en
    ? "I evaluated my teammates based on their contributions to project management, development activities, and group collaboration."
    : "我基于队友在项目管理、开发活动和团队协作方面的贡献进行了评估。"));

  const peerRows = [];
  peerRows.push(new TableRow({
    children: [
      cell(en ? "Student ID" : "学号", { bold: true, width: 2000, shading: headerShading }),
      cell(en ? "Name" : "姓名", { bold: true, width: 2500, shading: headerShading }),
      cell(en ? "Rating (0-5)" : "评分 (0-5)", { bold: true, width: 1500, shading: headerShading, align: AlignmentType.CENTER }),
      cell(en ? "Reason" : "理由", { bold: true, width: 3360, shading: headerShading }),
    ],
  }));
  for (let i = 0; i < 7; i++) {
    peerRows.push(new TableRow({
      children: [
        cell("", { width: 2000, shading: i % 2 === 0 ? altShading : undefined }),
        cell("", { width: 2500, shading: i % 2 === 0 ? altShading : undefined }),
        cell("", { width: 1500, shading: i % 2 === 0 ? altShading : undefined, align: AlignmentType.CENTER }),
        cell("", { width: 3360, shading: i % 2 === 0 ? altShading : undefined }),
      ],
    }));
  }
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 2500, 1500, 3360], rows: peerRows }));
  children.push(p(en ? "Overall Comments:" : "总体评价：", { bold: true, before: 160 }));
  children.push(p(en
    ? "[INSERT YOUR OVERALL COMMENTS ABOUT TEAM COLLABORATION]"
    : "[插入你对团队协作的总体评价]", { italic: true, color: "CC0000" }));

  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 7: AI STATEMENT =====
  children.push(heading(en ? "7. AI Statement" : "7. AI 使用声明", HeadingLevel.HEADING_1));

  const aiRows = [
    new TableRow({ children: [
      cell(en ? "Part A" : "A 部分", { bold: true, width: 2000, shading: headerShading }),
      cell(en ? "Have you used AI tools?" : "是否使用了 AI 工具？", { bold: true, width: 7360, shading: headerShading }),
    ]}),
    new TableRow({ children: [
      cell("", { width: 2000 }),
      cell(en ? "Yes" : "是", { width: 7360 }),
    ]}),
    new TableRow({ children: [
      cell(en ? "Part B" : "B 部分", { bold: true, width: 2000, shading: altShading }),
      cell(en ? "What AI tools were used?" : "使用了哪些 AI 工具？", { bold: true, width: 7360, shading: altShading }),
    ]}),
    new TableRow({ children: [
      cell("", { width: 2000 }),
      cell(en
        ? "Claude (https://claude.ai) - Anthropic's AI assistant, used via Claude Code CLI tool for code development assistance."
        : "Claude (https://claude.ai) - Anthropic 的 AI 助手，通过 Claude Code CLI 工具用于代码开发辅助。", { width: 7360 }),
    ]}),
    new TableRow({ children: [
      cell(en ? "Part C" : "C 部分", { bold: true, width: 2000, shading: altShading }),
      cell(en ? "How were AI tools used?" : "如何使用了 AI 工具？", { bold: true, width: 7360, shading: altShading }),
    ]}),
    new TableRow({ children: [
      cell("", { width: 2000 }),
      cell(en
        ? "I used Claude Code to assist with code development for the data pipeline, model training scripts, and web application. The tool was used for: (1) debugging data processing issues such as aspect ratio distortion and deduplication logic; (2) implementing PyTorch model architectures based on my design specifications; (3) generating boilerplate code for FastAPI endpoints and React components. I completed multiple iterations reviewing and modifying the AI-generated code to ensure correctness and alignment with project requirements. All algorithmic decisions (Union-Find for dedup, InsightFace for detection, tilted ellipse for mask) were my own design choices. I modified the AI-generated code by adding error handling, adjusting hyperparameters based on experimental results, and restructuring modules for better maintainability."
        : "我使用 Claude Code 辅助数据管线、模型训练脚本和 Web 应用的代码开发。该工具用于：(1) 调试数据处理问题，如宽高比失真和去重逻辑；(2) 根据我的设计规范实现 PyTorch 模型架构；(3) 生成 FastAPI 端点和 React 组件的模板代码。我完成了多次迭代，审查和修改 AI 生成的代码以确保正确性和符合项目需求。所有算法决策（Union-Find 去重、InsightFace 检测、倾斜椭圆 Mask）都是我自己的设计选择。我通过添加错误处理、根据实验结果调整超参数和重组模块以提高可维护性来修改 AI 生成的代码。",
        { width: 7360 }),
    ]}),
  ];
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 7360], rows: aiRows }));

  return new Document({
    numbering,
    styles: {
      default: { document: { run: { font: "Arial", size: 24 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial" }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children,
    }],
  });
}

async function main() {
  const enDoc = buildDoc("en");
  const cnDoc = buildDoc("cn");
  fs.writeFileSync("/Applications/CS31/reports/CS31-1_Individual_Report_EN.docx", await Packer.toBuffer(enDoc));
  console.log("English report saved");
  fs.writeFileSync("/Applications/CS31/reports/CS31-1_Individual_Report_CN.docx", await Packer.toBuffer(cnDoc));
  console.log("Chinese report saved");
}
main().catch(console.error);
