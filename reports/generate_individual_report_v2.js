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
const imgDir = "/Applications/CS31/tmp/report_images/";

function cell(text, opts = {}) {
  const { bold, width, shading, align } = opts;
  return new TableCell({
    borders, width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shading || undefined, margins: cm, verticalAlign: "center",
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: bold || false, font: "Arial", size: 22,
        color: shading === headerShading ? "FFFFFF" : "1B1A18" })] })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, before: opts.before || 0, line: 240 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "Arial", size: 24, bold: opts.bold || false,
      italics: opts.italic || false, color: opts.color || "1B1A18" })],
  });
}

function heading(text, level) {
  const sz = level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 28 : 24;
  return new Paragraph({ heading: level, spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: "Arial", bold: true, size: sz, color: "0D5C63" })] });
}

function img(filename, w, h, caption, en) {
  const items = [];
  const data = fs.readFileSync(imgDir + filename);
  items.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: filename } })] }));
  items.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, font: "Arial", size: 20, italics: true, color: "555555" })] }));
  return items;
}

function bullet(text, ref) {
  return new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 24 })] });
}

function buildDoc(lang) {
  const en = lang === "en";
  const numbering = { config: [{ reference: "b1", levels: [{ level: 0, format: LevelFormat.BULLET,
    text: "\u2022", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] };

  const c = []; // children

  // ===== COVER =====
  c.push(new Paragraph({ spacing: { before: 600 }, children: [] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: en ? "Predicting Rhinoplasty Outcomes Using AI and Facial Image Analysis"
      : "基于AI和面部图像分析的鼻整形术后效果预测", font: "Arial", size: 36, bold: true, color: "0D5C63" })] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [new TextRun({ text: en ? "Individual Contribution Report" : "个人贡献报告", font: "Arial", size: 28, color: "333333" })] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [new TextRun({ text: en ? "Postgraduate Capstone Project 5703" : "研究生毕业设计项目 5703", font: "Arial", size: 24, color: "555555" })] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "Group Number CS31-1", font: "Arial", size: 24 })] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: en ? ": [Your Full Name] (SID: [Your SID])" : ": [你的姓名] (SID: [你的学号])",
      font: "Arial", size: 24, color: "CC0000", italics: true })] }));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 1: PROGRESS STATUS =====
  c.push(heading(en ? "1. Progress Status" : "1. 进度状态", HeadingLevel.HEADING_1));
  c.push(p(en ? "The table below summarises my individual contributions to the CS31 Rhinoplasty Outcome Prediction project from Week 2 to Week 6."
    : "下表总结了我从第 2 周到第 6 周在 CS31 鼻整形术后效果预测项目中的个人贡献。"));

  const weeks = en ? [
    { week: "Week 2-3", role: "Data Engineer", tasks: "I designed and implemented the full data pipeline: recursive directory scanner for filesystem and subdirectory sources; Union-Find transitive deduplication combining filename and pHash similarity; image pair splitting (left-right for portrait, top-bottom for square); quality validation for blank/corrupt images.",
      deliverables: "Data manifest (2,568 scanned, 833 unique), splits.csv, dataset_summary.json, 6 unit tests passing.",
      issues: "Archives extracted to subdirectories but code only scanned root. Fixed by adding directory source type.",
      risks: "Dedup threshold sensitivity.", deviation: "Added Union-Find instead of sequential scanning." },
    { week: "Week 3-4", role: "CV Engineer", tasks: "I implemented face detection using InsightFace (94% detection rate on profiles). Classified 833 samples into profile/frontal/unknown. Excluded 252 frontal images. Implemented face alignment with skin-colour detection, aspect-ratio-preserving resize.",
      deliverables: "View-type annotated manifest, 581 trainable profile samples (465/58/58 split), alignment module.",
      issues: "MediaPipe only 7% detection on profiles. Switched to InsightFace.",
      risks: "~6% still use heuristic fallback.", deviation: "Added InsightFace as primary detector." },
    { week: "Week 4-5", role: "CV/ML Engineer", tasks: "I implemented nose ROI mask generation using InsightFace 5-point landmarks. Tilted ellipse along nose bridge axis, center at 65% eye-to-nose, long axis 0.75x, short axis 0.55x. Validated on 50+ samples. Implemented data augmentation (synchronized flip + colour jitter).",
      deliverables: "Mask generation pipeline, 581 mask files, NoseROIDataset, data augmentation.",
      issues: "8 iterations of mask tuning needed. Edge detection and silhouette approaches failed.",
      risks: "Unusual angles may have imperfect mask coverage.", deviation: "Mask tuning took longer than planned." },
    { week: "Week 5-6", role: "ML Engineer", tasks: "I trained 8 deep learning models: 4 full-face (256x256) and 4 nose-only (128x128) using Autoencoder, Pix2Pix, CycleGAN, Diffusion. Each 50 epochs on 465 samples with Adam (lr=2e-4). Implemented NLP surgical description module comparing 7 nasal measurements.",
      deliverables: "8 trained models with checkpoints, NLP description module, evaluation benchmark (SSIM/LPIPS/FID on 58 test samples).",
      issues: "CycleGAN terminated by timeout. Resolved with foreground training + checkpoint resume.",
      risks: "465 training samples may limit generalisation.", deviation: "Added nose-only training track (4x faster)." },
    { week: "Week 6", role: "ML Engineer", tasks: "I ran full evaluation on all models using 58 test samples. Computed SSIM, LPIPS (AlexNet), FID (InceptionV3, dims=2048) and landmark-based ROI metrics. Generated qualitative comparison grids and benchmark CSV.",
      deliverables: "Benchmark results, qualitative grids, evaluation pipeline.",
      issues: "Aspect ratio distortion found and fixed (600x1600 squashed to 256x256).",
      risks: "None significant.", deviation: "Added aspect-ratio fix and re-ran all preprocessing." },
  ] : [
    { week: "第2-3周", role: "数据工程师", tasks: "我设计并实现了完整的数据管线：支持文件系统和子目录的递归扫描器；结合文件名和pHash相似度的Union-Find传递性去重；图像对切分（竖版左右、正方形上下）；空白/损坏图片质量验证。",
      deliverables: "数据清单（扫描2,568张，去重后833个）、splits.csv、dataset_summary.json、6个单元测试通过。",
      issues: "压缩包已解压为子目录但代码只扫描根目录。新增directory源类型修复。",
      risks: "去重阈值敏感性。", deviation: "使用Union-Find替代顺序扫描。" },
    { week: "第3-4周", role: "计算机视觉工程师", tasks: "我使用InsightFace（侧面检测率94%）实现人脸检测。将833个样本分类为侧面/正面/未知。排除252张正面图片。实现基于肤色检测的人脸对齐，保持宽高比缩放。",
      deliverables: "标注视角类型的数据清单、581个可训练侧面样本（465/58/58划分）、对齐模块。",
      issues: "MediaPipe侧面检测率仅7%。切换至InsightFace。",
      risks: "约6%仍使用启发式回退。", deviation: "新增InsightFace作为主要检测器。" },
    { week: "第4-5周", role: "CV/ML工程师", tasks: "我使用InsightFace 5点关键点实现鼻部ROI Mask生成。沿鼻梁方向倾斜椭圆，中心在眼鼻65%处，长轴0.75倍，短轴0.55倍。50+样本人工验证。实现数据增强（同步翻转+颜色抖动）。",
      deliverables: "Mask生成管线、581个Mask文件、NoseROIDataset、数据增强。",
      issues: "Mask经历8轮迭代调优。边缘检测和轮廓方法均失败。",
      risks: "特殊角度可能覆盖不完美。", deviation: "Mask调优时间超出预期。" },
    { week: "第5-6周", role: "机器学习工程师", tasks: "我训练了8个深度学习模型：4个全脸(256x256)和4个鼻部(128x128)，使用Autoencoder、Pix2Pix、CycleGAN、Diffusion。每个50 epochs，465样本，Adam(lr=2e-4)。实现NLP手术描述模块，比较7项鼻部测量值。",
      deliverables: "8个训练完成的模型检查点、NLP描述模块、评估基准（58测试样本SSIM/LPIPS/FID）。",
      issues: "CycleGAN因超时被终止。前台运行+检查点续训解决。",
      risks: "465个训练样本可能限制泛化能力。", deviation: "新增鼻部ROI训练轨道（快4倍）。" },
    { week: "第6周", role: "机器学习工程师", tasks: "我在58个测试样本上运行完整评估。计算SSIM、LPIPS(AlexNet)、FID(InceptionV3, dims=2048)和基于关键点的ROI指标。生成定性对比网格和基准CSV。",
      deliverables: "基准测试结果、定性对比网格、评估管线。",
      issues: "发现并修复宽高比失真（600x1600被压缩为256x256）。",
      risks: "无重大风险。", deviation: "新增宽高比修复并重新运行所有预处理。" },
  ];

  const labels = en ? ["Role","Tasks","Deliverables","Issues","Risks","Deviation"]
    : ["角色","任务","交付物","问题","风险","偏差"];

  for (const wd of weeks) {
    c.push(p(wd.week, { bold: true, before: 160, after: 40, color: "0D5C63" }));
    const vals = [wd.role, wd.tasks, wd.deliverables, wd.issues, wd.risks, wd.deviation];
    c.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 7360],
      rows: labels.map((l, i) => new TableRow({ children: [
        cell(l, { bold: true, width: 2000, shading: i % 2 === 0 ? altShading : undefined }),
        cell(vals[i], { width: 7360, shading: i % 2 === 0 ? altShading : undefined }),
      ] })) }));
  }
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 2: INDIVIDUAL ACHIEVEMENTS =====
  c.push(heading(en ? "2. Individual Achievements" : "2. 个人成就", HeadingLevel.HEADING_1));

  // 2.1 Data Pipeline
  c.push(heading(en ? "2.1 Data Pipeline and Deduplication" : "2.1 数据管线与去重系统", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I built a comprehensive data processing pipeline that handles 2,568 raw WhatsApp-exported rhinoplasty images. The pipeline implements recursive scanning across filesystem files and extracted subdirectories, Union-Find transitive deduplication (combining filename and perceptual hash similarity), automatic image pair splitting, and quality validation."
    : "我构建了一个综合性数据处理管线，处理2,568张WhatsApp导出的鼻整形原始图片。管线实现了文件系统和子目录的递归扫描、Union-Find传递性去重（结合文件名和感知哈希相似度）、自动图像对切分和质量验证。"));
  c.push(p(en
    ? "The Union-Find algorithm correctly handles transitive duplicates that sequential scanning misses, reducing the dataset from 2,568 to 833 unique samples."
    : "Union-Find算法正确处理了顺序扫描会遗漏的传递性重复，将数据集从2,568缩减至833个独立样本。"));
  c.push(...img("02_data_pipeline_stats.png", 500, 250,
    en ? "Figure 1: Data pipeline processing - from raw images to training set" : "图1：数据管线处理——从原始图片到训练集", en));

  // 2.2 View Classification
  c.push(heading(en ? "2.2 Face Detection and View Classification" : "2.2 人脸检测与视角分类", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I evaluated multiple face detection approaches. MediaPipe achieved only 7% on extreme profiles, OpenCV Haar cascades 49%. I integrated InsightFace which achieved 94% detection rate, providing 5-point landmarks (two eyes, nose tip, two mouth corners). I classified all 833 samples and excluded 252 frontal images, resulting in 581 trainable profile samples."
    : "我评估了多种人脸检测方案。MediaPipe在极端侧面上仅达7%，OpenCV Haar级联49%。我集成了InsightFace，达到94%检测率，提供5点关键点（两眼、鼻尖、两嘴角）。分类833个样本后排除252张正面图片，最终得到581个可训练侧面样本。"));
  c.push(...img("03_view_classification.png", 500, 220,
    en ? "Figure 2: Profile views (top, used for training) vs frontal views (bottom, excluded)" : "图2：侧面视图（上方，用于训练）vs 正面视图（下方，已排除）", en));

  // 2.3 Alignment
  c.push(heading(en ? "2.3 Face Alignment" : "2.3 人脸对齐", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I implemented face alignment using skin-colour detection to robustly locate, crop, and center the face region. Images are normalized to 256x256 while preserving original aspect ratio with black padding, avoiding the distortion caused by forced square resizing."
    : "我使用基于肤色检测的人脸对齐方法，鲁棒地定位、裁剪和居中面部区域。图像归一化到256x256，保持原始宽高比并使用黑边填充，避免强制正方形缩放造成的变形。"));
  c.push(...img("04_aligned_pairs.png", 520, 140,
    en ? "Figure 3: Aligned pre-operative (top) and post-operative (bottom) pairs" : "图3：对齐后的术前（上）和术后（下）图像对", en));

  // 2.4 Nose ROI Mask
  c.push(heading(en ? "2.4 Nose ROI Mask Generation" : "2.4 鼻部ROI Mask生成", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I developed the nose ROI mask system using InsightFace 5-point landmarks. After 8 iterations of tuning, I achieved accurate coverage using a tilted ellipse along the nose bridge axis: center at 65% from eye midpoint toward nose tip, long axis = 0.75x eye-nose distance, short axis = 0.55x. The mask fully covers bridge, tip, ala, and nostrils with Gaussian-blurred soft edges."
    : "我使用InsightFace 5点关键点开发了鼻部ROI Mask系统。经过8轮调优，使用沿鼻梁方向倾斜的椭圆实现了准确覆盖：中心在眼部中点到鼻尖的65%处，长轴0.75倍、短轴0.55倍眼鼻距离。Mask完整覆盖鼻梁、鼻尖、鼻翼和鼻孔，使用高斯模糊柔化边缘。"));
  c.push(...img("05_nose_roi_masks.png", 500, 200,
    en ? "Figure 4: Nose ROI mask overlay on profile images - green indicates the training loss region" : "图4：侧面图像上的鼻部ROI Mask叠加——绿色表示训练损失计算区域", en));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // 2.5 Model Training
  c.push(heading(en ? "2.5 Deep Learning Model Training" : "2.5 深度学习模型训练", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I implemented and trained 8 generative models: 4 full-face (256x256) and 4 nose-only ROI (128x128). Architectures include Autoencoder (UNet + L1), Pix2Pix (conditional GAN), CycleGAN (bidirectional with cycle consistency), and Diffusion (conditional DDPM). All trained for 50 epochs on 465 samples with synchronized data augmentation."
    : "我实现并训练了8个生成模型：4个全脸(256x256)和4个鼻部ROI(128x128)。架构包括Autoencoder(UNet+L1)、Pix2Pix(条件GAN)、CycleGAN(带循环一致性的双向翻译)和Diffusion(条件DDPM)。所有模型在465个样本上训练50个epoch，配合同步数据增强。"));
  c.push(...img("06_training_curves.png", 540, 120,
    en ? "Figure 5: Validation L1 loss over 50 epochs for all 4 nose-only models" : "图5：4个鼻部模型50个epoch的验证L1损失曲线", en));
  c.push(...img("07_model_results.png", 480, 160,
    en ? "Figure 6: Nose ROI model training results summary" : "图6：鼻部ROI模型训练结果汇总", en));

  // 2.6 Evaluation
  c.push(heading(en ? "2.6 Comprehensive Evaluation" : "2.6 综合评估", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I ran full evaluation on 58 test samples using standard metrics: SSIM (structural similarity), LPIPS (perceptual quality, AlexNet), and FID (distribution distance, InceptionV3 dims=2048). The evaluation uses landmark-based dynamic ROI cropping instead of hardcoded pixel coordinates."
    : "我在58个测试样本上使用标准指标进行了完整评估：SSIM(结构相似性)、LPIPS(感知质量，AlexNet)和FID(分布距离，InceptionV3 dims=2048)。评估使用基于关键点的动态ROI裁剪替代硬编码像素坐标。"));
  c.push(...img("08_benchmark_table.png", 520, 140,
    en ? "Figure 7: Full-face model evaluation benchmark (58 test samples)" : "图7：全脸模型评估基准（58个测试样本）", en));

  // 2.7 NLP
  c.push(heading(en ? "2.7 NLP Surgical Description Generation" : "2.7 NLP手术描述生成", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "I developed an NLP module that compares pre/post nasal landmark measurements across 7 dimensions (bridge angle, tip projection, ala width, bridge length, nasofrontal angle, nasolabial angle, symmetry) and generates structured surgical change descriptions."
    : "我开发了NLP模块，比较术前术后鼻部关键点测量值在7个维度（鼻梁角度、鼻尖投射比、鼻翼宽度、鼻梁长度、鼻额角、鼻唇角、对称性）上的差异，生成结构化手术变化描述。"));
  c.push(...img("09_nlp_description.png", 500, 180,
    en ? "Figure 8: Example NLP surgical description output" : "图8：NLP手术描述生成示例输出", en));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 3: SIGNIFICANCE =====
  c.push(heading(en ? "3. Significance of My Work" : "3. 工作的重要性", HeadingLevel.HEADING_1));
  c.push(p(en
    ? "My contributions form the foundational infrastructure of the entire CS31 project. Without the data pipeline I built, the team would have no clean, deduplicated, properly split training data. Without face alignment and nose ROI mask, the models would learn on distorted images and waste capacity on unchanged facial regions."
    : "我的贡献构成了整个CS31项目的基础设施。没有我构建的数据管线，团队将没有清洁、去重、正确划分的训练数据。没有人脸对齐和鼻部ROI Mask，模型将在失真的图像上学习，并将学习能力浪费在不变的面部区域上。"));
  c.push(p(en
    ? "The Union-Find deduplication algorithm identified 22 additional transitive duplicates that sequential scanning would miss, directly improving data quality. My decision to switch from MediaPipe (7%) to InsightFace (94%) for profile face detection was pivotal - without reliable detection, neither alignment nor mask generation would work."
    : "Union-Find去重算法识别出顺序扫描会遗漏的22个额外传递性重复，直接提高了数据质量。我从MediaPipe(7%)切换到InsightFace(94%)进行侧面人脸检测的决策至关重要——没有可靠的检测，对齐和Mask生成都无法工作。"));
  c.push(p(en
    ? "I identified and fixed a critical aspect ratio distortion bug where 600x1600 images were being squashed to 256x256, which would have severely degraded all model training results. The nose ROI mask is the key technical contribution that enables the model to focus exclusively on the nasal region, as required by the project specification."
    : "我发现并修复了关键的宽高比失真bug——600x1600的图片被压缩为256x256，这会严重降低所有模型训练效果。鼻部ROI Mask是关键技术贡献，使模型能够专注于鼻部区域，满足项目规格要求。"));
  c.push(p(en
    ? "The nose-only ROI training approach I proposed reduced training time by 4x while focusing 100% of model capacity on the actual surgical change region. The NLP description module adds interpretability to the system, automatically explaining what changes occurred in natural language."
    : "我提出的鼻部ROI专项训练方法将训练时间减少4倍，同时将模型100%的学习能力集中在实际手术变化区域。NLP描述模块为系统增加了可解释性，用自然语言自动解释发生了什么变化。"));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 4: REFLECTIONS =====
  c.push(heading(en ? "4. Reflections on Learning Outcomes" : "4. 学习成果反思", HeadingLevel.HEADING_1));
  c.push(p(en ? "My degree specialisation is Computer Science." : "我的专业方向是计算机科学。"));
  c.push(heading(en ? "4.1 Existing Knowledge Applied" : "4.1 已有知识的应用", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "Data Structures and Algorithms: I applied Union-Find with path compression and rank optimisation for deduplication - a classic algorithm adapted for image similarity clustering. Software Engineering: I followed modular design with clear separation of concerns, wrote unit tests with pytest, and used configuration files. Database fundamentals informed the SQLite schema design for prediction history."
    : "数据结构与算法：我将带路径压缩和秩优化的Union-Find应用于去重——经典算法适配图像相似度聚类。软件工程：遵循模块化设计和关注点分离，使用pytest编写单元测试，使用配置文件。数据库基础知识指导了预测历史的SQLite模式设计。"));
  c.push(heading(en ? "4.2 New Knowledge Gained" : "4.2 新获得的知识", HeadingLevel.HEADING_2));
  c.push(p(en
    ? "Generative Adversarial Networks: I learned to implement and train Pix2Pix and CycleGAN from scratch, including the balance between generator/discriminator training and loss function design (adversarial + L1 + cycle consistency)."
    : "生成对抗网络：我学会了从零实现和训练Pix2Pix和CycleGAN，包括生成器/判别器训练的平衡和损失函数设计（对抗+L1+循环一致性）。"));
  c.push(p(en
    ? "Computer Vision for Medical Imaging: I gained experience with facial landmark detection (InsightFace, MediaPipe), perceptual hashing, ROI mask generation for region-specific learning, and evaluation metrics (SSIM, LPIPS, FID). Diffusion Models: I learned DDPM fundamentals including noise scheduling, time embedding, and iterative sampling."
    : "医学图像计算机视觉：我获得了面部关键点检测（InsightFace、MediaPipe）、感知哈希、区域特定学习的ROI Mask生成，以及评估指标（SSIM、LPIPS、FID）的经验。扩散模型：我学习了DDPM基础，包括噪声调度、时间嵌入和迭代采样。"));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 5: GROUP COLLABORATION =====
  c.push(heading(en ? "5. Group Collaboration" : "5. 团队协作", HeadingLevel.HEADING_1));
  c.push(p(en ? "I actively participated in group collaboration:" : "我积极参与了团队协作："));
  c.push(p(en ? "Weekly Team Meetings:" : "每周团队会议：", { bold: true }));
  c.push(p(en
    ? "I attended all weekly meetings from Week 2 to Week 6. I presented progress on the data pipeline, face alignment, and model training, discussed technical challenges, and coordinated next steps with team members."
    : "我参加了第2周到第6周的所有周会。我展示了数据管线、人脸对齐和模型训练的进展，讨论技术挑战，并与团队成员协调下一步工作。"));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: en ? "[INSERT SCREENSHOT: Meeting photos/screenshots Week 2-6]" : "[插入截图：第2-6周会议照片/截图]",
      font: "Arial", size: 22, italics: true, color: "CC0000" })] }));
  c.push(p(en ? "Communication and Code Collaboration:" : "沟通与代码协作：", { bold: true }));
  c.push(p(en
    ? "I shared updates on data processing results, model training progress, and technical decisions through the team communication channel. I maintained the codebase and shared implementations through version control."
    : "我通过团队沟通渠道分享数据处理结果、模型训练进度和技术决策的更新。我维护代码库并通过版本控制共享实现。"));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: en ? "[INSERT SCREENSHOT: Communication tool / GitHub screenshots]" : "[插入截图：沟通工具/GitHub截图]",
      font: "Arial", size: 22, italics: true, color: "CC0000" })] }));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 6: PEER REVIEW =====
  c.push(heading(en ? "6. Peer Review" : "6. 同行评审", HeadingLevel.HEADING_1));
  c.push(p(en ? "I evaluated my teammates based on their contributions to project management, development, and collaboration."
    : "我基于队友在项目管理、开发和协作方面的贡献进行了评估。"));
  const peerRows = [new TableRow({ children: [
    cell(en ? "Student ID" : "学号", { bold: true, width: 2000, shading: headerShading }),
    cell(en ? "Name" : "姓名", { bold: true, width: 2500, shading: headerShading }),
    cell(en ? "Rating (0-5)" : "评分(0-5)", { bold: true, width: 1500, shading: headerShading, align: AlignmentType.CENTER }),
    cell(en ? "Reason" : "理由", { bold: true, width: 3360, shading: headerShading }),
  ] })];
  for (let i = 0; i < 7; i++) {
    peerRows.push(new TableRow({ children: [
      cell("", { width: 2000, shading: i % 2 === 0 ? altShading : undefined }),
      cell("", { width: 2500, shading: i % 2 === 0 ? altShading : undefined }),
      cell("", { width: 1500, shading: i % 2 === 0 ? altShading : undefined, align: AlignmentType.CENTER }),
      cell("", { width: 3360, shading: i % 2 === 0 ? altShading : undefined }),
    ] }));
  }
  c.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 2500, 1500, 3360], rows: peerRows }));
  c.push(p(en ? "Overall Comments:" : "总体评价：", { bold: true, before: 160 }));
  c.push(p(en ? "[INSERT YOUR OVERALL COMMENTS]" : "[插入你的总体评价]", { italic: true, color: "CC0000" }));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ===== SECTION 7: AI STATEMENT =====
  c.push(heading(en ? "7. AI Statement" : "7. AI使用声明", HeadingLevel.HEADING_1));
  const aiRows = [
    new TableRow({ children: [cell(en?"Part A":"A部分",{bold:true,width:2000,shading:headerShading}), cell(en?"Have you used AI tools?":"是否使用了AI工具？",{bold:true,width:7360,shading:headerShading})] }),
    new TableRow({ children: [cell("",{width:2000}), cell(en?"Yes":"是",{width:7360})] }),
    new TableRow({ children: [cell(en?"Part B":"B部分",{bold:true,width:2000,shading:altShading}), cell(en?"What AI tools?":"使用了哪些AI工具？",{bold:true,width:7360,shading:altShading})] }),
    new TableRow({ children: [cell("",{width:2000}), cell(en?"Claude (https://claude.ai) - Anthropic's AI assistant via Claude Code CLI.":"Claude (https://claude.ai) - Anthropic的AI助手，通过Claude Code CLI使用。",{width:7360})] }),
    new TableRow({ children: [cell(en?"Part C":"C部分",{bold:true,width:2000,shading:altShading}), cell(en?"How were AI tools used?":"如何使用了AI工具？",{bold:true,width:7360,shading:altShading})] }),
    new TableRow({ children: [cell("",{width:2000}), cell(en
      ?"I used Claude Code to assist with code development for the data pipeline, model training scripts, and evaluation modules. The tool was used for: (1) debugging data processing issues such as aspect ratio distortion and deduplication logic; (2) implementing PyTorch model architectures based on my design specifications; (3) generating evaluation metric computation code. I completed multiple iterations reviewing and modifying the AI-generated code. All algorithmic decisions (Union-Find for dedup, InsightFace for detection, tilted ellipse for mask, nose-only ROI training) were my own design choices. I modified the AI-generated code by adding error handling, adjusting hyperparameters, and restructuring modules."
      :"我使用Claude Code辅助数据管线、模型训练脚本和评估模块的代码开发。该工具用于：(1)调试数据处理问题如宽高比失真和去重逻辑；(2)根据我的设计规范实现PyTorch模型架构；(3)生成评估指标计算代码。我完成了多次迭代审查和修改AI生成的代码。所有算法决策（Union-Find去重、InsightFace检测、倾斜椭圆Mask、鼻部ROI训练）都是我自己的设计选择。我通过添加错误处理、调整超参数和重组模块来修改AI生成的代码。",{width:7360})] }),
  ];
  c.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 7360], rows: aiRows }));

  return new Document({ numbering,
    styles: { default: { document: { run: { font: "Arial", size: 24 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial" }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
      ] },
    sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: c }],
  });
}

async function main() {
  fs.writeFileSync("/Applications/CS31/reports/CS31-1_Individual_Report_EN.docx", await Packer.toBuffer(buildDoc("en")));
  console.log("English report saved");
  fs.writeFileSync("/Applications/CS31/reports/CS31-1_Individual_Report_CN.docx", await Packer.toBuffer(buildDoc("cn")));
  console.log("Chinese report saved");
}
main().catch(console.error);
