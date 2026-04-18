const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, LevelFormat,
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function makeDoc(lang) {
  const isEN = lang === "en";

  const t = {
    title: isEN ? "CS31 - Weekly Meeting Minutes" : "CS31 - 每周会议记录",
    week: isEN ? "Week 7" : "第七周",
    project: isEN
      ? "Predicting Rhinoplasty Outcomes Using AI and Facial Image Analysis"
      : "基于AI和面部图像分析的鼻整形术后效果预测",

    goalsTitle: isEN ? "Goals" : "本周目标",
    goalsIntro: isEN
      ? "Based on the Week 6 meeting action items:"
      : "基于第六周会议的行动项:",
    goals: isEN
      ? [
          "Implement face alignment to normalize and standardize facial images for training",
          "Generate nose ROI masks to restrict model learning to the nasal region only",
          "Ensure other facial features remain unchanged during prediction (identity preservation)",
          "Complete data cleaning: deduplication + profile/frontal filtering",
          "Split images into before/after paired samples",
          "Prepare aligned data and masks for GAN/Diffusion model training",
        ]
      : [
          "实现人脸对齐，归一化和标准化训练用面部图像",
          "生成鼻部 ROI Mask，限制模型仅学习鼻部区域的变化",
          "确保预测时其他面部特征保持不变（身份一致性）",
          "完成数据清洗：去重 + 侧脸/正脸筛选",
          "将图像拆分为 before/after 配对样本",
          "准备对齐后的数据和掩码，用于 GAN/Diffusion 模型训练",
        ],

    progressTitle: isEN ? "Progress Made" : "本周进展",
    progressSections: isEN
      ? [
          {
            heading: "1. Data Cleaning and Deduplication",
            items: [
              "Scanned 2,568 raw images from filesystem and extracted archive subdirectories",
              "Implemented Union-Find transitive deduplication algorithm combining filename matching and perceptual hash (pHash) similarity (threshold=4), reducing dataset from 2,568 to 833 unique samples",
              "Fixed square image splitting (1080x1080 images use top-bottom split instead of left-right)",
              "Added quality validation to detect and skip blank/corrupt images",
              "All 833 unique samples passed quality checks",
            ],
          },
          {
            heading: "2. View Classification and Filtering",
            items: [
              "Used InsightFace deep learning model (94% detection rate on profile views) for face detection and 5-point landmark extraction (two eyes, nose tip, two mouth corners)",
              "Classified 833 samples: 266 profile + 315 unknown (extreme profile) + 252 frontal",
              "Excluded 252 frontal images from training set (project targets profile-to-profile prediction)",
              "Final trainable dataset: 581 profile samples (Train: 465 / Val: 58 / Test: 58)",
            ],
          },
          {
            heading: "3. Face Alignment",
            items: [
              "Implemented skin-color-based face region detection for robust alignment across different backgrounds",
              "Face images are cropped, centered, and normalized to consistent scale (256x256)",
              "Aspect ratio is preserved during resizing (black padding instead of distortion)",
              "Alignment uses InsightFace nose tip and eye midpoint as anchor references",
            ],
          },
          {
            heading: "4. Nose ROI Mask Generation",
            items: [
              "Used InsightFace 5-point landmarks to precisely locate the nose tip on each image",
              "Mask center positioned at 65% from eye midpoint toward nose tip (covers bridge to nostrils)",
              "Tilted elliptical mask aligned along the nose bridge direction (eye-to-nose axis)",
              "Mask dimensions: long axis = 0.75x eye-nose distance, short axis = 0.55x (covers full nose including nostrils)",
              "Soft Gaussian blur edges for smooth gradient transition at mask boundaries",
              "Validated on 50+ samples with manual inspection - confirmed accurate coverage of bridge, tip, ala, and nostrils",
            ],
          },
          {
            heading: "5. Model Training (Full-Face Baseline)",
            items: [
              "Trained 4 generative models on full-face images (256x256): Autoencoder, Pix2Pix, CycleGAN, Diffusion",
              "Each model trained for 50 epochs on 465 training samples with data augmentation (synchronized flip + color jitter)",
              "Best full-face validation L1: CycleGAN 0.0607, Autoencoder 0.0634, Pix2Pix 0.0711",
              "Full evaluation on 58 test samples with SSIM, LPIPS (AlexNet), and FID (InceptionV3, dims=2048)",
              "Best SSIM: 0.867 (Autoencoder), Best LPIPS: 0.152 (Pix2Pix)",
            ],
          },
          {
            heading: "6. Nose ROI Model Training",
            items: [
              "Trained 4 nose-only models (128x128 ROI crops): Autoencoder, Pix2Pix, CycleGAN, Diffusion",
              "Training speed improved ~4x compared to full-face (128x128 vs 256x256)",
              "Best nose-only validation L1: Autoencoder 0.0760, CycleGAN 0.0763, Pix2Pix 0.0771",
            ],
          },
          {
            heading: "7. Additional Modules Implemented",
            items: [
              "NLP surgical description generation: compares pre/post landmark measurements across 7 dimensions (bridge angle, tip projection, ala width, nasofrontal angle, nasolabial angle, bridge length, symmetry)",
              "Web application (FastAPI + React): upload, predict, compare models, view training curves, landmark features display",
              "Comprehensive evaluation pipeline with benchmark CSV export and qualitative comparison grids",
            ],
          },
        ]
      : [
          {
            heading: "1. 数据清洗与去重",
            items: [
              "扫描 2,568 张原始图片（文件系统 + 解压子目录）",
              "实现 Union-Find 传递性去重算法，结合文件名匹配和感知哈希(pHash)相似度(阈值=4)，将数据集从 2,568 缩减至 833 个独立样本",
              "修复正方形图片切分（1080x1080 改为上下切分而非左右切分）",
              "新增质量验证，检测并跳过空白/损坏图片",
              "833 个独立样本全部通过质量检查",
            ],
          },
          {
            heading: "2. 视角分类与筛选",
            items: [
              "使用 InsightFace 深度学习模型（侧面检测率 94%）进行人脸检测和 5 点关键点提取（两眼、鼻尖、两嘴角）",
              "833 个样本分类：266 侧面 + 315 未知（极端侧面）+ 252 正面",
              "排除 252 张正面图片（项目目标为侧面到侧面预测）",
              "最终可训练数据集：581 张侧面样本（训练 465 / 验证 58 / 测试 58）",
            ],
          },
          {
            heading: "3. 人脸对齐",
            items: [
              "基于肤色检测实现鲁棒的面部区域定位，适应不同背景",
              "面部图像裁剪、居中、归一化到统一尺度（256x256）",
              "缩放时保持宽高比（黑边填充，避免拉伸变形）",
              "使用 InsightFace 鼻尖和眼部中点作为对齐锚点",
            ],
          },
          {
            heading: "4. 鼻部 ROI Mask 生成",
            items: [
              "使用 InsightFace 5 点关键点精确定位每张图片的鼻尖",
              "Mask 中心位于眼部中点到鼻尖的 65% 处（覆盖鼻梁到鼻孔）",
              "倾斜椭圆 Mask 沿鼻梁方向对齐（眼-鼻轴线）",
              "Mask 尺寸：长轴 = 0.75 倍眼鼻距离，短轴 = 0.55 倍（完整覆盖鼻梁、鼻尖、鼻翼、鼻孔）",
              "高斯模糊柔化边缘，实现平滑渐变过渡",
              "在 50+ 样本上人工检查验证——确认精准覆盖鼻梁、鼻尖、鼻翼和鼻孔",
            ],
          },
          {
            heading: "5. 模型训练（全脸基线）",
            items: [
              "在全脸图像（256x256）上训练 4 种生成模型：Autoencoder、Pix2Pix、CycleGAN、Diffusion",
              "每个模型训练 50 epochs，使用 465 个训练样本，配合数据增强（同步翻转 + 颜色抖动）",
              "全脸最佳验证 L1：CycleGAN 0.0607、Autoencoder 0.0634、Pix2Pix 0.0711",
              "在 58 个测试样本上完整评估：SSIM、LPIPS (AlexNet)、FID (InceptionV3, dims=2048)",
              "最佳 SSIM：0.867（Autoencoder），最佳 LPIPS：0.152（Pix2Pix）",
            ],
          },
          {
            heading: "6. 鼻部 ROI 模型训练",
            items: [
              "训练 4 个鼻部模型（128x128 ROI 裁剪）：Autoencoder、Pix2Pix、CycleGAN、Diffusion",
              "训练速度相比全脸提升约 4 倍（128x128 vs 256x256）",
              "鼻部最佳验证 L1：Autoencoder 0.0760、CycleGAN 0.0763、Pix2Pix 0.0771",
            ],
          },
          {
            heading: "7. 其他已实现模块",
            items: [
              "NLP 手术描述生成：比较术前术后关键点测量值，涵盖 7 个维度（鼻梁角度、鼻尖投射比、鼻翼宽度、鼻额角、鼻唇角、鼻梁长度、对称性）",
              "Web 应用（FastAPI + React）：上传、预测、模型对比、训练曲线查看、关键点特征展示",
              "完整评估管线，支持 benchmark CSV 导出和定性对比网格",
            ],
          },
        ],

    blockersTitle: isEN ? "Blockers to Progress / Issues" : "进展阻碍 / 问题",
    blockers: isEN
      ? [
          "MediaPipe face landmark detection has low accuracy on extreme profile views (~7% success rate). Switched to InsightFace which achieves 94% detection rate on the same images.",
          "CycleGAN training was repeatedly terminated due to background task timeout limits. Resolved by running training in foreground with checkpoint-based resume capability.",
          "Original images had aspect ratio distortion (600x1600 forced to 256x256 square). Fixed by implementing aspect-ratio-preserving resize with black padding.",
          "Nose ROI mask positioning required multiple iterations of tuning. Initial approaches (MediaPipe heuristic, edge detection, silhouette analysis) failed on profile views. Final solution uses InsightFace 5-point landmarks with tilted ellipse along nose bridge axis.",
          "No real cost annotation data available - cost prediction module deferred pending client data.",
        ]
      : [
          "MediaPipe 面部关键点检测在极端侧面视图上准确率极低（约 7%）。已切换至 InsightFace，在相同图片上检测率达 94%。",
          "CycleGAN 训练因后台任务超时限制被反复终止。通过前台运行 + 基于检查点的断点续训解决。",
          "原始图像存在宽高比失真（600x1600 被强制压缩为 256x256 正方形）。已修复为保持宽高比的缩放 + 黑边填充。",
          "鼻部 ROI Mask 定位经过多轮调优。初始方案（MediaPipe 启发式、边缘检测、轮廓分析）在侧面图上均失败。最终方案使用 InsightFace 5 点关键点 + 沿鼻梁方向的倾斜椭圆。",
          "无真实费用标注数据——费用预测模块推迟至客户提供数据后实现。",
        ],

    nextTitle: isEN ? "Next Steps" : "下一步计划",
    nextSteps: isEN
      ? [
          "Integrate ROI mask into training loss function (masked L1/adversarial loss)",
          "Retrain all 4 models with mask-weighted loss on aligned data",
          "Evaluate mask-trained models and compare with baseline full-face results",
          "Refine the paste-back pipeline for inference (feathered blending of predicted nose onto original image)",
          "Update project report with alignment and mask results",
        ]
      : [
          "将 ROI Mask 集成到训练损失函数中（Mask 加权的 L1/对抗损失）",
          "使用 Mask 加权损失在对齐数据上重新训练全部 4 个模型",
          "评估 Mask 训练模型并与全脸基线结果对比",
          "优化推理时的贴回管线（羽化混合将预测鼻部贴回原图）",
          "更新项目报告，加入对齐和 Mask 结果",
        ],
  };

  const numbering = {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  };

  const children = [];

  // Title
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: t.title, bold: true, size: 36, font: "Arial" })],
      spacing: { after: 100 },
    })
  );
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `${t.week} | ${t.project}`, size: 22, color: "555555", font: "Arial" })],
      spacing: { after: 300 },
    })
  );

  // Goals
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: t.goalsTitle, bold: true, size: 28, font: "Arial", color: "0D5C63" })],
      spacing: { before: 200, after: 100 },
    })
  );
  children.push(
    new Paragraph({ children: [new TextRun({ text: t.goalsIntro, size: 22, font: "Arial", italics: true })], spacing: { after: 80 } })
  );
  for (const g of t.goals) {
    children.push(
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [new TextRun({ text: g, size: 22, font: "Arial" })],
        spacing: { after: 40 },
      })
    );
  }

  // Progress
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: t.progressTitle, bold: true, size: 28, font: "Arial", color: "0D5C63" })],
      spacing: { before: 300, after: 100 },
    })
  );
  for (const section of t.progressSections) {
    children.push(
      new Paragraph({
        children: [new TextRun({ text: section.heading, bold: true, size: 24, font: "Arial" })],
        spacing: { before: 160, after: 60 },
      })
    );
    for (const item of section.items) {
      children.push(
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: item, size: 22, font: "Arial" })],
          spacing: { after: 40 },
        })
      );
    }
  }

  // Blockers
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: t.blockersTitle, bold: true, size: 28, font: "Arial", color: "0D5C63" })],
      spacing: { before: 300, after: 100 },
    })
  );
  for (const b of t.blockers) {
    children.push(
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [new TextRun({ text: b, size: 22, font: "Arial" })],
        spacing: { after: 60 },
      })
    );
  }

  // Next Steps
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: t.nextTitle, bold: true, size: 28, font: "Arial", color: "0D5C63" })],
      spacing: { before: 300, after: 100 },
    })
  );
  for (const n of t.nextSteps) {
    children.push(
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [new TextRun({ text: n, size: 22, font: "Arial" })],
        spacing: { after: 40 },
      })
    );
  }

  return new Document({
    numbering,
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 36, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 12240, height: 15840 },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        children,
      },
    ],
  });
}

async function main() {
  const enDoc = makeDoc("en");
  const cnDoc = makeDoc("cn");

  const enBuf = await Packer.toBuffer(enDoc);
  fs.writeFileSync("/Applications/CS31/reports/Meeting_Minutes_Week7_EN.docx", enBuf);
  console.log("English minutes saved");

  const cnBuf = await Packer.toBuffer(cnDoc);
  fs.writeFileSync("/Applications/CS31/reports/Meeting_Minutes_Week7_CN.docx", cnBuf);
  console.log("Chinese minutes saved");
}

main().catch(console.error);
