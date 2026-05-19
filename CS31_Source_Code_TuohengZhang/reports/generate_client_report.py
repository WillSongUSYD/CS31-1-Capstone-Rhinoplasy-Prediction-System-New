"""Generate client-facing reports in English and Chinese."""

import csv
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image as RLImage,
)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

PRIMARY = HexColor("#0d5c63")
DARK = HexColor("#1b1a18")
GRAY = HexColor("#4b4c51")
LIGHT_BG = HexColor("#f0f4f8")
WHITE = HexColor("#ffffff")
GREEN = HexColor("#2e7d32")
RED = HexColor("#c62828")

ASSETS = Path("/Applications/CS31/tmp/report_assets")


def get_styles(font):
    return {
        "title": ParagraphStyle("Title", fontName=font, fontSize=22, leading=30, alignment=TA_CENTER, textColor=PRIMARY, spaceAfter=6),
        "subtitle": ParagraphStyle("Sub", fontName=font, fontSize=12, leading=18, alignment=TA_CENTER, textColor=GRAY, spaceAfter=20),
        "h1": ParagraphStyle("H1", fontName=font, fontSize=15, leading=22, textColor=PRIMARY, spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontName=font, fontSize=12, leading=18, textColor=DARK, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("Body", fontName=font, fontSize=10.5, leading=17, textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", fontName=font, fontSize=10.5, leading=17, textColor=DARK, leftIndent=18, bulletIndent=6, spaceAfter=4),
        "caption": ParagraphStyle("Caption", fontName=font, fontSize=9, leading=14, textColor=GRAY, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12),
        "small": ParagraphStyle("Small", fontName=font, fontSize=9, leading=13, textColor=GRAY, spaceAfter=4),
    }


def make_table(headers, rows, col_widths=None, font="Helvetica"):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 14),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def add_image(story, path, width_mm=160, caption=None, styles=None):
    img = RLImage(str(path), width=width_mm * mm, height=width_mm * mm * 0.4)
    img.hAlign = "CENTER"
    story.append(img)
    if caption and styles:
        story.append(Paragraph(caption, styles["caption"]))


def build_english():
    s = get_styles("Helvetica")
    font = "Helvetica"
    output = Path("/Applications/CS31/reports/CS31_Project_Report_EN.pdf")
    doc = SimpleDocTemplate(str(output), pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm, topMargin=25*mm, bottomMargin=25*mm)
    story = []

    # Cover
    story.append(Spacer(1, 50))
    story.append(Paragraph("CS31 Rhinoplasty Outcome Prediction", s["title"]))
    story.append(Paragraph("Project Development Report", ParagraphStyle("S2", fontName=font, fontSize=18, leading=26, alignment=TA_CENTER, textColor=DARK, spaceAfter=8)))
    story.append(HRFlowable(width="50%", thickness=1, color=PRIMARY, spaceAfter=16))
    story.append(Paragraph("Predicting Rhinoplasty Outcomes Using AI and Facial Image Analysis", s["subtitle"]))
    story.append(Spacer(1, 20))
    cover = [
        ["Project ID", "CS31"],
        ["Technology", "Python / PyTorch / FastAPI / React / MediaPipe"],
        ["Dataset", "2,568 images, 833 unique, 581 profile samples for training"],
        ["Models", "4 architectures x nose ROI (128x128), 50 epochs each"],
    ]
    t = Table(cover, colWidths=[120, 340])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("LEADING", (0, 0), (-1, -1), 18), ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"), ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
    ]))
    story.append(t)
    story.append(PageBreak())

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", s["h1"]))
    story.append(Paragraph(
        "This project develops a deep learning system that predicts post-rhinoplasty outcomes from pre-operative "
        "facial profile photographs. The system processes paired before-and-after images, extracts nose-region "
        "features using facial landmark detection, and trains generative models to produce realistic predictions "
        "of surgical results. A complete web application provides an interactive interface for clinicians and patients.", s["body"]))
    story.append(Paragraph(
        "Key achievements: fully automated data pipeline processing 2,568 images with intelligent deduplication; "
        "facial landmark detection with 7 quantitative nasal measurements; NLP-based surgical description generation; "
        "4 trained deep learning models (Autoencoder, Pix2Pix, CycleGAN, Diffusion); and a full-stack web application.", s["body"]))

    # 2. Data Pipeline
    story.append(Paragraph("2. Data Acquisition and Preprocessing", s["h1"]))
    story.append(Paragraph("2.1 Raw Data", s["h2"]))
    story.append(Paragraph(
        "The dataset consists of 2,568 WhatsApp-exported rhinoplasty images. Each original image is a side-by-side "
        "canvas with the pre-operative profile on the left and the post-operative profile on the right. "
        "The images come from both direct filesystem files (1,202) and extracted archive subdirectories (1,366).", s["body"]))

    story.append(Paragraph("2.2 Deduplication", s["h2"]))
    story.append(Paragraph(
        "A Union-Find algorithm performs transitive deduplication using both filename matching and perceptual hash "
        "(pHash) similarity with a distance threshold of 4. This reduced the dataset from 2,568 to 833 unique samples.", s["body"]))
    dedup = [
        ["Stage", "Count", "Description"],
        ["Raw scanned", "2,568", "Filesystem files + subdirectory files"],
        ["After dedup", "833", "Union-Find transitive clustering"],
        ["Duplicate (name)", "1,072", "Same filename across sources"],
        ["Duplicate (pHash)", "663", "Visually identical or near-identical"],
    ]
    story.append(make_table(dedup[0], dedup[1:], col_widths=[100, 60, 300], font=font))

    story.append(Paragraph("2.3 View Classification and Filtering", s["h2"]))
    story.append(Paragraph(
        "MediaPipe FaceLandmarker classifies each image as profile or frontal based on inter-eye distance "
        "and nose-tip offset. Since the project targets profile-to-profile prediction, 252 frontal images "
        "are excluded from training, leaving 581 profile samples.", s["body"]))
    add_image(story, ASSETS / "view_type_comparison.jpg", 155,
              "Figure 1: Profile views (top, used for training) vs. frontal views (bottom, excluded)", s)

    story.append(Paragraph("2.4 Image Splitting and Aspect Ratio Preservation", s["h2"]))
    story.append(Paragraph(
        "Each paired canvas is split into separate pre-operative and post-operative images. Portrait images "
        "(e.g., 600x1600) are resized to 256x256 while preserving the original aspect ratio, with black padding "
        "on both sides. Square images (1080x1080) are split top-bottom instead of left-right.", s["body"]))

    story.append(Paragraph("2.5 Data Samples", s["h2"]))
    add_image(story, ASSETS / "fullface_pairs_sample.jpg", 155,
              "Figure 2: Full-face sample pairs after preprocessing (top: pre-op, bottom: post-op)", s)
    story.append(PageBreak())

    # 3. Nose ROI
    story.append(Paragraph("3. Nose Region of Interest (ROI) Extraction", s["h1"]))
    story.append(Paragraph(
        "Since rhinoplasty only modifies the nasal region, training on full-face images forces the model to learn "
        "the trivial task of keeping 90% of pixels unchanged. By extracting nose ROI crops, 100% of the model's "
        "learning capacity is focused on the actual surgical changes.", s["body"]))
    for item in [
        "Landmark-based ROI: When MediaPipe detects the face, the nose bounding box is computed from 7 nasal landmarks with 30% padding",
        "Heuristic fallback: For extreme profile views where face detection fails (~93% of samples), a proportional crop is used",
        "All ROIs are resized to 128x128 pixels, maintaining aspect ratio",
        "581 pre/post nose ROI pairs successfully extracted",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))
    add_image(story, ASSETS / "nose_roi_pairs_sample.jpg", 155,
              "Figure 3: Nose ROI pairs used for training (top: pre-op, bottom: post-op)", s)

    # 4. Facial Landmarks & NLP
    story.append(Paragraph("4. Facial Landmark Detection and NLP", s["h1"]))
    story.append(Paragraph("4.1 Nasal Feature Extraction", s["h2"]))
    story.append(Paragraph(
        "Using MediaPipe's 478-point face mesh, 7 quantitative nasal features are computed for each image:", s["body"]))
    feat = [
        ["Feature", "Description"],
        ["Bridge Angle", "Angle of nasal bridge relative to vertical"],
        ["Tip Projection", "Horizontal nose tip offset / bridge length ratio"],
        ["Ala Width", "Distance between left and right nasal ala"],
        ["Bridge Length", "Distance from bridge top to nose tip"],
        ["Nasofrontal Angle", "Angle at the bridge-forehead junction"],
        ["Nasolabial Angle", "Angle between nose base and upper lip"],
        ["Symmetry Score", "Left-right ala symmetry ratio (0-1)"],
    ]
    story.append(make_table(feat[0], feat[1:], col_widths=[110, 350], font=font))

    story.append(Paragraph("4.2 Surgical Description Generation (NLP)", s["h2"]))
    story.append(Paragraph(
        "By comparing pre-operative and post-operative nasal measurements, the system automatically generates "
        "structured surgical change descriptions. Each dimension has a detection threshold; changes exceeding "
        "the threshold produce natural language descriptions such as 'Nasal bridge refined (+5.6 degrees)' "
        "or 'Nasal tip rotated upward'. The output includes a change list, summary, and detailed metrics.", s["body"]))
    story.append(PageBreak())

    # 5. Model Development
    story.append(Paragraph("5. Deep Learning Model Development", s["h1"]))
    story.append(Paragraph(
        "Four generative architectures are implemented and trained on 128x128 nose ROI crops:", s["body"]))

    story.append(Paragraph("5.1 Autoencoder", s["h2"]))
    story.append(Paragraph(
        "Baseline encoder-decoder using UNet architecture with skip connections. Trained with L1 loss "
        "for direct pixel-level supervision. Parameters: ~29.2M.", s["body"]))

    story.append(Paragraph("5.2 Pix2Pix (Conditional GAN)", s["h2"]))
    story.append(Paragraph(
        "Conditional adversarial network with UNet generator and PatchDiscriminator. Generator loss combines "
        "BCE adversarial loss with L1 reconstruction (lambda=100). Parameters: ~36M.", s["body"]))

    story.append(Paragraph("5.3 CycleGAN", s["h2"]))
    story.append(Paragraph(
        "Bidirectional translation with two generators and two discriminators. Training objectives include "
        "adversarial loss, cycle consistency (lambda=10), and identity preservation (lambda=5). Parameters: ~72M.", s["body"]))

    story.append(Paragraph("5.4 Diffusion Model", s["h2"]))
    story.append(Paragraph(
        "Lightweight conditional diffusion model (TinyDiffusionUNet) as a feasibility study. Operates at 64x64 "
        "with 100-step linear beta schedule. Parameters: ~1.9M.", s["body"]))

    story.append(Paragraph("5.5 Training Configuration", s["h2"]))
    config = [
        ["Parameter", "Value"],
        ["Training samples", "465 (80% of 581 profile images)"],
        ["Validation samples", "58 (10%)"],
        ["Test samples", "58 (10%)"],
        ["Epochs", "50 per model"],
        ["Optimizer", "Adam (lr=2e-4, betas=(0.5, 0.999))"],
        ["Image size", "128x128 (nose ROI)"],
        ["Data augmentation", "Synchronized flip + color jitter"],
    ]
    story.append(make_table(config[0], config[1:], col_widths=[140, 320], font=font))

    # 6. Results
    story.append(Paragraph("6. Results", s["h1"]))
    story.append(Paragraph("6.1 Training Curves", s["h2"]))
    add_image(story, ASSETS / "training_curves.jpg", 165,
              "Figure 4: Validation L1 loss over 50 epochs (orange dashed = best epoch)", s)

    story.append(Paragraph("6.2 Model Performance", s["h2"]))
    models_data = [["Model", "Mode", "Size", "Epochs", "Best Val L1"]]
    for m in ["autoencoder_nose", "pix2pix_nose", "cyclegan_nose", "diffusion_nose"]:
        try:
            meta = json.loads(open(f"/Applications/CS31/models/outcome/{m}/metadata.json").read())
            models_data.append([
                m.replace("_nose", "").title(), "Nose ROI",
                str(meta["image_size"]), str(meta["epochs"]), f"{meta['best_val_l1']:.4f}"
            ])
        except:
            pass
    story.append(make_table(models_data[0], models_data[1:], col_widths=[100, 70, 50, 55, 85], font=font))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Autoencoder achieves the best validation L1 (0.0760), closely followed by CycleGAN (0.0763) "
        "and Pix2Pix (0.0771). The diffusion model, operating at lower resolution (64x64), "
        "serves as a feasibility demonstration.", s["body"]))

    # Full-face benchmark if available
    bench_path = Path("/Applications/CS31/artifacts/eval/benchmark.csv")
    if bench_path.exists():
        story.append(Paragraph("6.3 Full-Face Model Evaluation (58 test samples, FID dims=2048)", s["h2"]))
        with open(bench_path) as f:
            bench = list(csv.DictReader(f))
        bench_table = [["Model", "SSIM", "ROI SSIM", "LPIPS", "ROI LPIPS", "FID"]]
        for row in bench:
            bench_table.append([
                row["model"].title(),
                f"{float(row['ssim']):.4f}", f"{float(row['roi_ssim']):.4f}",
                f"{float(row['lpips']):.4f}", f"{float(row['roi_lpips']):.4f}",
                f"{float(row['fid']):.1f}",
            ])
        story.append(make_table(bench_table[0], bench_table[1:], col_widths=[80, 65, 65, 65, 70, 60], font=font))
    story.append(PageBreak())

    # 7. Web Application
    story.append(Paragraph("7. Web Application", s["h1"]))
    story.append(Paragraph(
        "A full-stack web application provides an interactive interface for the prediction system.", s["body"]))
    story.append(Paragraph("7.1 Backend (FastAPI)", s["h2"]))
    api = [
        ["Endpoint", "Method", "Description"],
        ["/api/predict", "POST", "Upload image, run model inference, return prediction + description + landmarks"],
        ["/api/history", "GET", "Query prediction history"],
        ["/api/benchmarks", "GET", "Return evaluation benchmark data"],
        ["/api/training-history/{model}", "GET", "Return training loss curve data"],
    ]
    story.append(make_table(api[0], api[1:], col_widths=[160, 45, 255], font=font))

    story.append(Paragraph("7.2 Frontend (React + Vite)", s["h2"]))
    tabs = [
        ["Tab", "Features"],
        ["Upload", "Model selection (4 architectures), input mode (paired/single), image upload"],
        ["Prediction", "Pre/post/generated comparison, image slider, surgical description, landmark features"],
        ["Benchmark", "Performance metrics table, training loss curves (SVG)"],
        ["Compare", "Multi-model side-by-side comparison"],
        ["About", "Project description, data pipeline summary, clinical disclaimer"],
    ]
    story.append(make_table(tabs[0], tabs[1:], col_widths=[80, 380], font=font))

    # 8. Limitations & Future
    story.append(Paragraph("8. Limitations and Future Work", s["h1"]))
    story.append(Paragraph("8.1 Current Limitations", s["h2"]))
    for item in [
        "Limited dataset size (581 training samples) may constrain generalization",
        "MediaPipe face detection succeeds on only ~7% of extreme profile views; heuristic fallback is used for the rest",
        "Diffusion model operates at 64x64 resolution, limiting output quality",
        "No real cost annotation data available for cost prediction module",
        "Expert clinical review templates generated but not yet filled by clinicians",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))

    story.append(Paragraph("8.2 Future Work", s["h2"]))
    for item in [
        "Expand dataset with additional rhinoplasty case images",
        "Train at higher resolution (256x256 or 512x512) for finer detail",
        "Integrate clinical expert review for qualitative validation",
        "Add frontal-view prediction as a separate model track",
        "Implement cost prediction when real annotation data becomes available",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("Disclaimer", s["h2"]))
    story.append(Paragraph(
        "This system is a research prototype developed for academic purposes. It must not be used for clinical "
        "decision-making or surgical planning. All predictions are illustrative only.", s["small"]))

    doc.build(story)
    print(f"English report saved to: {output}")


def build_chinese():
    s = get_styles("STSong-Light")
    font = "STSong-Light"
    output = Path("/Applications/CS31/reports/CS31_Project_Report_CN.pdf")
    doc = SimpleDocTemplate(str(output), pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm, topMargin=25*mm, bottomMargin=25*mm)
    story = []

    # Cover
    story.append(Spacer(1, 50))
    story.append(Paragraph("CS31 鼻整形术后效果预测系统", s["title"]))
    story.append(Paragraph("项目开发报告", ParagraphStyle("S2", fontName=font, fontSize=18, leading=26, alignment=TA_CENTER, textColor=DARK, spaceAfter=8)))
    story.append(HRFlowable(width="50%", thickness=1, color=PRIMARY, spaceAfter=16))
    story.append(Paragraph("基于 AI 和面部图像分析的鼻整形术后效果预测", s["subtitle"]))
    story.append(Spacer(1, 20))
    cover = [
        ["项目编号", "CS31"],
        ["技术栈", "Python / PyTorch / FastAPI / React / MediaPipe"],
        ["数据集", "2,568 张图片, 833 独立样本, 581 侧面训练样本"],
        ["模型", "4 种架构 x 鼻部 ROI (128x128), 各 50 epochs"],
    ]
    t = Table(cover, colWidths=[120, 340])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("LEADING", (0, 0), (-1, -1), 18), ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"), ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
    ]))
    story.append(t)
    story.append(PageBreak())

    # 1
    story.append(Paragraph("1. 项目概要", s["h1"]))
    story.append(Paragraph(
        "本项目开发了一套基于深度学习的鼻整形术后效果预测系统。通过分析术前面部侧面照片,"
        "利用生成式 AI 模型预测手术后的外观变化,为患者和外科医生提供直观的术后效果可视化,"
        "辅助临床决策和个性化治疗方案制定。", s["body"]))
    story.append(Paragraph(
        "核心成果: 全自动数据管线处理 2,568 张图片并智能去重; 面部关键点检测与 7 项鼻部定量测量; "
        "NLP 手术描述自动生成; 4 种深度学习模型训练(Autoencoder, Pix2Pix, CycleGAN, Diffusion); "
        "以及完整的全栈 Web 应用。", s["body"]))

    # 2
    story.append(Paragraph("2. 数据采集与预处理", s["h1"]))
    story.append(Paragraph("2.1 原始数据", s["h2"]))
    story.append(Paragraph(
        "数据集包含 2,568 张 WhatsApp 导出的鼻整形手术图片。每张原始图片为左右拼接画布,"
        "左半边为术前侧面照,右半边为术后侧面照。数据来源包括根目录文件(1,202 张)和解压子目录(1,366 张)。", s["body"]))

    story.append(Paragraph("2.2 去重", s["h2"]))
    story.append(Paragraph(
        "采用 Union-Find 并查集算法进行传递性去重,结合文件名匹配和感知哈希(pHash)相似度(阈值=4)。"
        "将数据集从 2,568 张缩减至 833 个独立样本。", s["body"]))
    dedup = [
        ["阶段", "数量", "说明"],
        ["原始扫描", "2,568", "根目录文件 + 子目录文件"],
        ["去重后", "833", "Union-Find 传递性聚类"],
        ["重复(文件名)", "1,072", "跨来源同名文件"],
        ["重复(pHash)", "663", "视觉相同或高度相似"],
    ]
    story.append(make_table(dedup[0], dedup[1:], col_widths=[100, 60, 300], font=font))

    story.append(Paragraph("2.3 视角分类与过滤", s["h2"]))
    story.append(Paragraph(
        "使用 MediaPipe FaceLandmarker 对每张图片进行面部朝向分类。项目定位为侧面到侧面预测,"
        "因此排除 252 张正脸图片,保留 581 张侧面样本用于训练。", s["body"]))
    add_image(story, ASSETS / "view_type_comparison.jpg", 155,
              "图 1: 侧面视图(上方,用于训练) vs 正面视图(下方,已排除)", s)

    story.append(Paragraph("2.4 图像切分与宽高比保持", s["h2"]))
    story.append(Paragraph(
        "每张拼接画布被切分为独立的术前和术后图像。竖版图片(如 600x1600)在缩放到 256x256 时"
        "保持原始宽高比,两侧填充黑边。正方形图片(1080x1080)改为上下切分。", s["body"]))

    story.append(Paragraph("2.5 数据样例", s["h2"]))
    add_image(story, ASSETS / "fullface_pairs_sample.jpg", 155,
              "图 2: 预处理后的全脸样本对(上: 术前, 下: 术后)", s)
    story.append(PageBreak())

    # 3
    story.append(Paragraph("3. 鼻部感兴趣区域(ROI)提取", s["h1"]))
    story.append(Paragraph(
        "鼻整形手术仅改变鼻部区域,全脸训练迫使模型学习\"保持 90% 像素不变\"的冗余任务。"
        "通过提取鼻部 ROI 裁剪,模型 100% 的学习能力集中在实际的手术变化上。", s["body"]))
    for item in [
        "基于关键点的 ROI: MediaPipe 检测到面部时,从 7 个鼻部关键点计算包围盒并扩展 30% 边距",
        "启发式回退: 对无法检测面部的极端侧面图(约 93%),使用基于图像比例的裁剪",
        "所有 ROI 统一缩放至 128x128 像素",
        "581 组术前/术后鼻部 ROI 成功提取",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))
    add_image(story, ASSETS / "nose_roi_pairs_sample.jpg", 155,
              "图 3: 用于训练的鼻部 ROI 对(上: 术前, 下: 术后)", s)

    # 4
    story.append(Paragraph("4. 面部关键点检测与 NLP", s["h1"]))
    story.append(Paragraph("4.1 鼻部特征提取", s["h2"]))
    story.append(Paragraph(
        "使用 MediaPipe 478 点面部网格,计算 7 项鼻部定量特征:", s["body"]))
    feat = [
        ["特征", "说明"],
        ["鼻梁角度", "鼻梁线相对垂直方向的夹角"],
        ["鼻尖投射比", "鼻尖水平偏移 / 鼻梁长度"],
        ["鼻翼宽度", "左右鼻翼间距"],
        ["鼻梁长度", "鼻梁顶部到鼻尖的距离"],
        ["鼻额角", "鼻梁-前额交界处的角度"],
        ["鼻唇角", "鼻底与上唇之间的角度"],
        ["对称性", "左右鼻翼对称比 (0-1)"],
    ]
    story.append(make_table(feat[0], feat[1:], col_widths=[100, 360], font=font))

    story.append(Paragraph("4.2 手术描述生成 (NLP)", s["h2"]))
    story.append(Paragraph(
        "通过比较术前和术后鼻部测量值,系统自动生成结构化手术变化描述。每个维度设有检测阈值,"
        "超过阈值的变化会生成自然语言描述,如\"鼻梁修整(角度变化 +5.6 度)\"或\"鼻尖上旋\"。"
        "输出包含变化列表、摘要和详细数值指标。", s["body"]))
    story.append(PageBreak())

    # 5
    story.append(Paragraph("5. 深度学习模型开发", s["h1"]))
    story.append(Paragraph("在 128x128 鼻部 ROI 上实现并训练了四种生成式架构:", s["body"]))
    story.append(Paragraph("5.1 Autoencoder (自编码器)", s["h2"]))
    story.append(Paragraph("基线编码器-解码器模型,使用 UNet 架构和跳跃连接。以 L1 损失进行直接像素级监督。参数量约 29.2M。", s["body"]))
    story.append(Paragraph("5.2 Pix2Pix (条件 GAN)", s["h2"]))
    story.append(Paragraph("条件对抗生成网络,Generator 损失 = BCE 对抗损失 + 100 x L1 重建损失。参数量约 36M。", s["body"]))
    story.append(Paragraph("5.3 CycleGAN (循环一致性 GAN)", s["h2"]))
    story.append(Paragraph("双向翻译模型,包含对抗损失、循环一致性损失(lambda=10)和身份保持损失(lambda=5)。参数量约 72M。", s["body"]))
    story.append(Paragraph("5.4 Diffusion (扩散模型)", s["h2"]))
    story.append(Paragraph("轻量级条件扩散模型(TinyDiffusionUNet),在 64x64 分辨率运行,100 步线性 beta 调度。参数量约 1.9M。", s["body"]))
    story.append(Paragraph("5.5 训练配置", s["h2"]))
    config = [
        ["参数", "值"],
        ["训练样本", "465 (581 侧面样本的 80%)"],
        ["验证样本", "58 (10%)"],
        ["测试样本", "58 (10%)"],
        ["训练轮数", "每个模型 50 epochs"],
        ["优化器", "Adam (lr=2e-4, betas=(0.5, 0.999))"],
        ["图像尺寸", "128x128 (鼻部 ROI)"],
        ["数据增强", "同步翻转 + 颜色抖动"],
    ]
    story.append(make_table(config[0], config[1:], col_widths=[120, 340], font=font))

    # 6
    story.append(Paragraph("6. 实验结果", s["h1"]))
    story.append(Paragraph("6.1 训练曲线", s["h2"]))
    add_image(story, ASSETS / "training_curves.jpg", 165,
              "图 4: 验证 L1 损失随 epoch 变化(橙色虚线 = 最佳 epoch)", s)
    story.append(Paragraph("6.2 模型性能", s["h2"]))
    models_data = [["模型", "模式", "尺寸", "轮数", "最佳验证 L1"]]
    for m in ["autoencoder_nose", "pix2pix_nose", "cyclegan_nose", "diffusion_nose"]:
        try:
            meta = json.loads(open(f"/Applications/CS31/models/outcome/{m}/metadata.json").read())
            models_data.append([
                m.replace("_nose", "").title(), "鼻部 ROI",
                str(meta["image_size"]), str(meta["epochs"]), f"{meta['best_val_l1']:.4f}"
            ])
        except:
            pass
    story.append(make_table(models_data[0], models_data[1:], col_widths=[100, 65, 45, 45, 85], font=font))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Autoencoder 取得最佳验证 L1(0.0760),CycleGAN(0.0763)和 Pix2Pix(0.0771)紧随其后。"
        "Diffusion 模型在 64x64 低分辨率运行,作为可行性验证。", s["body"]))

    bench_path = Path("/Applications/CS31/artifacts/eval/benchmark.csv")
    if bench_path.exists():
        story.append(Paragraph("6.3 全脸模型评估 (58 测试样本, FID dims=2048)", s["h2"]))
        with open(bench_path) as f:
            bench = list(csv.DictReader(f))
        bench_table = [["模型", "SSIM", "ROI SSIM", "LPIPS", "ROI LPIPS", "FID"]]
        for row in bench:
            bench_table.append([
                row["model"].title(),
                f"{float(row['ssim']):.4f}", f"{float(row['roi_ssim']):.4f}",
                f"{float(row['lpips']):.4f}", f"{float(row['roi_lpips']):.4f}",
                f"{float(row['fid']):.1f}",
            ])
        story.append(make_table(bench_table[0], bench_table[1:], col_widths=[80, 65, 65, 65, 70, 60], font=font))
    story.append(PageBreak())

    # 7
    story.append(Paragraph("7. Web 应用系统", s["h1"]))
    story.append(Paragraph("完整的前后端 Web 应用为预测系统提供交互界面。", s["body"]))
    story.append(Paragraph("7.1 后端 (FastAPI)", s["h2"]))
    api = [
        ["端点", "方法", "功能"],
        ["/api/predict", "POST", "上传图片, 运行推理, 返回预测结果+描述+关键点"],
        ["/api/history", "GET", "查询推理历史记录"],
        ["/api/benchmarks", "GET", "返回评估基准数据"],
        ["/api/training-history/{model}", "GET", "返回训练 loss 曲线数据"],
    ]
    story.append(make_table(api[0], api[1:], col_widths=[160, 40, 260], font=font))
    story.append(Paragraph("7.2 前端 (React + Vite)", s["h2"]))
    tabs = [
        ["页面", "功能"],
        ["Upload", "模型选择(4种架构), 输入模式(配对/单图), 图片上传"],
        ["Prediction", "术前/术后/生成图对比, 滑动对比器, 手术描述, 关键点特征"],
        ["Benchmark", "性能指标表格, 训练 loss 曲线"],
        ["Compare", "多模型并排对比视图"],
        ["About", "项目说明, 数据管线摘要, 临床免责声明"],
    ]
    story.append(make_table(tabs[0], tabs[1:], col_widths=[80, 380], font=font))

    # 8
    story.append(Paragraph("8. 局限性与未来工作", s["h1"]))
    story.append(Paragraph("8.1 当前局限性", s["h2"]))
    for item in [
        "数据集规模有限(581 个训练样本),可能限制模型泛化能力",
        "MediaPipe 面部检测在极端侧面视图上成功率仅约 7%,其余使用启发式回退",
        "Diffusion 模型在 64x64 分辨率运行,输出质量有限",
        "无真实费用标注数据,未实现费用预测模块",
        "专家临床评审模板已生成但尚未由医生填写",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))
    story.append(Paragraph("8.2 未来工作", s["h2"]))
    for item in [
        "扩充数据集,增加更多鼻整形案例图片",
        "在更高分辨率(256x256 或 512x512)上训练以提升细节",
        "引入临床专家评审进行定性验证",
        "添加正面视角预测作为独立模型通道",
        "待真实标注数据可用后实现费用预测模块",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s["bullet"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("免责声明", s["h2"]))
    story.append(Paragraph(
        "本系统为学术研究原型,不得用于临床医疗决策或手术规划。所有预测结果仅供参考。", s["small"]))

    doc.build(story)
    print(f"Chinese report saved to: {output}")


if __name__ == "__main__":
    build_english()
    build_chinese()
