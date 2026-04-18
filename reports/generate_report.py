"""Generate today's work report as PDF."""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

FONT = "STSong-Light"
PRIMARY = HexColor("#0d5c63")
DARK = HexColor("#1b1a18")
GRAY = HexColor("#4b4c51")
LIGHT_BG = HexColor("#f0f4f8")
WHITE = HexColor("#ffffff")
GREEN = HexColor("#2e7d32")
RED = HexColor("#c62828")

style_title = ParagraphStyle("Title", fontName=FONT, fontSize=22, leading=30, alignment=TA_CENTER, textColor=PRIMARY, spaceAfter=6)
style_subtitle = ParagraphStyle("Sub", fontName=FONT, fontSize=12, leading=18, alignment=TA_CENTER, textColor=GRAY, spaceAfter=20)
style_h1 = ParagraphStyle("H1", fontName=FONT, fontSize=15, leading=22, textColor=PRIMARY, spaceBefore=16, spaceAfter=8)
style_h2 = ParagraphStyle("H2", fontName=FONT, fontSize=12, leading=18, textColor=DARK, spaceBefore=12, spaceAfter=6)
style_body = ParagraphStyle("Body", fontName=FONT, fontSize=10.5, leading=17, textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=6)
style_bullet = ParagraphStyle("Bullet", fontName=FONT, fontSize=10.5, leading=17, textColor=DARK, leftIndent=18, bulletIndent=6, spaceAfter=4)
style_done = ParagraphStyle("Done", fontName=FONT, fontSize=10.5, leading=17, textColor=GREEN, leftIndent=18, bulletIndent=6, spaceAfter=4)


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
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


def build():
    output_path = Path("/Applications/CS31/reports/今日工作报告.pdf")
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm, topMargin=25*mm, bottomMargin=25*mm)
    story = []

    # ===== COVER =====
    story.append(Spacer(1, 40))
    story.append(Paragraph("CS31 鼻整形术后效果预测系统", style_title))
    story.append(Paragraph("今日工作报告", ParagraphStyle("S2", fontName=FONT, fontSize=18, leading=26, alignment=TA_CENTER, textColor=DARK, spaceAfter=8)))
    story.append(HRFlowable(width="50%", thickness=1, color=PRIMARY, spaceAfter=16))
    story.append(Paragraph("日期: 2026 年 3 月 29 日", style_subtitle))
    story.append(Spacer(1, 20))

    # Quick summary
    story.append(Paragraph("工作概要", style_h1))
    story.append(Paragraph(
        "今日完成了 CS31 鼻整形术后效果预测项目的全面开发工作,包括: 项目现状审查与缺陷诊断、"
        "数据管线 4 个核心 Bug 修复、面部关键点检测模块开发、NLP 手术描述生成模块开发、"
        "鼻部 ROI 专项训练管线搭建、正脸图片检测与过滤、数据增强集成、"
        "8 个深度学习模型的完整训练(各 50 epochs)、全测试集评估、后端 API 扩展以及前端功能增强。", style_body))
    story.append(PageBreak())

    # ===== TASK 1: REVIEW =====
    story.append(Paragraph("一、项目现状审查与缺陷诊断", style_h1))
    story.append(Paragraph(
        "首先对项目全部源代码文件进行了完整审查,对照项目需求文档逐项检查完成度,"
        "发现以下 7 个核心缺陷:", style_body))
    defects = [
        ["#", "缺陷", "严重程度", "今日处理"],
        ["1", "模型只训练了 1 epoch / 16 样本,等于没训练", "严重", "已修复"],
        ["2", "Facial Landmark Detection 完全缺失", "严重", "已修复"],
        ["3", "Cost Prediction 完全缺失", "中等", "跳过(无真实数据)"],
        ["4", "NLP 模块完全缺失", "严重", "已修复"],
        ["5", "数据预处理无增强、无质量检查", "中等", "已修复"],
        ["6", "评估只用 4 个测试样本,FID dims=64", "严重", "已修复"],
        ["7", "Web 应用功能薄弱", "中等", "已修复"],
    ]
    story.append(make_table(defects[0], defects[1:], col_widths=[20, 230, 60, 80]))
    story.append(Paragraph("最终修复了 6/7 个缺陷,Cost Prediction 因无真实费用标注数据经确认跳过。", style_body))
    story.append(PageBreak())

    # ===== TASK 2: DATA PIPELINE =====
    story.append(Paragraph("二、数据管线 Bug 修复", style_h1))
    story.append(Paragraph("独立审查了 2568 张原始图片,发现并修复了 4 个数据管线 Bug:", style_body))

    story.append(Paragraph("Bug 1: 子目录不被扫描", style_h2))
    story.append(Paragraph(
        "原因: 用户将 ZIP 压缩包解压后删除了压缩包,但代码只扫描根目录文件和 ZIP,不递归子目录。"
        "修复: iter_source_entries() 新增 directory 分支,支持递归扫描子目录中的图片文件。"
        "同步更新 load_image_bytes() 支持从子目录加载图片。", style_body))

    story.append(Paragraph("Bug 2: 去重算法非传递性", style_h2))
    story.append(Paragraph(
        "原因: 原算法顺序扫描,A 和 B 相似时标记 B 为重复,但 C 和 A 相似时由于 A 已被移出候选列表,"
        "C 被保留为 canonical,遗漏了传递性重复。"
        "修复: 使用 Union-Find 并查集实现传递性聚类去重,canonical 数量从 855 降至 833。", style_body))

    story.append(Paragraph("Bug 3: 正方形图片切分错误", style_h2))
    story.append(Paragraph(
        "原因: 8 张 1080x1080 正方形图片实际是上下排列(上=术前,下=术后),但代码一律左右切分。"
        "修复: split_paired_image() 对 width==height 的图片改为上下切分。", style_body))

    story.append(Paragraph("Bug 4: 无质量检查", style_h2))
    story.append(Paragraph(
        "修复: 新增 validate_split_halves() 函数,检测空白/纯色图片(像素标准差 < 2.55)并跳过。"
        "全部 833 个样本均通过检查。", style_body))

    story.append(Paragraph("新增 6 个单元测试,全部通过。", style_body))

    fix_data = [
        ["修复项", "修复前", "修复后"],
        ["子目录扫描", "只扫描根目录 1202 张", "根目录+子目录共 2568 张"],
        ["去重算法", "顺序扫描, canonical=855", "Union-Find, canonical=833"],
        ["正方形切分", "左右切分(错误)", "上下切分(正确)"],
        ["质量检查", "无", "像素标准差检测"],
    ]
    story.append(make_table(fix_data[0], fix_data[1:], col_widths=[100, 180, 180]))
    story.append(PageBreak())

    # ===== TASK 3: FRONTAL FILTER =====
    story.append(Paragraph("三、正脸图片检测与过滤", style_h1))
    story.append(Paragraph(
        "使用 MediaPipe FaceLandmarker 对 833 个 canonical 样本逐一检测面部朝向。"
        "通过计算双眼间距和鼻尖偏移量判断侧面/正面。生成了 50 张正脸样本的网格图供人工确认后,"
        "将 252 张正脸标记为 frontal 并排除出训练集。", style_body))
    story.append(Paragraph(
        "排除原因: 项目定位为 Profile-to-Profile 预测,正脸与侧面的鼻部变化模式完全不同,"
        "混入训练会干扰模型学习。manifest.csv 新增 view_type 字段, splits.csv 只包含非 frontal 样本。", style_body))

    filter_data = [
        ["类别", "数量", "处理"],
        ["Profile (侧面)", "266", "参与训练"],
        ["Unknown (极端角度)", "315", "参与训练"],
        ["Frontal (正面)", "252", "排除"],
        ["最终训练可用", "581", "Train 465 / Val 58 / Test 58"],
    ]
    story.append(make_table(filter_data[0], filter_data[1:], col_widths=[130, 60, 270]))

    # ===== TASK 4: LANDMARKS =====
    story.append(Paragraph("四、面部关键点检测模块", style_h1))
    story.append(Paragraph("新建 ml/landmarks.py, 实现以下功能:", style_body))
    for item in [
        "MediaPipe FaceLandmarker 封装: 478 个三维面部关键点检测",
        "视角分类: classify_view() 自动判断侧面/正面",
        "鼻部特征提取: 7 个定量指标(鼻梁角度、鼻尖投射比、鼻翼宽度、鼻梁长度、鼻额角、鼻唇角、对称性)",
        "动态 ROI: 基于关键点包围盒计算鼻部感兴趣区域,替代硬编码像素坐标",
        "可视化: draw_landmarks_on_image() 在图片上叠加关键点和 ROI 框",
        "批量检测: batch_detect_view_types() 单实例批量处理,避免重复创建 landmarker",
    ]:
        story.append(Paragraph(f"\u2022 {item}", style_bullet))
    story.append(PageBreak())

    # ===== TASK 5: NLP =====
    story.append(Paragraph("五、NLP 手术描述生成模块", style_h1))
    story.append(Paragraph("新建 ml/description.py, 实现基于 landmark 测量差异的手术描述自动生成:", style_body))
    nlp_data = [
        ["检测维度", "阈值", "描述示例"],
        ["鼻梁角度", "> 2 度", "Nasal bridge refined (+5.6 degrees)"],
        ["鼻尖投射", "> 0.01", "Nasal tip projection increased"],
        ["鼻翼宽度", "> 0.005", "Nasal ala narrowed"],
        ["鼻额角", "> 3 度", "Nasofrontal angle opened"],
        ["鼻唇角", "> 3 度", "Nasal tip rotated upward"],
        ["鼻梁长度", "> 0.01", "Nasal bridge shortened"],
        ["对称性", "> 0.05", "Improved nasal symmetry"],
    ]
    story.append(make_table(nlp_data[0], nlp_data[1:], col_widths=[80, 60, 320]))
    story.append(Paragraph(
        "输出包含 changes(具体变化列表)、summary(总结)和 detail_metrics(精确数值)。"
        "已集成到后端 /api/predict 接口,前端 Prediction 页面自动展示。", style_body))

    # ===== TASK 6: DATA AUGMENTATION =====
    story.append(Paragraph("六、数据增强", style_h1))
    story.append(Paragraph("修改 ml/data.py, 训练阶段新增同步数据增强:", style_body))
    for item in [
        "随机水平翻转: pre 和 post 同步翻转,保持对应关系",
        "颜色抖动: 亮度 +/-10%, 对比度 +/-10%, 饱和度 +/-5%",
        "同一组随机参数同时应用于 pre 和 post,避免色彩不一致",
        "验证集和测试集不做增强",
    ]:
        story.append(Paragraph(f"\u2022 {item}", style_bullet))
    story.append(PageBreak())

    # ===== TASK 7: NOSE ROI =====
    story.append(Paragraph("七、鼻部 ROI 专项训练", style_h1))
    story.append(Paragraph(
        "核心思路: 鼻整形只改变鼻部区域,全脸模型需要学习\"保持 90% 像素不变\","
        "而鼻部 ROI 模型将 100% 学习能力集中在鼻部变化上。", style_body))
    story.append(Paragraph("新建 ml/nose_roi.py, 实现:", style_body))
    for item in [
        "get_nose_roi_box(): landmark 检测 + 启发式回退的鼻部 ROI 提取",
        "extract_nose_roi(): 提取并统一缩放到 128x128",
        "paste_nose_back(): 羽化混合贴回(高斯模糊渐变蒙版消除接缝)",
        "prepare_nose_rois(): 批量提取 581 个样本的鼻部 ROI",
        "NoseROIDataset: 加载预提取 ROI 的 PyTorch Dataset",
    ]:
        story.append(Paragraph(f"\u2022 {item}", style_bullet))

    eff_data = [
        ["对比项", "全脸 (256x256)", "鼻部 ROI (128x128)"],
        ["像素数", "65,536", "16,384 (1/4)"],
        ["CycleGAN 单 epoch", "~75 秒", "~18 秒"],
        ["50 epochs 总时间", "~60-90 分钟", "~10-15 分钟"],
    ]
    story.append(make_table(eff_data[0], eff_data[1:], col_widths=[130, 165, 165]))

    # ===== TASK 8: TRAINING =====
    story.append(Paragraph("八、模型训练", style_h1))
    story.append(Paragraph("训练了 8 个模型: 4 个全脸(256x256) + 4 个鼻部 ROI(128x128), 各 50 epochs:", style_body))

    models_data = [["模型", "模式", "尺寸", "Epochs", "Best Val L1"]]
    for m in ["autoencoder", "pix2pix", "cyclegan", "diffusion",
              "autoencoder_nose", "pix2pix_nose", "cyclegan_nose", "diffusion_nose"]:
        try:
            meta = json.loads(open(f"/Applications/CS31/models/outcome/{m}/metadata.json").read())
            mode = "鼻部 ROI" if "_nose" in m else "全脸"
            models_data.append([m, mode, str(meta["image_size"]), str(meta["epochs"]), f"{meta['best_val_l1']:.4f}"])
        except:
            pass
    story.append(make_table(models_data[0], models_data[1:], col_widths=[120, 65, 40, 50, 80]))
    story.append(Paragraph("全部使用 465 个训练样本, 58 个验证样本, Adam 优化器, lr=2e-4。", style_body))
    story.append(PageBreak())

    # ===== TASK 9: EVALUATION =====
    story.append(Paragraph("九、完整评估", style_h1))
    story.append(Paragraph("在 58 个测试样本上运行全脸模型评估, FID 使用标准 dims=2048:", style_body))

    import csv
    bench_path = Path("/Applications/CS31/artifacts/eval/benchmark.csv")
    if bench_path.exists():
        with open(bench_path) as f:
            bench = list(csv.DictReader(f))
        bench_table = [["模型", "样本", "SSIM", "ROI SSIM", "LPIPS", "ROI LPIPS", "FID"]]
        for row in bench:
            bench_table.append([
                row["model"], row["sample_count"],
                f"{float(row['ssim']):.4f}", f"{float(row['roi_ssim']):.4f}",
                f"{float(row['lpips']):.4f}", f"{float(row['roi_lpips']):.4f}",
                f"{float(row['fid']):.1f}",
            ])
        story.append(make_table(bench_table[0], bench_table[1:], col_widths=[80, 40, 55, 60, 55, 65, 50]))

    story.append(Paragraph("与修复前的对比:", style_h2))
    compare_data = [
        ["指标", "修复前 (1 epoch, 4 样本)", "修复后 (50 epochs, 58 样本)", "改善"],
        ["最佳 SSIM", "0.44", "0.87", "2 倍"],
        ["最佳 LPIPS", "0.69", "0.13", "5 倍"],
        ["测试样本数", "4", "58", "14.5 倍"],
        ["训练样本数", "16", "465", "29 倍"],
        ["训练轮数", "1 epoch", "50 epochs", "50 倍"],
    ]
    story.append(make_table(compare_data[0], compare_data[1:], col_widths=[90, 140, 140, 50]))

    # ===== TASK 10: WEB =====
    story.append(Paragraph("十、后端与前端增强", style_h1))
    story.append(Paragraph("后端新增/修改:", style_h2))
    for item in [
        "POST /api/predict 响应新增 description(手术描述)和 landmarks(关键点数据)字段",
        "GET /api/training-history/{model} 新增端点,返回训练 loss 曲线数据",
        "backend/inference.py 集成 landmark 检测和描述生成",
        "backend/schemas.py 扩展 PredictResponse 模型",
    ]:
        story.append(Paragraph(f"\u2022 {item}", style_bullet))
    story.append(Paragraph("前端新增/修改:", style_h2))
    for item in [
        "Prediction 页: 手术描述面板(变化列表+摘要+详细指标) + Landmark 特征展示(view_type + 7 项数值)",
        "Benchmark 页: 新增 SVG Loss 曲线图,展示每个模型的训练/验证损失变化",
        "新增 Compare 页: 多模型并排对比视图",
        "About 页: 更新项目说明,反映正脸过滤和鼻部 ROI 训练",
        "新增 CSS: description-card, feature-grid, feature-chip, loss-chart 等样式",
    ]:
        story.append(Paragraph(f"\u2022 {item}", style_bullet))
    story.append(PageBreak())

    # ===== SUMMARY =====
    story.append(Paragraph("十一、今日工作汇总", style_h1))

    summary_data = [
        ["类别", "工作项", "产出"],
        ["审查", "项目缺陷诊断", "7 个缺陷, 评分表"],
        ["数据管线", "4 个 Bug 修复 + 6 个新测试", "manifest 833 canonical"],
        ["正脸过滤", "MediaPipe 视角检测", "252 张正脸排除, 581 可用"],
        ["Landmark", "新建 ml/landmarks.py", "478 点检测 + 7 项鼻部特征"],
        ["NLP", "新建 ml/description.py", "7 维手术描述生成"],
        ["数据增强", "修改 ml/data.py", "同步翻转 + 颜色抖动"],
        ["鼻部 ROI", "新建 ml/nose_roi.py + NoseROIDataset", "581 个 ROI 提取, 训练速度 4 倍"],
        ["模型训练", "8 个模型 x 50 epochs", "全脸最佳 val L1=0.0607"],
        ["评估", "58 样本, FID dims=2048", "SSIM 0.87, LPIPS 0.13"],
        ["后端", "API 扩展", "3 个端点修改/新增"],
        ["前端", "功能增强", "手术描述 + 曲线 + 对比页"],
    ]
    story.append(make_table(summary_data[0], summary_data[1:], col_widths=[70, 200, 190]))

    story.append(Spacer(1, 16))
    story.append(Paragraph("修改/新建的文件:", style_h2))
    files_data = [
        ["文件", "操作", "说明"],
        ["ml/dataset_tools.py", "修改", "Union-Find 去重 + 子目录扫描 + square 切分 + 质量验证"],
        ["ml/landmarks.py", "新建", "MediaPipe 面部关键点检测和鼻部特征提取"],
        ["ml/description.py", "新建", "NLP 手术描述生成"],
        ["ml/nose_roi.py", "新建", "鼻部 ROI 提取和贴回"],
        ["ml/data.py", "修改", "数据增强 + 正脸过滤 + NoseROIDataset"],
        ["ml/prepare_pairs.py", "修改", "集成 view_type 检测"],
        ["ml/train_outcome.py", "修改", "启用增强 + --nose-only 参数"],
        ["ml/evaluate_outcome.py", "修改", "landmark ROI + FID dims=2048"],
        ["ml/runtime.py", "修改", "支持 _nose 后缀模型名"],
        ["backend/inference.py", "修改", "集成 landmark + 描述"],
        ["backend/schemas.py", "修改", "扩展响应字段"],
        ["backend/serve.py", "修改", "新增 training-history API"],
        ["frontend/src/App.jsx", "修改", "手术描述 + landmark + 曲线 + Compare"],
        ["frontend/src/styles.css", "修改", "新增样式"],
        ["requirements.txt", "修改", "添加 mediapipe, scikit-learn"],
        ["tests/test_dataset_tools.py", "修改", "新增 6 个测试"],
    ]
    story.append(make_table(files_data[0], files_data[1:], col_widths=[140, 40, 280]))

    doc.build(story)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    build()
