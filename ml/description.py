"""Generate surgical change descriptions by comparing pre/post nose features."""

from dataclasses import dataclass
from typing import List, Optional

from .landmarks import NoseFeatures, detect_landmarks

from PIL import Image


@dataclass
class SurgeryDescription:
    changes: List[str]
    summary: str
    detail_metrics: dict


def _compare_feature(pre_val: float, post_val: float, threshold: float) -> str:
    """Return 'increased', 'decreased', or 'unchanged'."""
    diff = post_val - pre_val
    if abs(diff) < threshold:
        return "unchanged"
    return "increased" if diff > 0 else "decreased"


def describe_changes(pre_features: NoseFeatures, post_features: NoseFeatures) -> SurgeryDescription:
    """Compare pre and post nose features and generate descriptions."""
    changes: List[str] = []
    metrics = {}

    # Bridge angle change
    angle_diff = post_features.bridge_angle - pre_features.bridge_angle
    metrics["bridge_angle_change"] = round(angle_diff, 2)
    if abs(angle_diff) > 2.0:
        direction = "refined" if angle_diff > 0 else "straightened"
        changes.append(f"Nasal bridge {direction} (angle change: {angle_diff:+.1f} degrees)")

    # Tip projection
    proj_diff = post_features.tip_projection - pre_features.tip_projection
    metrics["tip_projection_change"] = round(proj_diff, 4)
    if abs(proj_diff) > 0.01:
        direction = "increased" if proj_diff > 0 else "reduced"
        changes.append(f"Nasal tip projection {direction}")

    # Ala width
    ala_diff = post_features.ala_width - pre_features.ala_width
    metrics["ala_width_change"] = round(ala_diff, 4)
    if abs(ala_diff) > 0.005:
        direction = "widened" if ala_diff > 0 else "narrowed"
        changes.append(f"Nasal ala {direction}")

    # Nasofrontal angle
    nf_diff = post_features.nasofrontal_angle - pre_features.nasofrontal_angle
    metrics["nasofrontal_angle_change"] = round(nf_diff, 2)
    if abs(nf_diff) > 3.0:
        direction = "opened" if nf_diff > 0 else "sharpened"
        changes.append(f"Nasofrontal angle {direction} ({nf_diff:+.1f} degrees)")

    # Nasolabial angle
    nl_diff = post_features.nasolabial_angle - pre_features.nasolabial_angle
    metrics["nasolabial_angle_change"] = round(nl_diff, 2)
    if abs(nl_diff) > 3.0:
        if nl_diff > 0:
            changes.append(f"Nasal tip rotated upward (nasolabial angle +{nl_diff:.1f} degrees)")
        else:
            changes.append(f"Nasal tip rotated downward (nasolabial angle {nl_diff:.1f} degrees)")

    # Bridge length
    bl_diff = post_features.bridge_length - pre_features.bridge_length
    metrics["bridge_length_change"] = round(bl_diff, 4)
    if abs(bl_diff) > 0.01:
        direction = "lengthened" if bl_diff > 0 else "shortened"
        changes.append(f"Nasal bridge {direction}")

    # Symmetry
    sym_diff = post_features.symmetry_score - pre_features.symmetry_score
    metrics["symmetry_change"] = round(sym_diff, 4)
    if sym_diff > 0.05:
        changes.append("Improved nasal symmetry")
    elif sym_diff < -0.05:
        changes.append("Reduced nasal symmetry")

    if not changes:
        changes.append("No significant structural changes detected")

    # Generate summary
    if len(changes) == 1 and "No significant" in changes[0]:
        summary = "Minimal visible structural change between pre-operative and post-operative images."
    else:
        summary = f"Detected {len(changes)} structural change(s) in the nasal region."

    return SurgeryDescription(changes=changes, summary=summary, detail_metrics=metrics)


def generate_description(pre_image: Image.Image, post_image: Image.Image) -> Optional[SurgeryDescription]:
    """Full pipeline: detect landmarks on both images, compare, and describe."""
    pre_result = detect_landmarks(pre_image)
    post_result = detect_landmarks(post_image)

    if not pre_result.face_detected or not post_result.face_detected:
        return None
    if pre_result.nose_features is None or post_result.nose_features is None:
        return None

    return describe_changes(pre_result.nose_features, post_result.nose_features)
