import cv2
import numpy as np
import os

def crop_horizontal_whitespace(gray_img, threshold=250):
    col_means = cv2.reduce(gray_img, 0, cv2.REDUCE_AVG).flatten()
    dark_cols = np.where(col_means < threshold)[0]
    if len(dark_cols) == 0:
        return 0, gray_img.shape[1]
    left, right = dark_cols.min(), dark_cols.max()
    return left-15, right

def find_staff_lines(gray_img, left_crop=0, right_crop=None, darkness_threshold=130, max_line_thickness=4):
    if right_crop is None:
        right_crop = gray_img.shape[1]
    cropped = gray_img[:, left_crop:right_crop]
    row_avgs = np.mean(cropped, axis=1)
    dark_rows = np.where(row_avgs < darkness_threshold)[0]

    lines = []
    start = None
    prev = None
    for r in dark_rows:
        if start is None:
            start = r
            prev = r
        elif r <= prev + 1 and (r - start) < max_line_thickness:
            prev = r
        else:
            lines.append((start, prev))
            start = r
            prev = r
    if start is not None:
        lines.append((start, prev))

    staff_line_positions = [(start + end) // 2 for start, end in lines]
    return staff_line_positions

def group_staff_lines(staff_lines, lines_per_system=5, max_line_spacing=50, min_lines_per_system=3):
    systems = []
    staff_lines = sorted(staff_lines)
    temp_group = [staff_lines[0]]

    for line in staff_lines[1:]:
        if line - temp_group[-1] <= max_line_spacing:
            temp_group.append(line)
        else:
            if len(temp_group) >= min_lines_per_system:
                systems.append((temp_group[0] - 35, temp_group[-1] + 35))
            temp_group = [line]

    if len(temp_group) >= min_lines_per_system:
        systems.append((temp_group[0] - 35, temp_group[-1] + 35))

    return systems

def export_cropped_systems(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    left, right = crop_horizontal_whitespace(gray)
    cropped_gray = gray[:, left:right]

    staff_lines = find_staff_lines(cropped_gray)
    systems = group_staff_lines(staff_lines)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, (top, bottom) in enumerate(systems):
        system_crop = gray[top:bottom, left:right]
        output_path = os.path.join(output_dir, f"{base_name}_system_{idx+1}.png")
        cv2.imwrite(output_path, system_crop)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_dir = r"C:\Desktop\MMV2\TrainingData\Single Lined New PNGs"  # Your folder path
    output_dir = "Single_Lined_Cropped"

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            export_cropped_systems(image_path, output_dir)
