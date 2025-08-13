from flask import Flask, request, send_file, make_response
import cv2
import numpy as np
import io

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

def crop_horizontal_whitespace(gray_img, threshold=230):
    # Average pixel intensity per column
    col_means = cv2.reduce(gray_img, 0, cv2.REDUCE_AVG).flatten()
    dark_cols = np.where(col_means < threshold)[0]
    if len(dark_cols) == 0:
        return 0, gray_img.shape[1]
    left, right = dark_cols.min(), dark_cols.max()
    return left, right

def find_staff_lines(gray_img, left_crop=0, right_crop=None, darkness_threshold=130, max_line_thickness=4):
    if right_crop is None:
        right_crop = gray_img.shape[1]

    cropped = gray_img[:, left_crop:right_crop]
    height, width = cropped.shape

    # Average pixel value per row
    row_avgs = np.mean(cropped, axis=1)

    # Rows darker than threshold (candidate staff line rows)
    dark_rows = np.where(row_avgs < darkness_threshold)[0]

    # Group consecutive dark rows into lines (staff lines can be a few pixels thick)
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

    # Take middle of each line group as staff line position
    staff_line_positions = [(start + end) // 2 for start, end in lines]

    return staff_line_positions


def group_staff_lines(staff_lines, lines_per_system=5, max_line_spacing=50, min_lines_per_system=3):
    """
    Group detected staff lines into systems.
    Allows groups with at least min_lines_per_system lines.
    """
    systems = []
    staff_lines = sorted(staff_lines)
    temp_group = [staff_lines[0]]

    for line in staff_lines[1:]:
        if line - temp_group[-1] <= max_line_spacing:
            temp_group.append(line)
        else:
            # Close current group and start new
            if len(temp_group) >= min_lines_per_system:
                systems.append((temp_group[0], temp_group[-1]))  # small padding
            temp_group = [line]

    # Check last group
    if len(temp_group) >= min_lines_per_system:
        systems.append((temp_group[0] - 5, temp_group[-1] + 5))

    return systems




def find_system_bounds(gray_img, white_threshold=245, min_gap_height=15, min_system_height=30):
    gaps = find_staff_lines(gray_img, white_threshold, min_gap_height)
    height = gray_img.shape[0]

    # Use mid points of gaps as boundaries
    boundaries = [0]
    for start, end in gaps:
        boundaries.append((start + end) // 2)
    boundaries.append(height)

    systems = []
    for i in range(len(boundaries) - 1):
        top, bottom = boundaries[i], boundaries[i+1]
        if bottom - top >= min_system_height:
            systems.append((top, bottom))
    return systems

def detect_barlines(gray_img, top, bottom, darkness_threshold=200, min_coverage=0.8, group_dist=3, darkness_tolerance=15):
    # Work only on the system slice
    system_slice = gray_img[top:bottom, :]

    # Average intensity per column
    col_means = cv2.reduce(system_slice, 0, cv2.REDUCE_AVG).flatten()
    dark_cols = np.where(col_means < darkness_threshold)[0]

    if len(dark_cols) == 0:
        return []

    # Group close dark columns together into bar candidates
    groups = []
    current_group = [dark_cols[0]]
    for x in dark_cols[1:]:
        if x - current_group[-1] <= group_dist:
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]
    groups.append(current_group)

    candidates = []
    for group in groups:
        x_center = int(np.mean(group))
        col_pixels = system_slice[:, x_center]
        dark_pixels = np.where(col_pixels < darkness_threshold)[0]

        coverage = len(dark_pixels) / system_slice.shape[0]
        if coverage >= min_coverage:
            avg_darkness = np.mean(col_pixels[dark_pixels])
            candidates.append((x_center, avg_darkness))

    if not candidates:
        return []

    # Filter candidates by similar darkness to median
    median_darkness = np.median([c[1] for c in candidates])
    filtered = [c for c in candidates if abs(c[1] - median_darkness) <= darkness_tolerance]

    # Return sorted x positions of filtered candidates
    return sorted([c[0] for c in filtered])

@app.route('/upload', methods=['POST'])
def process_image():
    import sys

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop horizontal whitespace on left/right margins
    left, right = crop_horizontal_whitespace(gray)
    cropped_gray = gray[:, left:right]

    # Detect staff lines on cropped_gray
    staff_lines = find_staff_lines(cropped_gray, left_crop=0, right_crop=right - left)

    # Group staff lines into systems
    systems = group_staff_lines(staff_lines)

    measure_count = 0

    # Draw crop boundary lines for debugging (cyan)
    cv2.line(image, (left, 0), (left, image.shape[0] - 1), (0, 255, 255), 2)
    cv2.line(image, (right, 0), (right, image.shape[0] - 1), (0, 255, 255), 2)

    # Draw staff lines (red) for debugging
    for y in staff_lines:
        cv2.line(image, (left, y), (right, y), (0, 0, 255), 1)

    # Draw system bounds and detect barlines
    for idx, (top, bottom) in enumerate(systems):
        # Draw system boundaries (red top, blue bottom)
        cv2.line(image, (left, top), (right, top), (0, 0, 255), 2)
        cv2.line(image, (left, bottom), (right, bottom), (255, 0, 0), 2)

        # Detect barlines within this system on cropped grayscale image
        bars_x = detect_barlines(cropped_gray, top, bottom)

        print(f"[DEBUG] System {idx + 1} from {top} to {bottom} found {len(bars_x)} bars", file=sys.stderr)

        for i, x in enumerate(bars_x):
            measure_num = measure_count + i + 1
            # Draw measure number on original image, adjust x by left crop offset
            cv2.putText(image, str(measure_num), (x + left - 10, top + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw orange vertical barline
            cv2.line(image, (x + left, top), (x + left, bottom), (0, 165, 255), 2)  # Orange color

        measure_count += len(bars_x)

    # Encode and send processed image
    _, buffer = cv2.imencode('.png', image)
    io_buf = io.BytesIO(buffer)
    response = make_response(send_file(io_buf, mimetype='image/png', as_attachment=False, download_name='processed.png'))
    response.headers['X-Measure-Count'] = str(measure_count)
    return response



if __name__ == '__main__':
    app.run(debug=True)
