import os
import math
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Configuration parameters
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "camouflaged_object_image"
# Target output size
TARGET_SIZE = 1024
# Supersampling factor (4x for smoothness)
SAMPLING_FACTOR = 4

# Actual drawing canvas size
IMG_SIZE = TARGET_SIZE * SAMPLING_FACTOR
CENTER = (TARGET_SIZE // 2, TARGET_SIZE // 2)
RADIUS = 400  # Logical radius
STROKE_WIDTH = 40  # Logical stroke width

BG_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class ScaledDraw:
    """
    This class implements "Tube Drawing".
    It does not rely on Pillow's joint parameter; instead, it manually draws circular nodes at each coordinate point,
    ensuring all joints (including the start and end of closed shapes) are perfectly blended.
    """
    def __init__(self, target_img, scale):
        self.draw = ImageDraw.Draw(target_img)
        self.scale = scale
    
    def _scale_point(self, p):
        return (p[0] * self.scale, p[1] * self.scale)
    
    def _scale_bbox(self, bbox):
        return [x * self.scale for x in bbox]

    def line(self, points, fill, width, joint=None):
        # Ignore joint parameter; we handle joins manually within this function
        if not points:
            return

        scaled_points = [self._scale_point(p) for p in points]
        scaled_width = width * self.scale
        radius = scaled_width / 2

        # 1. Draw line segments (skeleton)
        # Slightly increase width (0.5px) to prevent micro gaps between line segments and circles
        self.draw.line(scaled_points, fill=fill, width=int(scaled_width))

        # 2. Key step: manually draw filled circles at each vertex (joints)
        # This solves all issues of "gap at closure" and "jagged corners at turns"
        for x, y in scaled_points:
            # bbox for the circle at vertex
            xy = [x - radius, y - radius, x + radius, y + radius]
            self.draw.ellipse(xy, fill=fill)

    def ellipse(self, bbox, outline, width):
        # For circles and ellipses, Pillow's anti-aliasing tends to have issues with larger widths.
        # For consistency, we still use draw.ellipse, but under supersampling the result is usually acceptable.
        # For perfect results, consider using arc simulation, but ellipse has better closure.
        scaled_bbox = self._scale_bbox(bbox)
        self.draw.ellipse(scaled_bbox, outline=outline, width=int(width * self.scale))
        
    def rectangle(self, bbox, outline, width):
        scaled_bbox = self._scale_bbox(bbox)
        # A rectangle is essentially a polygon; to ensure rounded corners, we use our own line method
        x0, y0, x1, y1 = scaled_bbox
        # Four points clockwise
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        # To reuse logic, implement directly here
        # Here we directly call the internal draw, simulating line logic
        radius = (width * self.scale) / 2

        # Draw four edges
        self.draw.line(pts, fill=outline, width=int(width * self.scale))
        # Draw circles at four corners
        for x, y in pts[:-1]: # Last point duplicates first, no need to draw again
             self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=outline)

    def arc(self, bbox, start, end, fill, width):
        scaled_bbox = self._scale_bbox(bbox)
        scaled_width = int(width * self.scale)
        self.draw.arc(scaled_bbox, start=start, end=end, fill=fill, width=scaled_width)

        # Optional: add round caps to both ends of the arc
        # Calculating arc start and end coordinates is somewhat complex; keeping default behavior for now
        # If round caps on both ends of the arc are needed, manually calculate coordinates to draw circles

def create_high_res_canvas():
    return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=BG_COLOR)

# === Drawing Logic ===

def draw_polygon(draw, sides, rotation=0):
    points = []
    for i in range(sides):
        angle = math.radians(rotation + i * (360 / sides))
        x = CENTER[0] + RADIUS * math.cos(angle)
        y = CENTER[1] + RADIUS * math.sin(angle)
        points.append((x, y))
    points.append(points[0]) # Close shape
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_circle(draw):
    bbox = [CENTER[0]-RADIUS, CENTER[1]-RADIUS, CENTER[0]+RADIUS, CENTER[1]+RADIUS]
    draw.ellipse(bbox, outline=LINE_COLOR, width=STROKE_WIDTH)

def draw_oval(draw):
    w_r, h_r = RADIUS, RADIUS * 0.6
    bbox = [CENTER[0]-w_r, CENTER[1]-h_r, CENTER[0]+w_r, CENTER[1]+h_r]
    draw.ellipse(bbox, outline=LINE_COLOR, width=STROKE_WIDTH)

def draw_rectangle_shape(draw):
    r = RADIUS * 0.8
    bbox = [CENTER[0]-r, CENTER[1]-r*0.7, CENTER[0]+r, CENTER[1]+r*0.7]
    draw.rectangle(bbox, outline=LINE_COLOR, width=STROKE_WIDTH)

def draw_rhombus(draw):
    w, h = RADIUS * 0.6, RADIUS
    points = [
        (CENTER[0], CENTER[1]-h),
        (CENTER[0]+w, CENTER[1]),
        (CENTER[0], CENTER[1]+h),
        (CENTER[0]-w, CENTER[1]),
        (CENTER[0], CENTER[1]-h)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_parallelogram(draw):
    w = RADIUS * 0.8
    h = RADIUS * 0.5
    skew = 200
    cx, cy = CENTER
    points = [
        (cx - w + skew/2, cy - h),
        (cx + w + skew/2, cy - h),
        (cx + w - skew/2, cy + h),
        (cx - w - skew/2, cy + h),
        (cx - w + skew/2, cy - h)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_trapezoid(draw):
    top_w = RADIUS * 0.5
    bot_w = RADIUS
    h = RADIUS * 0.7
    points = [
        (CENTER[0]-top_w, CENTER[1]-h),
        (CENTER[0]+top_w, CENTER[1]-h),
        (CENTER[0]+bot_w, CENTER[1]+h),
        (CENTER[0]-bot_w, CENTER[1]+h),
        (CENTER[0]-top_w, CENTER[1]-h)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_semicircle(draw):
    bbox = [CENTER[0]-RADIUS, CENTER[1]-RADIUS, CENTER[0]+RADIUS, CENTER[1]+RADIUS]
    draw.arc(bbox, start=180, end=0, fill=LINE_COLOR, width=STROKE_WIDTH)
    offset = STROKE_WIDTH / 2
    draw.line([(CENTER[0]-RADIUS+offset, CENTER[1]), (CENTER[0]+RADIUS-offset, CENTER[1])], fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_sector(draw):
    bbox = [CENTER[0]-RADIUS, CENTER[1]-RADIUS, CENTER[0]+RADIUS, CENTER[1]+RADIUS]
    start_angle = 210
    end_angle = 330
    draw.arc(bbox, start=start_angle, end=end_angle, fill=LINE_COLOR, width=STROKE_WIDTH)
    offset = STROKE_WIDTH / 2
    inner_R = RADIUS - offset
    
    p1 = (CENTER[0] + inner_R * math.cos(math.radians(start_angle)), CENTER[1] + inner_R * math.sin(math.radians(start_angle)))
    p2 = (CENTER[0] + inner_R * math.cos(math.radians(end_angle)), CENTER[1] + inner_R * math.sin(math.radians(end_angle)))
    
    draw.line([p1, CENTER, p2], fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_annulus(draw):
    r1 = RADIUS
    r2 = RADIUS * 0.5
    bbox1 = [CENTER[0]-r1, CENTER[1]-r1, CENTER[0]+r1, CENTER[1]+r1]
    bbox2 = [CENTER[0]-r2, CENTER[1]-r2, CENTER[0]+r2, CENTER[1]+r2]
    draw.ellipse(bbox1, outline=LINE_COLOR, width=STROKE_WIDTH)
    draw.ellipse(bbox2, outline=LINE_COLOR, width=STROKE_WIDTH)

def draw_capsule(draw):
    r = RADIUS * 0.5
    h_len = RADIUS * 0.5
    points = []
    
    # Upper semicircle
    for i in range(180, 361, 5): # Step size doesn't need to be too small because scale is large
        rad = math.radians(i)
        x = CENTER[0] + r * math.cos(rad)
        y = (CENTER[1] - h_len) + r * math.sin(rad)
        points.append((x, y))

    # Lower semicircle
    for i in range(0, 181, 5):
        rad = math.radians(i)
        x = CENTER[0] + r * math.cos(rad)
        y = (CENTER[1] + h_len) + r * math.sin(rad)
        points.append((x, y))
    
    points.append(points[0])
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_chevron(draw):
    w, h = RADIUS * 0.8, RADIUS * 0.5
    points = [
        (CENTER[0]-w, CENTER[1]-h),
        (CENTER[0], CENTER[1]+h),
        (CENTER[0]+w, CENTER[1]-h)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_star(draw):
    points = []
    outer_r = RADIUS
    inner_r = RADIUS * 0.382
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = outer_r if i % 2 == 0 else inner_r
        x = CENTER[0] + r * math.cos(angle)
        y = CENTER[1] + r * math.sin(angle)
        points.append((x, y))
    points.append(points[0])
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_arrow(draw):
    w_stem = RADIUS * 0.3
    h_stem = RADIUS * 0.5
    w_head = RADIUS * 0.7
    points = [
        (CENTER[0], CENTER[1]-RADIUS),
        (CENTER[0]+w_head, CENTER[1]),
        (CENTER[0]+w_stem, CENTER[1]),
        (CENTER[0]+w_stem, CENTER[1]+RADIUS),
        (CENTER[0]-w_stem, CENTER[1]+RADIUS),
        (CENTER[0]-w_stem, CENTER[1]),
        (CENTER[0]-w_head, CENTER[1]),
        (CENTER[0], CENTER[1]-RADIUS)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_lightning(draw):
    points = [
        (CENTER[0]+50, CENTER[1]-RADIUS),
        (CENTER[0]-150, CENTER[1]+50),
        (CENTER[0]-20, CENTER[1]+50),
        (CENTER[0]-80, CENTER[1]+RADIUS),
        (CENTER[0]+150, CENTER[1]-50),
        (CENTER[0]+20, CENTER[1]-50),
        (CENTER[0]+50, CENTER[1]-RADIUS)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_check(draw):
    points = [
        (CENTER[0]-300, CENTER[1]),
        (CENTER[0]-100, CENTER[1]+200),
        (CENTER[0]+300, CENTER[1]-300)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_cross(draw):
    w = 100
    h_top = 250
    h_bot = 400
    arm = 250
    cx, cy = CENTER
    points = [
        (cx-w, cy-h_top), (cx+w, cy-h_top),
        (cx+w, cy-w), (cx+arm, cy-w),
        (cx+arm, cy+w), (cx+w, cy+w),
        (cx+w, cy+h_bot), (cx-w, cy+h_bot),
        (cx-w, cy+w), (cx-arm, cy+w),
        (cx-arm, cy-w), (cx-w, cy-w),
        (cx-w, cy-h_top)
    ]
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_moon(draw):
    R = RADIUS
    r = RADIUS * 0.9 
    shift = 120
    
    x_int = (R**2 - r**2 + shift**2) / (2 * shift)
    y_int = math.sqrt(R**2 - x_int**2)
    
    angle_outer_top = math.atan2(-y_int, x_int)
    angle_outer_bot = math.atan2(y_int, x_int)
    angle_inner_top = math.atan2(-y_int, x_int - shift)
    angle_inner_bot = math.atan2(y_int, x_int - shift)
    
    points = []
    
    start_deg = math.degrees(angle_outer_bot)
    end_deg = math.degrees(angle_outer_top)
    if end_deg < start_deg: end_deg += 360
    
    # Appropriate density: one point every 1-2 degrees is sufficient, because at 4x scale the circle radius is large (20*4=80px), resulting in very strong coverage
    for i in range(int(start_deg * 2), int(end_deg * 2) + 1):
        rad = math.radians(i / 2)
        points.append((CENTER[0] + R * math.cos(rad), CENTER[1] + R * math.sin(rad)))
        
    start_deg_in = math.degrees(angle_inner_top)
    end_deg_in = math.degrees(angle_inner_bot)
    
    steps = 200
    diff = end_deg_in - start_deg_in
    if diff > 0: diff -= 360
    
    for i in range(steps + 1):
        current_deg = start_deg_in + (diff * i / steps)
        rad = math.radians(current_deg)
        points.append((CENTER[0] + shift + r * math.cos(rad), CENTER[1] + r * math.sin(rad)))
    
    points.append(points[0])
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_heart(draw):
    points = []
    # Appropriately reduce density from 628*2 to 628, because the circle radius is large enough for connection
    for t in range(0, 628):
        t = t / 100.0
        x = 16 * math.sin(t)**3
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        scale = 25
        points.append((CENTER[0] + x * scale, CENTER[1] + y * scale))
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_spiral(draw):
    points = []
    turns = 3
    for i in range(0, 360 * turns, 5):
        angle = math.radians(i)
        r = 5 + 0.5 * i
        x = CENTER[0] + r * math.cos(angle)
        y = CENTER[1] + r * math.sin(angle)
        points.append((x, y))
    draw.line(points, fill=LINE_COLOR, width=STROKE_WIDTH)

def draw_wifi(draw):
    r_dot = 40
    # Filled circle
    r_scaled = r_dot * SAMPLING_FACTOR
    # Manually draw a filled circle
    draw.ellipse([CENTER[0]-r_dot, CENTER[1]+200-r_dot, CENTER[0]+r_dot, CENTER[1]+200+r_dot], outline=LINE_COLOR, width=r_dot) # Hack to fill

    for r in [150, 280, 410]:
        bbox = [CENTER[0]-r, CENTER[1]+200-r, CENTER[0]+r, CENTER[1]+200+r]
        draw.arc(bbox, 225, 315, fill=LINE_COLOR, width=STROKE_WIDTH)

# Mapping table
# 55, a cloud icon shape,055_geometry_and_shapes_complex_concave_shape_and_icon_cloud_shape.png
# 57, a question mark symbol,057_geometry_and_shapes_complex_concave_shape_and_icon_question_mark.png
# 58, a musical note symbol,058_geometry_and_shapes_complex_concave_shape_and_icon_musical_note.png
# 59, a Yin-Yang symbol,059_geometry_and_shapes_complex_concave_shape_and_icon_taiji.png
# 61, a spade symbol (playing card suit),061_geometry_and_shapes_complex_concave_shape_and_icon_spade.png
# 62, a club symbol (playing card suit),062_geometry_and_shapes_complex_concave_shape_and_icon_club.png
# 63, the shape of a cogwheel,063_geometry_and_shapes_complex_concave_shape_and_icon_cogwheel.png
# The above images are not generated by code
draw_map = {
    33: (draw_circle, "033_geometry_and_shapes_basic_convex_shape_circle.png"),
    34: (lambda d: draw_polygon(d, 4, 45), "034_geometry_and_shapes_basic_convex_shape_square.png"),
    35: (lambda d: draw_polygon(d, 3, -90), "035_geometry_and_shapes_basic_convex_shape_triangle.png"),
    36: (draw_rectangle_shape, "036_geometry_and_shapes_basic_convex_shape_rectangle.png"),
    37: (draw_oval, "037_geometry_and_shapes_basic_convex_shape_oval.png"),
    38: (lambda d: draw_polygon(d, 5, -90), "038_geometry_and_shapes_basic_convex_shape_pentagon.png"),
    39: (lambda d: draw_polygon(d, 6), "039_geometry_and_shapes_basic_convex_shape_hexagon.png"),
    40: (lambda d: draw_polygon(d, 8, 22.5), "040_geometry_and_shapes_basic_convex_shape_octagon.png"),
    41: (draw_rhombus, "041_geometry_and_shapes_basic_convex_shape_rhombus.png"),
    42: (draw_parallelogram, "042_geometry_and_shapes_basic_convex_shape_parallelogram.png"),
    43: (draw_trapezoid, "043_geometry_and_shapes_basic_convex_shape_trapezoid.png"),
    44: (draw_semicircle, "044_geometry_and_shapes_basic_convex_shape_semicircle.png"),
    45: (draw_annulus, "045_geometry_and_shapes_basic_convex_shape_annulus.png"),
    46: (draw_capsule, "046_geometry_and_shapes_basic_convex_shape_capsule_shape.png"),
    47: (draw_sector, "047_geometry_and_shapes_basic_convex_shape_sector.png"),
    48: (draw_chevron, "048_geometry_and_shapes_basic_convex_shape_chevron.png"),
    49: (draw_star, "049_geometry_and_shapes_complex_concave_shape_and_icon_star.png"),
    50: (draw_heart, "050_geometry_and_shapes_complex_concave_shape_and_icon_heart.png"),
    51: (draw_moon, "051_geometry_and_shapes_complex_concave_shape_and_icon_crescent_moon.png"),
    52: (draw_cross, "052_geometry_and_shapes_complex_concave_shape_and_icon_cross.png"),
    53: (draw_arrow, "053_geometry_and_shapes_complex_concave_shape_and_icon_arrow.png"),
    54: (draw_lightning, "054_geometry_and_shapes_complex_concave_shape_and_icon_lightning_bolt.png"),
    56: (draw_check, "056_geometry_and_shapes_complex_concave_shape_and_icon_checkmark.png"),
    60: (draw_wifi, "060_geometry_and_shapes_complex_concave_shape_and_icon_wifi_symbol.png"),
    64: (draw_spiral, "064_geometry_and_shapes_complex_concave_shape_and_icon_spiral.png"),
}

def generate_all():
    print("Starting geometry shape generation (ultimate seamless version)...")
    count = 0
    for img_id, (draw_func, filename) in draw_map.items():
        # 1. Create high-resolution canvas
        img_high_res = create_high_res_canvas()

        # 2. Instantiate drawing proxy
        scaled_draw = ScaledDraw(img_high_res, SAMPLING_FACTOR)

        try:
            draw_func(scaled_draw)

            # 3. Downscale
            # Use BOX filter, which provides smoother anti-aliasing when scaling down significantly (4x -> 1x),
            # avoiding the ringing artifacts (white halos/jagged feel) produced by LANCZOS
            final_img = img_high_res.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.BOX)

            save_path = os.path.join(OUTPUT_DIR, filename)
            final_img.save(save_path)
            print(f"[OK] Generated: {filename}")
            count += 1
        except Exception as e:
            print(f"[Fail] ID {img_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Generation complete. Total: {count} images.")

if __name__ == "__main__":
    generate_all()