from maix import camera, display, image, nn, app
import time

# Optimize detector initialization
detector = nn.YOLOv8(
    model="/root/models/yolov8_korban.mud", 
    dual_buff=True,
    # Consider adding these optimization parameters if available
    # quantized=True,  # If model supports quantization
    # core=1,          # Specify specific AI core
)

# Reduce image processing overhead
cam = camera.Camera(
    detector.input_width(), 
    detector.input_height(), 
    detector.input_format(),
    # Consider adding buffer options
    # buffer_count=2
)
dis = display.Display()

# Performance tracking
frame_count = 0
start_time = time.time()

while not app.need_exit():
    # Read image with potential timeout
    img = cam.read()
    if img is None:
        continue

    # Reduce detection frequency or complexity
    objs = detector.detect(
        img, 
        conf_th=0.3,    # Lower confidence threshold
        iou_th=0.4,     # Slightly lower IoU threshold
        max_target=5    # Limit maximum number of detected objects
    )

    # Efficient drawing
    for obj in objs:
        # Combine drawing operations
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
        
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y - 10, msg, color=image.COLOR_RED)

    dis.show(img)

    # FPS calculation
    frame_count += 1
    if frame_count % 30 == 0:
        fps = frame_count / (time.time() - start_time)
        print(f'Current FPS: {fps:.2f}')
        frame_count = 0
        start_time = time.time()