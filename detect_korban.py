from maix import camera, display, image, nn, app, uart

detector = nn.YOLOv5(model="/root/models/yolov5_korban.mud", dual_buff = True)

cam = camera.Camera(500, 250, detector.input_format())
dis = display.Display()
device = "/dev/ttyS0"

serial0 = uart.UART(device, 115200)
while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    for obj in objs:
        # Calculate the center point
        center_x = obj.x + obj.w // 2
        center_y = obj.y + obj.h // 2
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        print(f'Center: (X:{center_x}, Y:{center_y})\r\n')
        data = f"{center_x},{center_y}\r\n".encode()
        serial0.write(data)
    dis.show(img)
