import cv2
from picamera2 import Picamera2
from libcamera import controls

# Picameraを起動
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={
	"format": 'XRGB8888',
	"size": (3000, 2000)
}))
camera.start()
camera.set_controls({'AfMode': controls.AfModeEnum.Continuous})

# カメラから画像を取得
image = camera.capture_array()

# 画像が3チャンネル以外の場合は3チャンネルに変換する
channels = 1 if len(image.shape) == 2 else image.shape[2]
if channels == 1:
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if channels == 4:
	image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# jpgに保存
cv2.imwrite('test.jpg', image)
