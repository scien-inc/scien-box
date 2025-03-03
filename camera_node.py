import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
from libcamera import controls
import threading
import time
import socket
import json
import logging
import uuid
import os
import requests

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定
NODE_NAME = os.environ.get('CAMERA_NODE_NAME', f'camera-{socket.gethostname()}')
NODE_ID = str(uuid.uuid4())[:8]  # ユニークID
CENTRAL_SERVER = os.environ.get('CENTRAL_SERVER', 'http://192.168.179.200:5001')  # 中央サーバーのアドレス
API_PORT = int(os.environ.get('API_PORT', 8000))
STREAM_QUALITY = int(os.environ.get('STREAM_QUALITY', 70))  # JPEG品質
RESOLUTION = (1280, 720)  # カメラ解像度
NODE_IP = os.environ.get('NODE_IP', None)  # 環境変数からノードのIPを取得

# Flaskアプリの初期化
app = Flask(__name__)

# グローバル変数
frame = None
lock = threading.Lock()
camera_running = False
node_info = {
    'id': NODE_ID,
    'name': NODE_NAME,
    'ip': None,
    'port': API_PORT,
    'status': 'initializing',
    'resolution': RESOLUTION,
    'last_heartbeat': None
}

# ローカルIPアドレスを取得する関数
def get_local_ip():
    # 環境変数でIPが指定されている場合はそれを使用
    if NODE_IP:
        logger.info(f"環境変数から指定されたIPアドレスを使用します: {NODE_IP}")
        return NODE_IP
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"IPアドレス取得エラー: {e}")
        return '127.0.0.1'

# カメラの初期化
def initialize_camera():
    global camera_running
    try:
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(main={
            "format": 'XRGB8888',
            "size": RESOLUTION
        }))
        camera.start()
        camera.set_controls({'AfMode': controls.AfModeEnum.Continuous})
        camera_running = True
        node_info['status'] = 'running'
        logger.info("カメラを初期化しました")
        return camera
    except Exception as e:
        logger.error(f"カメラの初期化に失敗しました: {e}")
        camera_running = False
        node_info['status'] = 'error'
        return None

# フレームをキャプチャするスレッド関数
def capture_frames(camera):
    global frame, camera_running
    
    logger.info("フレームキャプチャスレッドを開始しました")
    
    while camera_running:
        try:
            # フレームのキャプチャ
            img = camera.capture_array()
            
            # 必要に応じてBGRに変換
            channels = 1 if len(img.shape) == 2 else img.shape[2]
            if channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # グローバルフレームの更新
            with lock:
                frame = img
            
            # フレームレートの制御
            time.sleep(0.03)  # 約30FPS
        
        except Exception as e:
            logger.error(f"フレームキャプチャエラー: {e}")
            time.sleep(1)
    
    logger.info("フレームキャプチャスレッドを停止しました")

# ストリーミング用のフレーム生成
def generate_frames():
    global frame
    
    while True:
        # フレームが利用可能になるまで待機
        if frame is None:
            time.sleep(0.1)
            continue
        
        try:
            # 最新のフレームを取得
            with lock:
                img = frame.copy()
            
            # フレームをJPEGとしてエンコード
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
            if not ret:
                continue
            
            # MJPEGフォーマットでフレームを返す
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            logger.error(f"フレーム生成エラー: {e}")
            time.sleep(0.5)

# 中央サーバーへの登録スレッド
def registration_thread():
    while True:
        try:
            # ノード情報を更新
            node_info['ip'] = get_local_ip()
            node_info['last_heartbeat'] = time.time()
            
            # 中央サーバーに登録
            logger.info(f"中央サーバーに登録を試みます: {CENTRAL_SERVER}/api/register")
            try:
                response = requests.post(f"{CENTRAL_SERVER}/api/register", json=node_info, timeout=5)
                logger.info(f"登録リクエスト送信完了。ステータスコード: {response.status_code}")
                if response.status_code == 200:
                    logger.info(f"中央サーバーへの登録に成功しました: {response.json()}")
                else:
                    logger.warning(f"中央サーバーへの登録に失敗しました: {response.status_code}, レスポンス: {response.text}")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"中央サーバーへのリクエスト中にエラーが発生: {req_err}")
        
        except Exception as e:
            logger.error(f"中央サーバーへの登録エラー: {e}")
        
        # 次のハートビートまで待機
        time.sleep(30)

# --- API エンドポイント ---

# ノード情報
@app.route('/api/info', methods=['GET'])
def get_node_info():
    return jsonify(node_info)

# ヘルスチェック
@app.route('/api/health', methods=['GET'])
def health_check():
    if camera_running:
        return jsonify({'status': 'ok', 'camera': 'running'})
    else:
        return jsonify({'status': 'error', 'camera': 'not running'}), 500

# ビデオストリーム
@app.route('/stream')
def video_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# スナップショット取得
@app.route('/api/snapshot', methods=['GET'])
def snapshot():
    global frame
    if frame is None:
        return jsonify({'error': 'No frame available'}), 400
    
    try:
        # 現在のフレームをJPEGとしてエンコード
        with lock:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not ret:
            return jsonify({'error': 'Failed to encode image'}), 500
        
        # Base64でエンコード
        import base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'timestamp': time.time(),
            'image': img_str
        })
    
    except Exception as e:
        logger.error(f"スナップショットエラー: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # IPアドレスの取得と設定
    node_info['ip'] = get_local_ip()
    
    # カメラの初期化
    camera = initialize_camera()
    
    if camera is not None:
        # フレームキャプチャスレッドの開始
        capture_thread = threading.Thread(target=capture_frames, args=(camera,))
        capture_thread.daemon = True
        capture_thread.start()
        
        # 登録スレッドの開始
        reg_thread = threading.Thread(target=registration_thread)
        reg_thread.daemon = True
        reg_thread.start()
        
        # サーバーの開始
        logger.info(f"カメラノードサーバーを開始します: http://{node_info['ip']}:{API_PORT}")
        app.run(host='0.0.0.0', port=API_PORT, threaded=True)
    else:
        logger.error("カメラの初期化に失敗したため、アプリケーションを終了します")