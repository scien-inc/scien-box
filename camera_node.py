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
    retry_count = 0
    retry_delay = 5  # 開始リトライ間隔（秒）
    max_retry_delay = 60  # 最大リトライ間隔（秒）
    heartbeat_interval = 30  # 通常のハートビート間隔（秒）
    
    while True:
        try:
            # ノード情報を更新
            node_info['ip'] = get_local_ip()
            node_info['last_heartbeat'] = time.time()
            
            # 中央サーバーに登録
            logger.info(f"中央サーバーに登録を試みます: {CENTRAL_SERVER}/api/register")
            try:
                response = requests.post(f"{CENTRAL_SERVER}/api/register", json=node_info, timeout=10)
                logger.info(f"登録リクエスト送信完了。ステータスコード: {response.status_code}")
                if response.status_code == 200:
                    logger.info(f"中央サーバーへの登録に成功しました: {response.json()}")
                    # 成功したらリトライカウントとディレイをリセット
                    retry_count = 0
                    retry_delay = 5
                    # 次のハートビートまで通常間隔で待機
                    time.sleep(heartbeat_interval)
                    continue
                else:
                    logger.warning(f"中央サーバーへの登録に失敗しました: {response.status_code}, レスポンス: {response.text}")
                    retry_count += 1
            except requests.exceptions.RequestException as req_err:
                logger.error(f"中央サーバーへのリクエスト中にエラーが発生: {req_err}")
                retry_count += 1
        
        except Exception as e:
            logger.error(f"中央サーバーへの登録エラー: {e}")
            retry_count += 1
        
        # エラー発生時はバックオフ戦略でリトライ
        if retry_count > 0:
            # 指数バックオフ（最大まで）
            current_delay = min(retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.info(f"サーバーへの接続リトライを {current_delay}秒後に行います。(リトライ回数: {retry_count})")
            time.sleep(current_delay)
        else:
            # 通常のハートビート間隔
            time.sleep(heartbeat_interval)

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
    global frame, camera_running
    if frame is None:
        return jsonify({'error': 'No frame available'}), 400
    
    try:
        # 一時的に高解像度で撮影
        if camera_running:
            try:
                # 現在のカメラを使用して高解像度で撮影
                camera = Picamera2()
                high_res_config = camera.create_still_configuration(main={"size": (2592, 1944)})
                camera.configure(high_res_config)
                camera.start()
                time.sleep(0.5)  # カメラの安定化を待つ
                high_res_img = camera.capture_array()
                camera.stop()
                
                # 必要に応じてBGRに変換
                channels = 1 if len(high_res_img.shape) == 2 else high_res_img.shape[2]
                if channels == 1:
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_GRAY2BGR)
                elif channels == 4:
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGRA2BGR)
                
                # 高解像度画像をJPEGとしてエンコード
                ret, buffer = cv2.imencode('.jpg', high_res_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if not ret:
                    # 高解像度撮影に失敗した場合、通常のフレームを使用
                    logger.warning("高解像度撮影に失敗しました。通常解像度で対応します。")
                    with lock:
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
            except Exception as e:
                logger.error(f"高解像度撮影エラー: {e}")
                # エラーが発生した場合、通常のフレームを使用
                with lock:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            # カメラが実行中でない場合、通常のフレームを使用
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