from flask import Flask, render_template, request, jsonify, Response
import json
import threading
import time
import logging
import os
import requests
from datetime import datetime
import cv2
import uuid
from picamera2 import Picamera2
from libcamera import controls
import socket
import base64
import numpy as np

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flaskアプリの初期化
app = Flask(__name__)

# グローバル変数
cameras = {}  # カメラノード情報を格納する辞書
camera_lock = threading.Lock()  # スレッドセーフな操作のためのロック
HEARTBEAT_TIMEOUT = 300  # ハートビートタイムアウト（秒）- 5分に延長

# サーバー設定
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5001))
SERVER_IP = os.environ.get('SERVER_IP', '192.168.179.200')
NODE_ID = str(uuid.uuid4())[:8]  # サーバー自身のユニークID
NODE_NAME = os.environ.get('SERVER_NODE_NAME', 'server-camera')
RESOLUTION = (1280, 720)  # カメラ解像度

# ローカルカメラ変数
frame = None
frame_lock = threading.Lock()
camera_running = False

# --- 内部ヘルパー関数 ---

# ノードの活性チェック
def is_node_alive(node_info):
    last_heartbeat = node_info.get('last_heartbeat', 0)
    return (time.time() - last_heartbeat) < HEARTBEAT_TIMEOUT

# ノードにリクエストを送信する関数
def request_node(node_id, endpoint, method='GET', data=None, timeout=3):
    if node_id not in cameras:
        return None, 'Node not found'
    
    node = cameras[node_id]
    url = f"http://{node['ip']}:{node['port']}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None, f'Unsupported method: {method}'
        
        return response.json() if response.status_code == 200 else None, response.status_code
    
    except requests.exceptions.RequestException as e:
        logger.error(f"ノード {node_id} へのリクエストエラー: {e}")
        return None, str(e)

# ノードのクリーンアップを行うスレッド
def cleanup_thread():
    while True:
        try:
            current_time = time.time()
            with camera_lock:
                # タイムアウトしたノードを特定（サーバー自身は除外）
                timed_out_nodes = [
                    node_id for node_id, info in cameras.items()
                    if (current_time - info.get('last_heartbeat', 0) > HEARTBEAT_TIMEOUT) and (node_id != NODE_ID)
                ]
                
                # タイムアウトしたノードを削除
                for node_id in timed_out_nodes:
                    logger.info(f"ノード {node_id} ({cameras[node_id].get('name', 'unknown')}) がタイムアウトしました")
                    cameras.pop(node_id, None)
                
                # サーバー自身のハートビートを更新
                if NODE_ID in cameras:
                    cameras[NODE_ID]['last_heartbeat'] = current_time
                    cameras[NODE_ID]['status'] = 'running' if camera_running else 'error'
                
                # 各ノードのステータスを更新（サーバー自身は除外）
                for node_id, info in list(cameras.items()):
                    if node_id == NODE_ID:
                        continue  # サーバー自身はスキップ
                        
                    # 30秒ごとにヘルスチェック
                    if current_time - info.get('last_checked', 0) > 30:
                        try:
                            response = requests.get(
                                f"http://{info['ip']}:{info['port']}/api/health", 
                                timeout=2
                            )
                            info['status'] = 'running' if response.status_code == 200 else 'error'
                        except:
                            info['status'] = 'unreachable'
                        
                        info['last_checked'] = current_time
                        cameras[node_id] = info  # 更新
        
        except Exception as e:
            logger.error(f"クリーンアップスレッドエラー: {e}")
        
        # 10秒待機
        time.sleep(10)

# カメラ初期化関数（サーバー自身のカメラ）
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
        logger.info("サーバーカメラを初期化しました")
        return camera
    except Exception as e:
        logger.error(f"サーバーカメラの初期化に失敗しました: {e}")
        camera_running = False
        return None

# フレームをキャプチャするスレッド関数
def capture_frames(camera):
    global frame, camera_running
    
    logger.info("サーバーカメラのフレームキャプチャスレッドを開始しました")
    
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
            with frame_lock:
                frame = img
            
            # フレームレートの制御
            time.sleep(0.03)  # 約30FPS
        
        except Exception as e:
            logger.error(f"サーバーカメラのフレームキャプチャエラー: {e}")
            time.sleep(1)
    
    logger.info("サーバーカメラのフレームキャプチャスレッドを停止しました")

# ストリーミング用のフレーム生成（サーバーカメラ用）
def generate_frames():
    global frame
    
    while True:
        # フレームが利用可能になるまで待機
        if frame is None:
            time.sleep(0.1)
            continue
        
        try:
            # 最新のフレームを取得
            with frame_lock:
                img = frame.copy()
            
            # フレームをJPEGとしてエンコード
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            
            # MJPEGフォーマットでフレームを返す
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            logger.error(f"サーバーカメラのフレーム生成エラー: {e}")
            time.sleep(0.5)

# サーバー自身をカメラノードとして登録
def register_server_camera():
    global cameras
    
    server_info = {
        'id': NODE_ID,
        'name': NODE_NAME,
        'ip': SERVER_IP,
        'port': SERVER_PORT,
        'status': 'running' if camera_running else 'error',
        'resolution': RESOLUTION,
        'last_heartbeat': time.time(),
        'last_checked': time.time()
    }
    
    with camera_lock:
        cameras[NODE_ID] = server_info
        logger.info(f"サーバー自身をカメラノードとして登録しました: {NODE_ID}")

# サーバー自身のカメラステータスを更新するスレッド
def server_camera_status_thread():
    global camera_running, frame
    
    logger.info("サーバーカメラステータス監視スレッドを開始しました")
    
    while True:
        try:
            with camera_lock:
                if NODE_ID in cameras:
                    # カメラが動作しているか確認
                    if frame is None:
                        camera_running = False
                        cameras[NODE_ID]['status'] = 'error'
                    else:
                        with frame_lock:
                            if frame is not None:
                                camera_running = True
                                cameras[NODE_ID]['status'] = 'running'
                    
                    # ハートビートを更新
                    cameras[NODE_ID]['last_heartbeat'] = time.time()
                    cameras[NODE_ID]['last_checked'] = time.time()
        except Exception as e:
            logger.error(f"サーバーカメラステータス更新エラー: {e}")
        
        time.sleep(10)

# --- APIエンドポイント ---

# カメラノードの登録/ハートビート
@app.route('/api/register', methods=['POST'])
def register_camera():
    logger.info(f"カメラ登録リクエストを受信しました: {request.remote_addr}")
    try:
        node_info = request.json
        logger.info(f"登録データ: {node_info}")
        node_id = node_info.get('id')
        
        if not node_id:
            logger.error("Node IDがリクエストに含まれていません")
            return jsonify({'error': 'Node ID is required'}), 400
        
        # タイムスタンプを更新
        node_info['last_heartbeat'] = time.time()
        
        # ノード情報を保存/更新
        with camera_lock:
            if node_id in cameras:
                # 既存のノードを更新
                cameras[node_id].update(node_info)
                logger.info(f"ノード {node_id} ({node_info.get('name')}) のハートビートを受信しました")
            else:
                # 新しいノードを登録
                cameras[node_id] = node_info
                logger.info(f"新しいノード {node_id} ({node_info.get('name')}) を登録しました")
            
            # デバッグ用：現在登録されているすべてのカメラを表示
            logger.info(f"現在登録されているカメラ: {list(cameras.keys())}")
        
        return jsonify({'status': 'registered', 'id': node_id})
    
    except Exception as e:
        logger.error(f"カメラ登録処理中にエラーが発生しました: {str(e)}")
        return jsonify({'error': str(e)}), 500

# すべてのカメラノード情報を取得
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    active_cameras = {}
    
    with camera_lock:
        for node_id, info in cameras.items():
            # 不要なデータをフィルタリング
            filtered_info = {
                'id': info.get('id'),
                'name': info.get('name'),
                'ip': info.get('ip'),
                'port': info.get('port'),
                'status': info.get('status'),
                'resolution': info.get('resolution'),
                'url': f"http://{info.get('ip')}:{info.get('port')}/stream",
                'last_seen': datetime.fromtimestamp(info.get('last_heartbeat', 0)).strftime('%Y-%m-%d %H:%M:%S')
            }
            active_cameras[node_id] = filtered_info
    
    return jsonify(active_cameras)

# 特定のカメラノードからスナップショットを取得
@app.route('/api/snapshot/<node_id>', methods=['GET'])
def get_snapshot(node_id):
    # サーバー自身のカメラの場合
    if node_id == NODE_ID:
        global frame, camera_running
        if frame is None:
            return jsonify({'error': 'No frame available'}), 404
        
        try:
            # サーバーカメラでも高解像度撮影を試みる
            if camera_running:
                try:
                    # 一時的に高解像度で撮影
                    camera = Picamera2()
                    high_res_config = camera.create_still_configuration(main={"size": (2592, 1944)})
                    camera.configure(high_res_config)
                    camera.start()
                    time.sleep(0.5)
                    high_res_img = camera.capture_array()
                    camera.stop()
                    
                    # 必要に応じてBGRに変換
                    channels = 1 if len(high_res_img.shape) == 2 else high_res_img.shape[2]
                    if channels == 1:
                        high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_GRAY2BGR)
                    elif channels == 4:
                        high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGRA2BGR)
                    
                    # 高解像度画像をエンコード
                    ret, buffer = cv2.imencode('.jpg', high_res_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if not ret:
                        with frame_lock:
                            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception as e:
                    logger.error(f"サーバー高解像度撮影エラー: {e}")
                    with frame_lock:
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                with frame_lock:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if not ret:
                return jsonify({'error': 'Failed to encode image'}), 500
            
            # Base64でエンコード
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'timestamp': time.time(),
                'image': img_str
            })
        
        except Exception as e:
            logger.error(f"サーバーカメラのスナップショットエラー: {e}")
            return jsonify({'error': str(e)}), 500
    
    # 他のカメラノードの場合
    if node_id not in cameras:
        return jsonify({'error': 'Camera not found'}), 404
    
    data, status = request_node(node_id, '/api/snapshot')
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': f'Failed to get snapshot: {status}'}), 500

# サーバーカメラのストリーム
@app.route('/stream')
def video_stream():
    global camera_running, frame
    
    # カメラが動作していることを確認
    if not camera_running or frame is None:
        # カメラが動作していない場合、オフライン画像を返す
        try:
            with open('static/offline.jpg', 'rb') as f:
                offline_image = f.read()
                
            # 単一フレームのMJPEGとして返す
            def generate_offline():
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + offline_image + b'\r\n')
                
            return Response(generate_offline(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        except:
            pass
    
    # 通常のストリームを返す
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# サーバーカメラのヘルスチェック
@app.route('/api/health', methods=['GET'])
def health_check():
    if camera_running:
        return jsonify({'status': 'ok', 'camera': 'running'})
    else:
        return jsonify({'status': 'error', 'camera': 'not running'}), 500

# メインページ
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':

    
    # static ディレクトリの作成とオフライン画像の作成
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # オフライン画像の作成（カメラが利用できない場合の表示用）
    offline_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(offline_img, "Camera Offline", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('static/offline.jpg', offline_img)
    
    # サーバーカメラの初期化
    try:
        camera = initialize_camera()
        
        if camera is not None:
            # フレームキャプチャスレッドの開始
            capture_thread = threading.Thread(target=capture_frames, args=(camera,))
            capture_thread.daemon = True
            capture_thread.start()
            
            # サーバー自身をカメラノードとして登録
            register_server_camera()
    except Exception as e:
        logger.error(f"サーバーカメラの初期化エラー: {e}")
    
    # クリーンアップスレッドの開始
    cleanup_thread = threading.Thread(target=cleanup_thread)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # サーバーの開始
    logger.info(f"中央サーバーを開始します: http://{SERVER_IP}:{SERVER_PORT}")
    app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True)