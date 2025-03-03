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
    # templatesディレクトリが存在しない場合は作成
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # dashboardテンプレートの作成
    dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>カメラ監視システム | SCIEN, Inc</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="utf-8">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #0062cc;
            --primary-light: #e6f0ff;
            --primary-dark: #004c9e;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --border-radius: 8px;
            --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            --transition-speed: 0.3s;
        }
        
        body {
            font-family: 'Noto Sans JP', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
            display: flex;
            min-height: 100vh;
            line-height: 1.5;
        }
        
        /* サイドバーナビゲーション */
        .sidebar {
            width: 240px;
            background: linear-gradient(to bottom, #ffffff, #f8f9fa);
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
            z-index: 100;
            padding: 0;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            transition: transform var(--transition-speed) ease;
            overflow: hidden;
            border-right: 1px solid rgba(0,0,0,0.05);
        }
        
        .sidebar-logo {
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            background-color: white;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .logo-img {
            height: 36px;
            width: auto;
            margin-right: 10px;
        }
        
        .sidebar-logo h1 {
            font-size: 16px;
            margin: 0;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--dark-color);
        }
        
        .mobile-toggle {
            display: none;
            font-size: 24px;
            background: none;
            border: none;
            color: var(--dark-color);
            cursor: pointer;
            margin-left: auto;
        }
        
        .tabs {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            padding: 20px 0;
        }
        
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            font-weight: 500;
            color: var(--secondary-color);
            position: relative;
            display: flex;
            align-items: center;
            transition: all var(--transition-speed) ease;
            border-left: 3px solid transparent;
            margin: 4px 0;
        }
        
        .tab:hover {
            background-color: var(--primary-light);
            color: var(--primary-color);
        }
        
        .tab.active {
            color: var(--primary-color);
            background-color: var(--primary-light);
            border-left-color: var(--primary-color);
            font-weight: 600;
        }
        
        .tab .tab-icon {
            margin-right: 12px;
            width: 20px;
            text-align: center;
            font-size: 18px;
        }
        
        .sidebar-footer {
            padding: 15px 20px;
            border-top: 1px solid rgba(0,0,0,0.05);
            font-size: 12px;
            color: var(--secondary-color);
            text-align: center;
            background-color: white;
        }
        
        .sidebar-footer img {
            height: 24px;
            margin-bottom: 10px;
        }
        
        /* メインコンテンツエリア */
        .main-content {
            flex-grow: 1;
            padding: 24px;
            margin-left: 240px;
            transition: margin-left var(--transition-speed) ease;
            background-color: #f8f9fa;
        }
        
        /* ヘッダー部分 */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 12px rgba(0,0,0,0.03);
            border-radius: var(--border-radius);
            border: 1px solid rgba(0,0,0,0.03);
        }
        
        .header-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
            color: var(--dark-color);
            position: relative;
            padding-bottom: 5px;
        }
        
        .header-title:after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 40px;
            height: 3px;
            background-color: var(--primary-color);
            border-radius: 3px;
        }
        
        .controls {
            display: flex;
            gap: 12px;
        }
        
        .button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: var(--border-radius);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .button:hover {
            background-color: var(--primary-dark);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .button:active {
            transform: translateY(1px);
            box-shadow: 0 1px 2px rgba(0, 98, 204, 0.3);
        }
        
        .button-icon {
            margin-right: 8px;
            font-size: 16px;
        }
        
        /* タブコンテンツ */
        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease forwards;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* カメラグリッド */
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 24px;
            margin-bottom: 30px;
        }
        
        .camera-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
            border: 1px solid rgba(0,0,0,0.03);
        }
        
        .camera-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background-color: rgba(0,0,0,0.02);
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .camera-title {
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            font-size: 1rem;
            color: var(--dark-color);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            position: relative;
        }
        
        .status-indicator::after {
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            border-radius: 50%;
            background-color: transparent;
            z-index: 1;
        }
        
        .status-running {
            background-color: var(--success-color);
            box-shadow: 0 0 0 2px rgba(40, 167, 69, 0.2);
        }
        
        .status-running::after {
            animation: pulse 2s infinite;
            border: 2px solid rgba(40, 167, 69, 0.4);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            70% { transform: scale(1.5); opacity: 0; }
            100% { transform: scale(1.5); opacity: 0; }
        }
        
        .status-error, .status-unreachable {
            background-color: var(--danger-color);
        }
        
        .status-initializing {
            background-color: var(--warning-color);
        }
        
        .camera-actions {
            display: flex;
            gap: 8px;
        }
        
        .camera-actions button {
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: var(--border-radius);
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .camera-actions button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .camera-stream {
            width: 100%;
            height: 300px;
            background-color: #111;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        
        .camera-stream img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            transition: transform var(--transition-speed) ease;
        }
        
        .camera-stream.zoomed {
            cursor: move;
        }
        
        .zoom-controls {
            position: absolute;
            bottom: 15px;
            right: 15px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            backdrop-filter: blur(2px);
        }
        
        .zoom-controls button:hover {
            background-color: rgba(0, 98, 204, 0.8);
            transform: translateY(-2px);
        }
        
        .camera-info {
            padding: 15px 20px;
            font-size: 13px;
            color: var(--secondary-color);
            border-top: 1px solid rgba(0,0,0,0.05);
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            background: linear-gradient(to bottom, #ffffff, #fafafa);
        }
        
        .camera-info p {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        
        .camera-info p strong {
            color: var(--dark-color);
            margin-right: 6px;
            display: inline-block;
            width: 80px;
            position: relative;
        }
        
        .camera-info p strong::after {
            content: ':';
            position: absolute;
            right: 6px;
        }
        
        .loading {
            color: white;
            font-size: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        
        .loading::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-top: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .placeholder {
            text-align: center;
            padding: 40px;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .placeholder p {
            color: var(--secondary-color);
            margin: 10px 0;
        }
        
        .placeholder-icon {
            font-size: 48px;
            color: var(--primary-light);
            margin-bottom: 20px;
        }
        
        /* モーダル用のスタイル */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
            backdrop-filter: blur(3px);
        }
        
        .modal.visible {
            opacity: 1;
        }
        
        .modal-content {
            background-color: white;
            border-radius: var(--border-radius);
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transform: scale(0.9);
            opacity: 0;
            transition: all var(--transition-speed) ease;
        }
        
        .modal.visible .modal-content {
            transform: scale(1);
            opacity: 1;
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: var(--secondary-color);
            transition: color var(--transition-speed) ease;
            line-height: 1;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }
        
        .modal-close:hover {
            color: var(--danger-color);
            background-color: rgba(220, 53, 69, 0.1);
        }
        
        .snapshot-img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            transition: transform var(--transition-speed) ease;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .error-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(220, 53, 69, 0.8);
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
            text-align: center;
            padding: 20px;
            backdrop-filter: blur(2px);
        }
        
        .error-overlay button {
            margin-top: 15px;
            background-color: white;
            color: var(--danger-color);
            border: none;
            padding: 8px 16px;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        .error-overlay button:hover {
            background-color: #f8f9fa;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        /* 機能カード */
        .camera-grid-function {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 24px;
            margin-top: 20px;
        }
        
        .function-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            padding: 0;
            border: 1px solid rgba(0,0,0,0.03);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
        }
        
        .function-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .function-card h3 {
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 15px 20px;
            margin: 0;
            background-color: rgba(0,0,0,0.02);
            font-weight: 600;
            font-size: 1rem;
            color: var(--dark-color);
            position: relative;
        }
        
        .function-card h3:after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 30px;
            height: 3px;
            background-color: var(--primary-color);
        }
        
        .function-card-content {
            padding: 20px;
        }
        
        .capture-btn {
            display: flex;
            width: 100%;
            padding: 14px;
            margin-bottom: 15px;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 600;
            transition: all var(--transition-speed) ease;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
        }
        
        .capture-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .capture-btn:active {
            transform: translateY(1px);
        }
        
        .capture-btn::before {
            content: "📷";
            margin-right: 8px;
            font-size: 1.2em;
        }
        
        /* アノテーション用スタイル */
        .annotation-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .annotation-image {
            max-width: 100%;
            max-height: 100%;
            display: block;
            margin: 0 auto;
            transform-origin: center;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
            touch-action: none;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 15px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .annotation-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .annotation-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }
        
        .color-picker {
            margin-right: 10px;
            height: 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .size-control {
            display: flex;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            padding: 6px 12px;
            border-radius: var(--border-radius);
        }
        
        .size-control label {
            margin-right: 8px;
            font-weight: 500;
            color: var(--secondary-color);
        }
        
        .size-control input[type="range"] {
            width: 80px;
            accent-color: var(--primary-color);
        }
        
        .zoom-annotation-controls {
            position: absolute;
            bottom: 15px;
            right: 15px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-annotation-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            backdrop-filter: blur(2px);
        }
        
        .zoom-annotation-controls button:hover {
            background-color: rgba(0, 98, 204, 0.8);
            transform: translateY(-2px);
        }
        
        /* 寸法検知用スタイル */
        .dimension-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .dimension-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
            transition: transform var(--transition-speed) ease;
        }
        
        .dimension-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 15px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .dimension-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .dimension-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }
        
        .dimension-info {
            margin-top: 15px;
            padding: 15px;
            border-radius: var(--border-radius);
            background-color: white;
            box-shadow: var(--card-shadow);
            border: 1px solid var(--primary-light);
        }
        
        /* 異常検知用スタイル */
        .anomaly-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .anomaly-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 15px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .anomaly-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .anomaly-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }
        
        .anomaly-info {
            margin-top: 15px;
            padding: 15px;
            border-radius: var(--border-radius);
            background-color: white;
            box-shadow: var(--card-shadow);
            border: 1px solid var(--primary-light);
        }
        
        .heatmap-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            opacity: 0.6;
            transition: transform var(--transition-speed) ease;
        }
        
        /* モバイル対応 */
        @media (max-width: 1024px) {
            .camera-grid, .camera-grid-function {
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            }
            
            .sidebar {
                width: 80px;
                transform: translateX(0);
            }
            
            .sidebar.expanded {
                width: 240px;
            }
            
            .sidebar-logo h1 {
                display: none;
            }
            
            .sidebar.expanded .sidebar-logo h1 {
                display: block;
            }
            
            .tab .tab-text {
                display: none;
            }
            
            .sidebar.expanded .tab .tab-text {
                display: inline;
            }
            
            .sidebar-footer {
                display: none;
            }
            
            .sidebar.expanded .sidebar-footer {
                display: block;
            }
            
            .main-content {
                margin-left: 80px;
                padding: 15px;
            }
            
            .sidebar.expanded + .main-content {
                margin-left: 240px;
            }
            
            .mobile-toggle {
                display: block;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
                width: 240px;
            }
            
            .sidebar.mobile-visible {
                transform: translateX(0);
            }
            
            .tab .tab-text {
                display: inline;
            }
            
            .sidebar-logo h1 {
                display: block;
            }
            
            .sidebar-footer {
                display: block;
            }
            
            .main-content {
                margin-left: 0;
                width: 100%;
                padding: 10px;
            }
            
            .mobile-menu-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 99;
            }
            
            .mobile-menu-overlay.visible {
                display: block;
            }
            
            .mobile-menu-button {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                background-color: var(--primary-color);
                color: white;
                box-shadow: 0 3px 10px rgba(0, 98, 204, 0.3);
                border: none;
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 98;
                font-size: 24px;
                cursor: pointer;
                transition: all var(--transition-speed) ease;
            }
            
            .mobile-menu-button:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0, 98, 204, 0.4);
            }
            
            .camera-grid, .camera-grid-function {
                grid-template-columns: 1fr;
            }
            
            .camera-stream, .annotation-container, .dimension-container, .anomaly-container {
                height: 250px;
            }
            
            .header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .controls {
                margin-top: 15px;
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <!-- サイドバーナビゲーション -->
    <nav class="sidebar" id="sidebar">
        <div class="sidebar-logo">
            <div class="logo-container">
                <img src="/static/logo.png" alt="SCIEN Logo" class="logo-img">
                <h1>カメラ監視システム</h1>
            </div>
            <button class="mobile-toggle" id="sidebar-toggle">≡</button>
        </div>
        <div class="tabs">
            <div class="tab active" data-tab="streaming">
                <span class="tab-icon">📹</span>
                <span class="tab-text">ストリーミング</span>
            </div>
            <div class="tab" data-tab="annotation">
                <span class="tab-icon">✏️</span>
                <span class="tab-text">アノテーション</span>
            </div>
            <div class="tab" data-tab="dimension">
                <span class="tab-icon">📏</span>
                <span class="tab-text">寸法検知</span>
            </div>
            <div class="tab" data-tab="anomaly">
                <span class="tab-icon">🔍</span>
                <span class="tab-text">異常検知</span>
            </div>
        </div>
        <div class="sidebar-footer">
            <img src="/static/logo.png" alt="SCIEN Logo">
            <p>© 2025 SCIEN, Inc</p>
            <p>All rights reserved</p>
        </div>
    </nav>
    
    <!-- モバイルメニューオーバーレイ -->
    <div class="mobile-menu-overlay" id="mobile-overlay"></div>
    <button class="mobile-menu-button" id="mobile-menu-button">☰</button>
    
    <!-- メインコンテンツ -->
    <div class="main-content">
        <!-- ヘッダー -->
        <div class="header">
            <h2 class="header-title" id="page-title">ストリーミング</h2>
            <div class="controls">
                <button id="refresh-btn" class="button">
                    <span class="button-icon">🔄</span>更新
                </button>
                <button id="grid-toggle-btn" class="button">
                    <span class="button-icon">⊞</span>グリッド切替
                </button>
            </div>
        </div>
        
        <!-- ストリーミングタブ -->
        <div id="streaming-tab" class="tab-content active">
            <div id="camera-grid" class="camera-grid">
                <div class="placeholder">
                    <div class="placeholder-icon">🎥</div>
                    <p>カメラを読み込み中...</p>
                </div>
            </div>
        </div>
        
        <!-- アノテーションタブ -->
        <div id="annotation-tab" class="tab-content">
            <div id="annotation-grid" class="camera-grid-function">
                <div class="placeholder">
                    <div class="placeholder-icon">✏️</div>
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 寸法検知タブ -->
        <div id="dimension-tab" class="tab-content">
            <div id="dimension-grid" class="camera-grid-function">
                <div class="placeholder">
                    <div class="placeholder-icon">📏</div>
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 異常検知タブ -->
        <div id="anomaly-tab" class="tab-content">
            <div id="anomaly-grid" class="camera-grid-function">
                <div class="placeholder">
                    <div class="placeholder-icon">🔍</div>
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- スナップショットモーダル -->
    <div id="snapshot-modal" class="modal">
        <div class="modal-content">
            <span class="modal-close" id="close-modal">&times;</span>
            <h2 id="snapshot-title">スナップショット</h2>
            <img id="snapshot-img" class="snapshot-img" src="" alt="スナップショット">
        </div>
    </div>
    
    <script>
        // DOM要素の取得
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const mobileMenuButton = document.getElementById('mobile-menu-button');
        const mobileOverlay = document.getElementById('mobile-overlay');
        const pageTitle = document.getElementById('page-title');
        const cameraGrid = document.getElementById('camera-grid');
        const annotationGrid = document.getElementById('annotation-grid');
        const dimensionGrid = document.getElementById('dimension-grid');
        const anomalyGrid = document.getElementById('anomaly-grid');
        const refreshBtn = document.getElementById('refresh-btn');
        const gridToggleBtn = document.getElementById('grid-toggle-btn');
        const snapshotModal = document.getElementById('snapshot-modal');
        const closeModalBtn = document.getElementById('close-modal');
        const snapshotImg = document.getElementById('snapshot-img');
        const snapshotTitle = document.getElementById('snapshot-title');
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // カメラリスト
        let cameras = {};
        
        // グリッド列数
        let gridColumns = 'auto-fill';
        
        // 現在選択されているタブ
        let currentTab = 'streaming';
        
        // 撮影された画像データを保存するオブジェクト
        let capturedImages = {
            annotation: {},
            dimension: {},
            anomaly: {}
        };
        
        // サイドバーの切り替え
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('expanded');
        });
        
        // モバイルメニュー
        mobileMenuButton.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-visible');
            mobileOverlay.classList.toggle('visible');
        });
        
        mobileOverlay.addEventListener('click', () => {
            sidebar.classList.remove('mobile-visible');
            mobileOverlay.classList.remove('visible');
        });
        
        // タブの切り替え
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 現在のアクティブなタブを非アクティブにする
                document.querySelector('.tab.active').classList.remove('active');
                document.querySelector('.tab-content.active').classList.remove('active');
                
                // クリックされたタブをアクティブにする
                tab.classList.add('active');
                currentTab = tab.dataset.tab;
                pageTitle.textContent = tab.querySelector('.tab-text').textContent;
                document.getElementById(`${currentTab}-tab`).classList.add('active');
                
                // モバイルメニューを閉じる
                sidebar.classList.remove('mobile-visible');
                mobileOverlay.classList.remove('visible');
                
                // タブが変更されたときに必要な処理
                if (currentTab === 'streaming') {
                    fetchCameras();
                } else if (Object.keys(cameras).length === 0) {
                    fetchCameras().then(() => {
                        updateFunctionGrid(currentTab);
                    });
                } else {
                    updateFunctionGrid(currentTab);
                }
            });
        });
        
        // カメラ情報を取得する関数
        async function fetchCameras() {
            try {
                const response = await fetch('/api/cameras');
                if (!response.ok) {
                    throw new Error('サーバーからのレスポンスエラー');
                }
                
                cameras = await response.json();
                
                if (currentTab === 'streaming') {
                    renderCameraGrid();
                } else {
                    updateFunctionGrid(currentTab);
                }
                
                return cameras;
            } catch (error) {
                console.error('カメラ情報取得エラー:', error);
                const errorMessage = `
                    <div class="placeholder">
                        <div class="placeholder-icon">⚠️</div>
                        <p>カメラ情報の取得に失敗しました。</p>
                        <p>${error.message}</p>
                    </div>
                `;
                
                cameraGrid.innerHTML = errorMessage;
                annotationGrid.innerHTML = errorMessage;
                dimensionGrid.innerHTML = errorMessage;
                anomalyGrid.innerHTML = errorMessage;
            }
        }
        
        // 機能タブのグリッドを更新する関数
        function updateFunctionGrid(tabName) {
            const grid = document.getElementById(`${tabName}-grid`);
            
            if (Object.keys(cameras).length === 0) {
                grid.innerHTML = `
                    <div class="placeholder">
                        <div class="placeholder-icon">⚠️</div>
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                let functionContent = '';
                
                // 各タブごとの特有のコンテンツ
                if (tabName === 'annotation') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-annotation-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="annotation-container-${nodeId}" class="annotation-container" style="display: none;">
                                <img id="annotation-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="annotation-canvas-${nodeId}" class="annotation-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-annotation-${nodeId}">+</button>
                                    <button id="zoom-out-annotation-${nodeId}">-</button>
                                    <button id="zoom-reset-annotation-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="annotation-controls-${nodeId}" class="annotation-controls" style="display: none;">
                                <div>
                                    <button id="clear-annotation-${nodeId}">消去</button>
                                    <button id="save-annotation-${nodeId}">保存</button>
                                    <button id="recapture-annotation-${nodeId}">再撮影</button>
                                </div>
                                <div class="size-control">
                                    <label for="color-${nodeId}">色:</label>
                                    <input type="color" id="color-${nodeId}" class="color-picker" value="#ff0000">
                                    <label for="size-${nodeId}">太さ:</label>
                                    <input type="range" id="size-${nodeId}" min="1" max="20" value="5">
                                </div>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'dimension') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-dimension-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="dimension-container-${nodeId}" class="dimension-container" style="display: none;">
                                <img id="dimension-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="dimension-canvas-${nodeId}" class="dimension-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-dimension-${nodeId}">+</button>
                                    <button id="zoom-out-dimension-${nodeId}">-</button>
                                    <button id="zoom-reset-dimension-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="dimension-controls-${nodeId}" class="dimension-controls" style="display: none;">
                                <button id="clear-dimension-${nodeId}">リセット</button>
                                <button id="recapture-dimension-${nodeId}">再撮影</button>
                            </div>
                            <div id="dimension-info-${nodeId}" class="dimension-info" style="display: none;">
                                <p>2点間の距離を測定するには、画像上で2点をクリックしてください。</p>
                                <p id="dimension-result-${nodeId}">測定結果: まだ測定されていません</p>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'anomaly') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-anomaly-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="anomaly-container-${nodeId}" class="anomaly-container" style="display: none;">
                                <img id="anomaly-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="anomaly-canvas-${nodeId}" class="heatmap-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-anomaly-${nodeId}">+</button>
                                    <button id="zoom-out-anomaly-${nodeId}">-</button>
                                    <button id="zoom-reset-anomaly-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="anomaly-controls-${nodeId}" class="anomaly-controls" style="display: none;">
                                <button id="detect-anomaly-${nodeId}">異常検知実行</button>
                                <button id="recapture-anomaly-${nodeId}">再撮影</button>
                            </div>
                            <div id="anomaly-info-${nodeId}" class="anomaly-info" style="display: none;">
                                <p>異常検知の結果がここに表示されます。</p>
                                <p id="anomaly-result-${nodeId}">検知結果: まだ実行されていません</p>
                            </div>
                        </div>
                    `;
                }
                
                gridHTML += `
                    <div class="function-card" data-id="${nodeId}">
                        <h3>${camera.name}</h3>
                        ${functionContent}
                    </div>
                `;
            }
            
            if (gridHTML === '') {
                grid.innerHTML = `
                    <div class="placeholder">
                        <div class="placeholder-icon">⚠️</div>
                        <p>利用可能なカメラがありません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
            } else {
                grid.innerHTML = gridHTML;
                
                // 各タブごとのイベントリスナーを設定
                if (tabName === 'annotation') {
                    setupAnnotationEvents();
                } else if (tabName === 'dimension') {
                    setupDimensionEvents();
                } else if (tabName === 'anomaly') {
                    setupAnomalyEvents();
                }
            }
        }
        
        // アノテーションタブのイベントセットアップ
        function setupAnnotationEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-annotation-${nodeId}`);
                const container = document.getElementById(`annotation-container-${nodeId}`);
                const controls = document.getElementById(`annotation-controls-${nodeId}`);
                const clearBtn = document.getElementById(`clear-annotation-${nodeId}`);
                const saveBtn = document.getElementById(`save-annotation-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-annotation-${nodeId}`);
                const canvas = document.getElementById(`annotation-canvas-${nodeId}`);
                const img = document.getElementById(`annotation-img-${nodeId}`);
                const colorPicker = document.getElementById(`color-${nodeId}`);
                const sizeSlider = document.getElementById(`size-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.annotation[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.annotation[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupCanvas(canvas, img, nodeId);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
                
                // 保存ボタンのイベント
                saveBtn.addEventListener('click', () => {
                    const annotatedImage = combineImageAndCanvas(img, canvas);
                    // ここで保存処理（サンプルではダウンロードとして実装）
                    const link = document.createElement('a');
                    link.download = `annotation_${camera.name}_${new Date().toISOString()}.png`;
                    link.href = annotatedImage;
                    link.click();
                    alert('アノテーションを保存しました');
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 寸法検知タブのイベントセットアップ
        function setupDimensionEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-dimension-${nodeId}`);
                const container = document.getElementById(`dimension-container-${nodeId}`);
                const controls = document.getElementById(`dimension-controls-${nodeId}`);
                const infoBox = document.getElementById(`dimension-info-${nodeId}`);
                const clearBtn = document.getElementById(`clear-dimension-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-dimension-${nodeId}`);
                const canvas = document.getElementById(`dimension-canvas-${nodeId}`);
                const img = document.getElementById(`dimension-img-${nodeId}`);
                const resultText = document.getElementById(`dimension-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.dimension[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.dimension[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupDimensionCanvas(canvas, img, nodeId, resultText);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '測定結果: まだ測定されていません';
                    // 測定点をリセット
                    canvas.points = [];
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 異常検知タブのイベントセットアップ
        function setupAnomalyEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-anomaly-${nodeId}`);
                const container = document.getElementById(`anomaly-container-${nodeId}`);
                const controls = document.getElementById(`anomaly-controls-${nodeId}`);
                const infoBox = document.getElementById(`anomaly-info-${nodeId}`);
                const detectBtn = document.getElementById(`detect-anomaly-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-anomaly-${nodeId}`);
                const canvas = document.getElementById(`anomaly-canvas-${nodeId}`);
                const img = document.getElementById(`anomaly-img-${nodeId}`);
                const resultText = document.getElementById(`anomaly-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.anomaly[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.anomaly[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupAnomalyCanvas(canvas, img);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // 異常検知ボタンのイベント
                detectBtn.addEventListener('click', () => {
                    // 異常検知の実行（サンプルとして擬似的なヒートマップを生成）
                    detectAnomalies(canvas, img, resultText);
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '検知結果: まだ実行されていません';
                });
            }
        }
        
        // アノテーション用キャンバスのセットアップ
        function setupCanvas(canvas, img, nodeId) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let startDist = 0;
            
            // タッチでの描画
            canvas.addEventListener('touchstart', function(e) {
                if (e.touches.length === 1) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    lastX = (touch.clientX - rect.left) / scale - translateX;
                    lastY = (touch.clientY - rect.top) / scale - translateY;
                    isDrawing = true;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    startDist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                }
            });
            
            canvas.addEventListener('touchmove', function(e) {
                if (e.touches.length === 1 && isDrawing) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    const x = (touch.clientX - rect.left) / scale - translateX;
                    const y = (touch.clientY - rect.top) / scale - translateY;
                    
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(x, y);
                    ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                    ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                    ctx.lineCap = 'round';
                    ctx.stroke();
                    
                    lastX = x;
                    lastY = y;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    const dist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                    
                    const newScale = scale * (dist / startDist);
                    if (newScale > 0.5 && newScale < 5) {  // スケール制限
                        scale = newScale;
                        
                        // 中心点を計算
                        const centerX = (touch1.clientX + touch2.clientX) / 2;
                        const centerY = (touch1.clientY + touch2.clientY) / 2;
                        const rect = canvas.getBoundingClientRect();
                        const canvasCenterX = (centerX - rect.left) / scale - translateX;
                        const canvasCenterY = (centerY - rect.top) / scale - translateY;
                        
                        // 変換を更新
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    startDist = dist;
                }
            });
            
            canvas.addEventListener('touchend', function(e) {
                if (e.touches.length === 0) {
                    isDrawing = false;
                }
            });
            
            // マウスでの描画（デスクトップ用）
            canvas.addEventListener('mousedown', function(e) {
                const rect = canvas.getBoundingClientRect();
                lastX = (e.clientX - rect.left) / scale - translateX;
                lastY = (e.clientY - rect.top) / scale - translateY;
                isDrawing = true;
            });
            
            canvas.addEventListener('mousemove', function(e) {
                if (!isDrawing) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / scale - translateX;
                const y = (e.clientY - rect.top) / scale - translateY;
                
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                ctx.lineCap = 'round';
                ctx.stroke();
                
                lastX = x;
                lastY = y;
            });
            
            canvas.addEventListener('mouseup', function() {
                isDrawing = false;
            });
            
            canvas.addEventListener('mouseout', function() {
                isDrawing = false;
            });
        }
        
        // 寸法検知用キャンバスのセットアップ
        function setupDimensionCanvas(canvas, img, nodeId, resultText) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            canvas.points = [];
            
            // クリックイベント（タッチ含む）
            function handleClick(e) {
                e.preventDefault();
                
                const rect = canvas.getBoundingClientRect();
                let x, y;
                
                if (e.type === 'touchstart') {
                    x = e.touches[0].clientX - rect.left;
                    y = e.touches[0].clientY - rect.top;
                } else {
                    x = e.clientX - rect.left;
                    y = e.clientY - rect.top;
                }
                
                // 点の描画
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = 'red';
                ctx.fill();
                
                // テキスト
                ctx.fillStyle = 'white';
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;
                ctx.font = '12px Arial';
                ctx.strokeText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                ctx.fillText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                
                // 点を保存
                canvas.points.push({x, y});
                
                // 2点目が追加されたら線を引く
                if (canvas.points.length === 2) {
                    const p1 = canvas.points[0];
                    const p2 = canvas.points[1];
                    
                    // 線を引く
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = 'yellow';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // 距離の計算
                    const distance = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                    
                    // 中間点に距離を表示
                    const midX = (p1.x + p2.x) / 2;
                    const midY = (p1.y + p2.y) / 2;
                    
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = 'black';
                    ctx.font = '14px Arial';
                    
                    const dimensionText = `${distance.toFixed(1)}px`;
                    ctx.strokeText(dimensionText, midX, midY - 10);
                    ctx.fillText(dimensionText, midX, midY - 10);
                    
                    // 結果テキストの更新
                    resultText.textContent = `測定結果: 2点間の距離は ${distance.toFixed(1)}px です`;
                    
                    // リセットする（次の測定のため）
                    setTimeout(() => {
                        canvas.points = [];
                    }, 500);
                }
            }
            
            canvas.addEventListener('mousedown', handleClick);
            canvas.addEventListener('touchstart', handleClick);
        }
        
        // 異常検知用キャンバスのセットアップ
        function setupAnomalyCanvas(canvas, img) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }
        
        // 異常検知実行関数（サンプル実装）
        function detectAnomalies(canvas, img, resultText) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // サンプルとして擬似的なヒートマップを生成
            const gradients = [
                {pos: [0.2, 0.3], radius: 30, intensity: 0.8},
                {pos: [0.5, 0.6], radius: 40, intensity: 0.6},
                {pos: [0.8, 0.2], radius: 25, intensity: 0.9}
            ];
            
            // ヒートマップの描画
            for (const grad of gradients) {
                const centerX = canvas.width * grad.pos[0];
                const centerY = canvas.height * grad.pos[1];
                
                const grd = ctx.createRadialGradient(
                    centerX, centerY, 1,
                    centerX, centerY, grad.radius
                );
                
                grd.addColorStop(0, `rgba(255, 0, 0, ${grad.intensity})`);
                grd.addColorStop(1, 'rgba(255, 0, 0, 0)');
                
                ctx.fillStyle = grd;
                ctx.beginPath();
                ctx.arc(centerX, centerY, grad.radius, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // 結果テキストの更新
            resultText.textContent = `検知結果: ${gradients.length}箇所の異常が検出されました`;
        }
        
        // 画像とキャンバスを合成する関数
        function combineImageAndCanvas(img, canvas) {
            const combinedCanvas = document.createElement('canvas');
            combinedCanvas.width = canvas.width;
            combinedCanvas.height = canvas.height;
            
            const ctx = combinedCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            ctx.drawImage(canvas, 0, 0);
            
            return combinedCanvas.toDataURL('image/png');
        }
        
        // カメラグリッドを描画する関数
        function renderCameraGrid() {
            const cameraCount = Object.keys(cameras).length;
            
            if (cameraCount === 0) {
                cameraGrid.innerHTML = `
                    <div class="placeholder">
                        <div class="placeholder-icon">🎥</div>
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                gridHTML += `
                    <div class="camera-card" data-id="${nodeId}">
                        <div class="camera-header">
                            <h3 class="camera-title">
                                <span class="status-indicator status-${camera.status}"></span>
                                ${camera.name}
                            </h3>
                            <div class="camera-actions">
                                <button class="refresh-stream-btn" data-id="${nodeId}">リフレッシュ</button>
                                <button class="snapshot-btn" data-id="${nodeId}">スナップショット</button>
                            </div>
                        </div>
                        <div class="camera-stream" id="stream-${nodeId}" data-zoom="1" data-translate-x="0" data-translate-y="0">
                            <div class="loading">読み込み中...</div>
                            ${camera.status === 'running' 
                                ? `<img src="${camera.url}" alt="${camera.name}" onerror="handleStreamError('${nodeId}')">`
                                : `<div class="error-overlay">カメラ接続エラー</div>`
                            }
                            <div class="zoom-controls">
                                <button class="zoom-in-btn" data-id="${nodeId}">+</button>
                                <button class="zoom-out-btn" data-id="${nodeId}">-</button>
                                <button class="zoom-reset-btn" data-id="${nodeId}">↺</button>
                            </div>
                        </div>
                        <div class="camera-info">
                            <p><strong>ID</strong> ${nodeId}</p>
                            <p><strong>解像度</strong> ${camera.resolution ? camera.resolution.join(' x ') : '不明'}</p>
                            <p><strong>ステータス</strong> ${getStatusText(camera.status)}</p>
                            <p><strong>最終確認</strong> ${camera.last_seen}</p>
                        </div>
                    </div>
                `;
            }
            
            cameraGrid.innerHTML = gridHTML;
            
            // イベントリスナーを追加
            document.querySelectorAll('.refresh-stream-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    refreshStream(nodeId);
                });
            });
            
            document.querySelectorAll('.snapshot-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    takeSnapshot(nodeId);
                });
            });
            
            // ズームコントロールのイベントリスナーを設定
            setupZoomControls();
        }
        
        // ズームコントロールの設定
        function setupZoomControls() {
            // ストリーミングタブのズームコントロール
            document.querySelectorAll('.zoom-in-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.min(scale + 0.2, 3);  // 最大ズーム3倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                });
            });
            
            document.querySelectorAll('.zoom-out-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.max(scale - 0.2, 1);  // 最小ズーム等倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                    
                    // ズームアウト時に位置をリセットする場合
                    if (scale === 1) {
                        streamContainer.dataset.translateX = 0;
                        streamContainer.dataset.translateY = 0;
                    }
                });
            });
            
            document.querySelectorAll('.zoom-reset-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    streamContainer.dataset.zoom = 1;
                    streamContainer.dataset.translateX = 0;
                    streamContainer.dataset.translateY = 0;
                    img.style.transform = 'scale(1) translate(0px, 0px)';
                });
            });
            
            // ストリーミングタブでの画像ドラッグ機能
            document.querySelectorAll('.camera-stream img').forEach(img => {
                let isDragging = false;
                let startX, startY;
                let translateX = 0;
                let translateY = 0;
                
                // ドラッグ開始
                img.addEventListener('mousedown', (e) => {
                    const streamContainer = e.target.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                    
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    translateX = parseFloat(streamContainer.dataset.translateX || 0);
                    translateY = parseFloat(streamContainer.dataset.translateY || 0);
                    
                    streamContainer.classList.add('zoomed');
                });
                
                // ドラッグ中
                window.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;
                    
                    const x = e.clientX;
                    const y = e.clientY;
                    const deltaX = (x - startX) / 5;
                    const deltaY = (y - startY) / 5;
                    
                    const newTranslateX = translateX + deltaX;
                    const newTranslateY = translateY + deltaY;
                    
                    const streamContainer = img.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    
                    streamContainer.dataset.translateX = newTranslateX;
                    streamContainer.dataset.translateY = newTranslateY;
                    
                    img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                });
                
                // ドラッグ終了
                window.addEventListener('mouseup', () => {
                    isDragging = false;
                    document.querySelectorAll('.camera-stream.zoomed').forEach(el => {
                        el.classList.remove('zoomed');
                    });
                });
            });
            
            // タッチデバイス用の処理
            document.querySelectorAll('.camera-stream img').forEach(img => {
                const streamContainer = img.closest('.camera-stream');
                
                // ピンチズーム用の変数
                let initialDistance = 0;
                let initialScale = 1;
                
                // タッチ開始
                streamContainer.addEventListener('touchstart', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        initialDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        initialScale = parseFloat(streamContainer.dataset.zoom || 1);
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                        
                        const touch = e.touches[0];
                        streamContainer.dataset.startX = touch.clientX;
                        streamContainer.dataset.startY = touch.clientY;
                        streamContainer.dataset.translateStartX = parseFloat(streamContainer.dataset.translateX || 0);
                        streamContainer.dataset.translateStartY = parseFloat(streamContainer.dataset.translateY || 0);
                        streamContainer.classList.add('zoomed');
                        e.preventDefault();
                    }
                });
                
                // タッチ移動
                streamContainer.addEventListener('touchmove', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        const currentDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        
                        // スケールの計算
                        let newScale = initialScale * (currentDistance / initialDistance);
                        newScale = Math.max(1, Math.min(3, newScale));  // 1～3倍に制限
                        
                        streamContainer.dataset.zoom = newScale;
                        img.style.transform = `scale(${newScale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                        
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;
                        
                        const touch = e.touches[0];
                        const startX = parseFloat(streamContainer.dataset.startX || 0);
                        const startY = parseFloat(streamContainer.dataset.startY || 0);
                        const translateStartX = parseFloat(streamContainer.dataset.translateStartX || 0);
                        const translateStartY = parseFloat(streamContainer.dataset.translateStartY || 0);
                        
                        const deltaX = (touch.clientX - startX) / 5;
                        const deltaY = (touch.clientY - startY) / 5;
                        
                        const newTranslateX = translateStartX + deltaX;
                        const newTranslateY = translateStartY + deltaY;
                        
                        streamContainer.dataset.translateX = newTranslateX;
                        streamContainer.dataset.translateY = newTranslateY;
                        
                        img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                        
                        e.preventDefault();
                    }
                });
                
                // タッチ終了
                streamContainer.addEventListener('touchend', () => {
                    streamContainer.classList.remove('zoomed');
                });
            });
            
            // アノテーション、寸法、異常検知のズームコントロール
            ['annotation', 'dimension', 'anomaly'].forEach(tabType => {
                for (const [nodeId, camera] of Object.entries(cameras)) {
                    const zoomInBtn = document.getElementById(`zoom-in-${tabType}-${nodeId}`);
                    const zoomOutBtn = document.getElementById(`zoom-out-${tabType}-${nodeId}`);
                    const zoomResetBtn = document.getElementById(`zoom-reset-${tabType}-${nodeId}`);
                    const img = document.getElementById(`${tabType}-img-${nodeId}`);
                    const canvas = document.getElementById(`${tabType}-canvas-${nodeId}`);
                    
                    if (!zoomInBtn || !zoomOutBtn || !zoomResetBtn || !img || !canvas) continue;
                    
                    let scale = 1;
                    let translateX = 0;
                    let translateY = 0;
                    
                    // ズームイン
                    zoomInBtn.addEventListener('click', () => {
                        scale = Math.min(scale + 0.2, 3);
                        updateTransform();
                    });
                    
                    // ズームアウト
                    zoomOutBtn.addEventListener('click', () => {
                        scale = Math.max(scale - 0.2, 1);
                        if (scale === 1) {
                            translateX = 0;
                            translateY = 0;
                        }
                        updateTransform();
                    });
                    
                    // リセット
                    zoomResetBtn.addEventListener('click', () => {
                        scale = 1;
                        translateX = 0;
                        translateY = 0;
                        updateTransform();
                    });
                    
                    // 変換を更新する関数
                    function updateTransform() {
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    // ドラッグ処理
                    let isDragging = false;
                    let startX, startY;
                    
                    const container = document.getElementById(`${tabType}-container-${nodeId}`);
                    
                    container.addEventListener('mousedown', (e) => {
                        if (scale <= 1) return;
                        isDragging = true;
                        startX = e.clientX;
                        startY = e.clientY;
                    });
                    
                    container.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        
                        const x = e.clientX;
                        const y = e.clientY;
                        const deltaX = (x - startX) / 5;
                        const deltaY = (y - startY) / 5;
                        
                        translateX += deltaX;
                        translateY += deltaY;
                        
                        updateTransform();
                        
                        startX = x;
                        startY = y;
                    });
                    
                    container.addEventListener('mouseup', () => {
                        isDragging = false;
                    });
                    
                    container.addEventListener('mouseleave', () => {
                        isDragging = false;
                    });
                    
                    // タッチデバイス用
                    container.addEventListener('touchstart', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            container.dataset.initialDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            container.dataset.initialScale = scale;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            container.dataset.startX = touch.clientX;
                            container.dataset.startY = touch.clientY;
                            e.preventDefault();
                        }
                    });
                    
                    container.addEventListener('touchmove', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            const currentDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            
                            const initialDistance = parseFloat(container.dataset.initialDistance || 0);
                            const initialScale = parseFloat(container.dataset.initialScale || 1);
                            
                            if (initialDistance > 0) {
                                scale = Math.max(1, Math.min(3, initialScale * (currentDistance / initialDistance)));
                                updateTransform();
                            }
                            
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            const startX = parseFloat(container.dataset.startX || 0);
                            const startY = parseFloat(container.dataset.startY || 0);
                            
                            if (startX && startY) {
                                const deltaX = (touch.clientX - startX) / 5;
                                const deltaY = (touch.clientY - startY) / 5;
                                
                                translateX += deltaX;
                                translateY += deltaY;
                                
                                updateTransform();
                                
                                container.dataset.startX = touch.clientX;
                                container.dataset.startY = touch.clientY;
                            }
                            
                            e.preventDefault();
                        }
                    });
                }
            });
        }
        
        // スナップショットを取得する関数
        async function takeSnapshot(nodeId) {
            if (!cameras[nodeId]) return;
            
            try {
                const response = await fetch(`/api/snapshot/${nodeId}`);
                if (!response.ok) {
                    throw new Error('スナップショット取得エラー');
                }
                
                const data = await response.json();
                
                if (data.success && data.image) {
                    // スナップショットをモーダルに表示
                    snapshotTitle.textContent = `スナップショット: ${cameras[nodeId].name}`;
                    snapshotImg.src = `data:image/jpeg;base64,${data.image}`;
                    
                    // ズーム機能を追加<!DOCTYPE html>
<html>
<head>
    <title>カメラ監視システム</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
            --primary-color: #0062cc;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --border-radius: 8px;
            --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            --transition-speed: 0.3s;
            display: flex;
            min-height: 100vh;
        }
        
        /* サイドバーナビゲーション */
        .sidebar {
            width: 220px;
            background-color: #fff;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            z-index: 100;
            padding: 20px 0;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            transition: transform var(--transition-speed) ease;
        }
        
        .sidebar-logo {
            padding: 0 20px 20px;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .sidebar-logo h1 {
            font-size: 18px;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .mobile-toggle {
            display: none;
            font-size: 24px;
            background: none;
            border: none;
            color: var(--dark-color);
            cursor: pointer;
        }
        
        .tabs {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            font-weight: 500;
            color: var(--secondary-color);
            position: relative;
            display: flex;
            align-items: center;
            transition: all var(--transition-speed) ease;
            border-left: 3px solid transparent;
        }
        
        .tab:hover {
            background-color: rgba(0,0,0,0.03);
            color: var(--primary-color);
        }
        
        .tab.active {
            color: var(--primary-color);
            background-color: rgba(0, 98, 204, 0.05);
            border-left-color: var(--primary-color);
        }
        
        .tab .tab-icon {
            margin-right: 10px;
            width: 20px;
            text-align: center;
        }
        
        .sidebar-footer {
            padding: 10px 20px;
            border-top: 1px solid rgba(0,0,0,0.05);
            font-size: 12px;
            color: var(--secondary-color);
            text-align: center;
        }
        
        /* メインコンテンツエリア */
        .main-content {
            flex-grow: 1;
            padding: 20px;
            margin-left: 220px;
            transition: margin-left var(--transition-speed) ease;
        }
        
        /* ヘッダー部分 */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-radius: var(--border-radius);
        }
        
        .header-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
            color: var(--dark-color);
        }
        
        .controls {
            display: flex;
            gap: 10px;
        }
        
        .button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: var(--border-radius);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .button:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .button:active {
            transform: translateY(1px);
            box-shadow: 0 1px 2px rgba(0, 98, 204, 0.3);
        }
        
        .button-icon {
            margin-right: 6px;
            font-size: 16px;
        }
        
        /* タブコンテンツ */
        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease forwards;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* カメラグリッド */
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 24px;
            margin-bottom: 30px;
        }
        
        .camera-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
            border: 1px solid rgba(0,0,0,0.03);
        }
        
        .camera-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background-color: rgba(0,0,0,0.02);
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .camera-title {
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            font-size: 1rem;
            color: var(--dark-color);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            position: relative;
        }
        
        .status-indicator::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            border-radius: 50%;
            background-color: transparent;
            z-index: 1;
        }
        
        .status-running {
            background-color: var(--success-color);
            box-shadow: 0 0 0 2px rgba(40, 167, 69, 0.2);
        }
        
        .status-running::after {
            animation: pulse 2s infinite;
            border: 2px solid rgba(40, 167, 69, 0.4);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            70% { transform: scale(1.5); opacity: 0; }
            100% { transform: scale(1.5); opacity: 0; }
        }
        
        .status-error, .status-unreachable {
            background-color: var(--danger-color);
        }
        
        .status-initializing {
            background-color: var(--warning-color);
        }
        
        .camera-actions {
            display: flex;
            gap: 8px;
        }
        
        .camera-actions button {
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: var(--border-radius);
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .camera-actions button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .camera-stream {
            width: 100%;
            height: 300px;
            background-color: #111;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        
        .camera-stream img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            transition: transform var(--transition-speed) ease;
        }
        
        .camera-stream.zoomed {
            cursor: move;
        }
        
        .zoom-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: background-color var(--transition-speed) ease;
        }
        
        .zoom-controls button:hover {
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        .camera-info {
            padding: 15px 20px;
            font-size: 13px;
            color: var(--secondary-color);
            border-top: 1px solid rgba(0,0,0,0.05);
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        
        .camera-info p {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        
        .camera-info p strong {
            color: var(--dark-color);
            margin-right: 6px;
            display: inline-block;
            width: 80px;
        }
        
        .loading {
            color: white;
            font-size: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .loading::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-top: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .placeholder {
            text-align: center;
            padding: 30px;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .placeholder p {
            color: var(--secondary-color);
            margin: 10px 0;
        }
        
        /* モーダル用のスタイル */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
        }
        
        .modal.visible {
            opacity: 1;
        }
        
        .modal-content {
            background-color: white;
            border-radius: var(--border-radius);
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transform: scale(0.9);
            opacity: 0;
            transition: all var(--transition-speed) ease;
        }
        
        .modal.visible .modal-content {
            transform: scale(1);
            opacity: 1;
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: var(--secondary-color);
            transition: color var(--transition-speed) ease;
        }
        
        .modal-close:hover {
            color: var(--danger-color);
        }
        
        .snapshot-img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            transition: transform var(--transition-speed) ease;
        }
        
        .error-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(220, 53, 69, 0.8);
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
            text-align: center;
            padding: 20px;
        }
        
        .error-overlay button {
            margin-top: 15px;
            background-color: white;
            color: var(--danger-color);
            border: none;
            padding: 8px 16px;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 500;
            transition: all var(--transition-speed) ease;
        }
        
        .error-overlay button:hover {
            background-color: #f8f9fa;
            transform: translateY(-2px);
        }
        
        /* 機能カード */
        .camera-grid-function {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 24px;
            margin-top: 20px;
        }
        
        .function-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            padding: 0;
            border: 1px solid rgba(0,0,0,0.03);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
        }
        
        .function-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .function-card h3 {
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 15px 20px;
            margin: 0;
            background-color: rgba(0,0,0,0.02);
            font-weight: 600;
            font-size: 1rem;
            color: var(--dark-color);
        }
        
        .function-card-content {
            padding: 20px;
        }
        
        .capture-btn {
            display: flex;
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 600;
            transition: all var(--transition-speed) ease;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
        }
        
        .capture-btn:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .capture-btn::before {
            content: "📷";
            margin-right: 8px;
            font-size: 1.2em;
        }
        
        /* アノテーション用スタイル */
        .annotation-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .annotation-image {
            max-width: 100%;
            max-height: 100%;
            display: block;
            margin: 0 auto;
            transform-origin: center;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
            touch-action: none;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 12px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .annotation-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .annotation-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .color-picker {
            margin-right: 10px;
            height: 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .size-control {
            display: flex;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            padding: 6px 10px;
            border-radius: var(--border-radius);
        }
        
        .size-control label {
            margin-right: 8px;
            font-weight: 500;
            color: var(--secondary-color);
        }
        
        .size-control input[type="range"] {
            width: 80px;
            accent-color: var(--primary-color);
        }
        
        .zoom-annotation-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-annotation-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: background-color var(--transition-speed) ease;
        }
        
        .zoom-annotation-controls button:hover {
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        /* 寸法検知用スタイル */
        .dimension-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .dimension-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
            transition: transform var(--transition-speed) ease;
        }
        
        .dimension-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 12px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .dimension-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .dimension-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .dimension-info {
            margin-top: 15px;
            padding: 15px;
            border-radius: var(--border-radius);
            background-color: white;
            box-shadow: var(--card-shadow);
        }
        
        /* 異常検知用スタイル */
        .anomaly-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .anomaly-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 12px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .anomaly-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .anomaly-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .anomaly-info {
            margin-top: 15px;
            padding: 15px;
            border-radius: var(--border-radius);
            background-color: white;
            box-shadow: var(--card-shadow);
        }
        
        .heatmap-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            opacity: 0.6;
            transition: transform var(--transition-speed) ease;
        }
        
        /* モバイル対応 */
        @media (max-width: 1024px) {
            .camera-grid, .camera-grid-function {
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            }
            
            .sidebar {
                width: 80px;
                transform: translateX(0);
            }
            
            .sidebar.expanded {
                width: 220px;
            }
            
            .sidebar-logo h1 {
                display: none;
            }
            
            .sidebar.expanded .sidebar-logo h1 {
                display: block;
            }
            
            .tab .tab-text {
                display: none;
            }
            
            .sidebar.expanded .tab .tab-text {
                display: inline;
            }
            
            .sidebar-footer {
                display: none;
            }
            
            .sidebar.expanded .sidebar-footer {
                display: block;
            }
            
            .main-content {
                margin-left: 80px;
            }
            
            .sidebar.expanded + .main-content {
                margin-left: 220px;
            }
            
            .mobile-toggle {
                display: block;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
                width: 220px;
            }
            
            .sidebar.mobile-visible {
                transform: translateX(0);
            }
            
            .tab .tab-text {
                display: inline;
            }
            
            .sidebar-logo h1 {
                display: block;
            }
            
            .sidebar-footer {
                display: block;
            }
            
            .main-content {
                margin-left: 0;
                width: 100%;
            }
            
            .mobile-menu-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 99;
            }
            
            .mobile-menu-overlay.visible {
                display: block;
            }
            
            .mobile-menu-button {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: white;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                border: none;
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 98;
                font-size: 24px;
                color: var(--dark-color);
                cursor: pointer;
            }
            
            .camera-grid, .camera-grid-function {
                grid-template-columns: 1fr;
            }
            
            .camera-stream, .annotation-container, .dimension-container, .anomaly-container {
                height: 250px;
            }
            
            .header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .controls {
                margin-top: 10px;
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <!-- サイドバーナビゲーション -->
    <nav class="sidebar" id="sidebar">
        <div class="sidebar-logo">
            <h1>カメラ監視システム</h1>
            <button class="mobile-toggle" id="sidebar-toggle">≡</button>
        </div>
        <div class="tabs">
            <div class="tab active" data-tab="streaming">
                <span class="tab-icon">📹</span>
                <span class="tab-text">ストリーミング</span>
            </div>
            <div class="tab" data-tab="annotation">
                <span class="tab-icon">✏️</span>
                <span class="tab-text">アノテーション</span>
            </div>
            <div class="tab" data-tab="dimension">
                <span class="tab-icon">📏</span>
                <span class="tab-text">寸法検知</span>
            </div>
            <div class="tab" data-tab="anomaly">
                <span class="tab-icon">🔍</span>
                <span class="tab-text">異常検知</span>
            </div>
        </div>
        <div class="sidebar-footer">
            <p>© 2025 カメラ監視システム</p>
        </div>
    </nav>
    
    <!-- モバイルメニューオーバーレイ -->
    <div class="mobile-menu-overlay" id="mobile-overlay"></div>
    <button class="mobile-menu-button" id="mobile-menu-button">☰</button>
    
    <!-- メインコンテンツ -->
    <div class="main-content">
        <!-- ヘッダー -->
        <div class="header">
            <h2 class="header-title" id="page-title">ストリーミング</h2>
            <div class="controls">
                <button id="refresh-btn" class="button">
                    <span class="button-icon">🔄</span>更新
                </button>
                <button id="grid-toggle-btn" class="button">
                    <span class="button-icon">⊞</span>グリッド切替
                </button>
            </div>
        </div>
        
        <!-- ストリーミングタブ -->
        <div id="streaming-tab" class="tab-content active">
            <div id="camera-grid" class="camera-grid">
                <div class="placeholder">
                    <p>カメラを読み込み中...</p>
                </div>
            </div>
        </div>
        
        <!-- アノテーションタブ -->
        <div id="annotation-tab" class="tab-content">
            <div id="annotation-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 寸法検知タブ -->
        <div id="dimension-tab" class="tab-content">
            <div id="dimension-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 異常検知タブ -->
        <div id="anomaly-tab" class="tab-content">
            <div id="anomaly-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- スナップショットモーダル -->
    <div id="snapshot-modal" class="modal">
        <div class="modal-content">
            <span class="modal-close" id="close-modal">&times;</span>
            <h2 id="snapshot-title">スナップショット</h2>
            <img id="snapshot-img" class="snapshot-img" src="" alt="スナップショット">
        </div>
    </div>
    
    <script>
        // DOM要素の取得
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const mobileMenuButton = document.getElementById('mobile-menu-button');
        const mobileOverlay = document.getElementById('mobile-overlay');
        const pageTitle = document.getElementById('page-title');
        const cameraGrid = document.getElementById('camera-grid');
        const annotationGrid = document.getElementById('annotation-grid');
        const dimensionGrid = document.getElementById('dimension-grid');
        const anomalyGrid = document.getElementById('anomaly-grid');
        const refreshBtn = document.getElementById('refresh-btn');
        const gridToggleBtn = document.getElementById('grid-toggle-btn');
        const snapshotModal = document.getElementById('snapshot-modal');
        const closeModalBtn = document.getElementById('close-modal');
        const snapshotImg = document.getElementById('snapshot-img');
        const snapshotTitle = document.getElementById('snapshot-title');
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // カメラリスト
        let cameras = {};
        
        // グリッド列数
        let gridColumns = 'auto-fill';
        
        // 現在選択されているタブ
        let currentTab = 'streaming';
        
        // 撮影された画像データを保存するオブジェクト
        let capturedImages = {
            annotation: {},
            dimension: {},
            anomaly: {}
        };
        
        // サイドバーの切り替え
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('expanded');
        });
        
        // モバイルメニュー
        mobileMenuButton.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-visible');
            mobileOverlay.classList.toggle('visible');
        });
        
        mobileOverlay.addEventListener('click', () => {
            sidebar.classList.remove('mobile-visible');
            mobileOverlay.classList.remove('visible');
        });
        
        // タブの切り替え
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 現在のアクティブなタブを非アクティブにする
                document.querySelector('.tab.active').classList.remove('active');
                document.querySelector('.tab-content.active').classList.remove('active');
                
                // クリックされたタブをアクティブにする
                tab.classList.add('active');
                currentTab = tab.dataset.tab;
                pageTitle.textContent = tab.querySelector('.tab-text').textContent;
                document.getElementById(`${currentTab}-tab`).classList.add('active');
                
                // モバイルメニューを閉じる
                sidebar.classList.remove('mobile-visible');
                mobileOverlay.classList.remove('visible');
                
                // タブが変更されたときに必要な処理
                if (currentTab === 'streaming') {
                    fetchCameras();
                } else if (Object.keys(cameras).length === 0) {
                    fetchCameras().then(() => {
                        updateFunctionGrid(currentTab);
                    });
                } else {
                    updateFunctionGrid(currentTab);
                }
            });
        });
        
        // カメラ情報を取得する関数
        async function fetchCameras() {
            try {
                const response = await fetch('/api/cameras');
                if (!response.ok) {
                    throw new Error('サーバーからのレスポンスエラー');
                }
                
                cameras = await response.json();
                
                if (currentTab === 'streaming') {
                    renderCameraGrid();
                } else {
                    updateFunctionGrid(currentTab);
                }
                
                return cameras;
            } catch (error) {
                console.error('カメラ情報取得エラー:', error);
                const errorMessage = `
                    <div class="placeholder">
                        <p>カメラ情報の取得に失敗しました。</p>
                        <p>${error.message}</p>
                    </div>
                `;
                
                cameraGrid.innerHTML = errorMessage;
                annotationGrid.innerHTML = errorMessage;
                dimensionGrid.innerHTML = errorMessage;
                anomalyGrid.innerHTML = errorMessage;
            }
        }
        
        // 機能タブのグリッドを更新する関数
        function updateFunctionGrid(tabName) {
            const grid = document.getElementById(`${tabName}-grid`);
            
            if (Object.keys(cameras).length === 0) {
                grid.innerHTML = `
                    <div class="placeholder">
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                let functionContent = '';
                
                // 各タブごとの特有のコンテンツ
                if (tabName === 'annotation') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-annotation-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="annotation-container-${nodeId}" class="annotation-container" style="display: none;">
                                <img id="annotation-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="annotation-canvas-${nodeId}" class="annotation-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-annotation-${nodeId}">+</button>
                                    <button id="zoom-out-annotation-${nodeId}">-</button>
                                    <button id="zoom-reset-annotation-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="annotation-controls-${nodeId}" class="annotation-controls" style="display: none;">
                                <div>
                                    <button id="clear-annotation-${nodeId}">消去</button>
                                    <button id="save-annotation-${nodeId}">保存</button>
                                    <button id="recapture-annotation-${nodeId}">再撮影</button>
                                </div>
                                <div class="size-control">
                                    <label for="color-${nodeId}">色:</label>
                                    <input type="color" id="color-${nodeId}" class="color-picker" value="#ff0000">
                                    <label for="size-${nodeId}">太さ:</label>
                                    <input type="range" id="size-${nodeId}" min="1" max="20" value="5">
                                </div>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'dimension') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-dimension-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="dimension-container-${nodeId}" class="dimension-container" style="display: none;">
                                <img id="dimension-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="dimension-canvas-${nodeId}" class="dimension-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-dimension-${nodeId}">+</button>
                                    <button id="zoom-out-dimension-${nodeId}">-</button>
                                    <button id="zoom-reset-dimension-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="dimension-controls-${nodeId}" class="dimension-controls" style="display: none;">
                                <button id="clear-dimension-${nodeId}">リセット</button>
                                <button id="recapture-dimension-${nodeId}">再撮影</button>
                            </div>
                            <div id="dimension-info-${nodeId}" class="dimension-info" style="display: none;">
                                <p>2点間の距離を測定するには、画像上で2点をクリックしてください。</p>
                                <p id="dimension-result-${nodeId}">測定結果: まだ測定されていません</p>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'anomaly') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-anomaly-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="anomaly-container-${nodeId}" class="anomaly-container" style="display: none;">
                                <img id="anomaly-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="anomaly-canvas-${nodeId}" class="heatmap-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-anomaly-${nodeId}">+</button>
                                    <button id="zoom-out-anomaly-${nodeId}">-</button>
                                    <button id="zoom-reset-anomaly-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="anomaly-controls-${nodeId}" class="anomaly-controls" style="display: none;">
                                <button id="detect-anomaly-${nodeId}">異常検知実行</button>
                                <button id="recapture-anomaly-${nodeId}">再撮影</button>
                            </div>
                            <div id="anomaly-info-${nodeId}" class="anomaly-info" style="display: none;">
                                <p>異常検知の結果がここに表示されます。</p>
                                <p id="anomaly-result-${nodeId}">検知結果: まだ実行されていません</p>
                            </div>
                        </div>
                    `;
                }
                
                gridHTML += `
                    <div class="function-card" data-id="${nodeId}">
                        <h3>${camera.name}</h3>
                        ${functionContent}
                    </div>
                `;
            }
            
            if (gridHTML === '') {
                grid.innerHTML = `
                    <div class="placeholder">
                        <p>利用可能なカメラがありません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
            } else {
                grid.innerHTML = gridHTML;
                
                // 各タブごとのイベントリスナーを設定
                if (tabName === 'annotation') {
                    setupAnnotationEvents();
                } else if (tabName === 'dimension') {
                    setupDimensionEvents();
                } else if (tabName === 'anomaly') {
                    setupAnomalyEvents();
                }
            }
        }
        
        // アノテーションタブのイベントセットアップ
        function setupAnnotationEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-annotation-${nodeId}`);
                const container = document.getElementById(`annotation-container-${nodeId}`);
                const controls = document.getElementById(`annotation-controls-${nodeId}`);
                const clearBtn = document.getElementById(`clear-annotation-${nodeId}`);
                const saveBtn = document.getElementById(`save-annotation-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-annotation-${nodeId}`);
                const canvas = document.getElementById(`annotation-canvas-${nodeId}`);
                const img = document.getElementById(`annotation-img-${nodeId}`);
                const colorPicker = document.getElementById(`color-${nodeId}`);
                const sizeSlider = document.getElementById(`size-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.annotation[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.annotation[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupCanvas(canvas, img, nodeId);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
                
                // 保存ボタンのイベント
                saveBtn.addEventListener('click', () => {
                    const annotatedImage = combineImageAndCanvas(img, canvas);
                    // ここで保存処理（サンプルではダウンロードとして実装）
                    const link = document.createElement('a');
                    link.download = `annotation_${camera.name}_${new Date().toISOString()}.png`;
                    link.href = annotatedImage;
                    link.click();
                    alert('アノテーションを保存しました');
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 寸法検知タブのイベントセットアップ
        function setupDimensionEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-dimension-${nodeId}`);
                const container = document.getElementById(`dimension-container-${nodeId}`);
                const controls = document.getElementById(`dimension-controls-${nodeId}`);
                const infoBox = document.getElementById(`dimension-info-${nodeId}`);
                const clearBtn = document.getElementById(`clear-dimension-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-dimension-${nodeId}`);
                const canvas = document.getElementById(`dimension-canvas-${nodeId}`);
                const img = document.getElementById(`dimension-img-${nodeId}`);
                const resultText = document.getElementById(`dimension-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.dimension[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.dimension[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupDimensionCanvas(canvas, img, nodeId, resultText);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '測定結果: まだ測定されていません';
                    // 測定点をリセット
                    canvas.points = [];
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 異常検知タブのイベントセットアップ
        function setupAnomalyEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-anomaly-${nodeId}`);
                const container = document.getElementById(`anomaly-container-${nodeId}`);
                const controls = document.getElementById(`anomaly-controls-${nodeId}`);
                const infoBox = document.getElementById(`anomaly-info-${nodeId}`);
                const detectBtn = document.getElementById(`detect-anomaly-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-anomaly-${nodeId}`);
                const canvas = document.getElementById(`anomaly-canvas-${nodeId}`);
                const img = document.getElementById(`anomaly-img-${nodeId}`);
                const resultText = document.getElementById(`anomaly-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.anomaly[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.anomaly[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupAnomalyCanvas(canvas, img);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // 異常検知ボタンのイベント
                detectBtn.addEventListener('click', () => {
                    // 異常検知の実行（サンプルとして擬似的なヒートマップを生成）
                    detectAnomalies(canvas, img, resultText);
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '検知結果: まだ実行されていません';
                });
            }
        }
        
        // アノテーション用キャンバスのセットアップ
        function setupCanvas(canvas, img, nodeId) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let startDist = 0;
            
            // タッチでの描画
            canvas.addEventListener('touchstart', function(e) {
                if (e.touches.length === 1) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    lastX = (touch.clientX - rect.left) / scale - translateX;
                    lastY = (touch.clientY - rect.top) / scale - translateY;
                    isDrawing = true;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    startDist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                }
            });
            
            canvas.addEventListener('touchmove', function(e) {
                if (e.touches.length === 1 && isDrawing) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    const x = (touch.clientX - rect.left) / scale - translateX;
                    const y = (touch.clientY - rect.top) / scale - translateY;
                    
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(x, y);
                    ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                    ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                    ctx.lineCap = 'round';
                    ctx.stroke();
                    
                    lastX = x;
                    lastY = y;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    const dist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                    
                    const newScale = scale * (dist / startDist);
                    if (newScale > 0.5 && newScale < 5) {  // スケール制限
                        scale = newScale;
                        
                        // 中心点を計算
                        const centerX = (touch1.clientX + touch2.clientX) / 2;
                        const centerY = (touch1.clientY + touch2.clientY) / 2;
                        const rect = canvas.getBoundingClientRect();
                        const canvasCenterX = (centerX - rect.left) / scale - translateX;
                        const canvasCenterY = (centerY - rect.top) / scale - translateY;
                        
                        // 変換を更新
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    startDist = dist;
                }
            });
            
            canvas.addEventListener('touchend', function(e) {
                if (e.touches.length === 0) {
                    isDrawing = false;
                }
            });
            
            // マウスでの描画（デスクトップ用）
            canvas.addEventListener('mousedown', function(e) {
                const rect = canvas.getBoundingClientRect();
                lastX = (e.clientX - rect.left) / scale - translateX;
                lastY = (e.clientY - rect.top) / scale - translateY;
                isDrawing = true;
            });
            
            canvas.addEventListener('mousemove', function(e) {
                if (!isDrawing) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / scale - translateX;
                const y = (e.clientY - rect.top) / scale - translateY;
                
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                ctx.lineCap = 'round';
                ctx.stroke();
                
                lastX = x;
                lastY = y;
            });
            
            canvas.addEventListener('mouseup', function() {
                isDrawing = false;
            });
            
            canvas.addEventListener('mouseout', function() {
                isDrawing = false;
            });
        }
        
        // 寸法検知用キャンバスのセットアップ
        function setupDimensionCanvas(canvas, img, nodeId, resultText) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            canvas.points = [];
            
            // クリックイベント（タッチ含む）
            function handleClick(e) {
                e.preventDefault();
                
                const rect = canvas.getBoundingClientRect();
                let x, y;
                
                if (e.type === 'touchstart') {
                    x = e.touches[0].clientX - rect.left;
                    y = e.touches[0].clientY - rect.top;
                } else {
                    x = e.clientX - rect.left;
                    y = e.clientY - rect.top;
                }
                
                // 点の描画
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = 'red';
                ctx.fill();
                
                // テキスト
                ctx.fillStyle = 'white';
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;
                ctx.font = '12px Arial';
                ctx.strokeText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                ctx.fillText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                
                // 点を保存
                canvas.points.push({x, y});
                
                // 2点目が追加されたら線を引く
                if (canvas.points.length === 2) {
                    const p1 = canvas.points[0];
                    const p2 = canvas.points[1];
                    
                    // 線を引く
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = 'yellow';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // 距離の計算
                    const distance = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                    
                    // 中間点に距離を表示
                    const midX = (p1.x + p2.x) / 2;
                    const midY = (p1.y + p2.y) / 2;
                    
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = 'black';
                    ctx.font = '14px Arial';
                    
                    const dimensionText = `${distance.toFixed(1)}px`;
                    ctx.strokeText(dimensionText, midX, midY - 10);
                    ctx.fillText(dimensionText, midX, midY - 10);
                    
                    // 結果テキストの更新
                    resultText.textContent = `測定結果: 2点間の距離は ${distance.toFixed(1)}px です`;
                    
                    // リセットする（次の測定のため）
                    setTimeout(() => {
                        canvas.points = [];
                    }, 500);
                }
            }
            
            canvas.addEventListener('mousedown', handleClick);
            canvas.addEventListener('touchstart', handleClick);
        }
        
        // 異常検知用キャンバスのセットアップ
        function setupAnomalyCanvas(canvas, img) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }
        
        // 異常検知実行関数（サンプル実装）
        function detectAnomalies(canvas, img, resultText) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // サンプルとして擬似的なヒートマップを生成
            const gradients = [
                {pos: [0.2, 0.3], radius: 30, intensity: 0.8},
                {pos: [0.5, 0.6], radius: 40, intensity: 0.6},
                {pos: [0.8, 0.2], radius: 25, intensity: 0.9}
            ];
            
            // ヒートマップの描画
            for (const grad of gradients) {
                const centerX = canvas.width * grad.pos[0];
                const centerY = canvas.height * grad.pos[1];
                
                const grd = ctx.createRadialGradient(
                    centerX, centerY, 1,
                    centerX, centerY, grad.radius
                );
                
                grd.addColorStop(0, `rgba(255, 0, 0, ${grad.intensity})`);
                grd.addColorStop(1, 'rgba(255, 0, 0, 0)');
                
                ctx.fillStyle = grd;
                ctx.beginPath();
                ctx.arc(centerX, centerY, grad.radius, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // 結果テキストの更新
            resultText.textContent = `検知結果: ${gradients.length}箇所の異常が検出されました`;
        }
        
        // 画像とキャンバスを合成する関数
        function combineImageAndCanvas(img, canvas) {
            const combinedCanvas = document.createElement('canvas');
            combinedCanvas.width = canvas.width;
            combinedCanvas.height = canvas.height;
            
            const ctx = combinedCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            ctx.drawImage(canvas, 0, 0);
            
            return combinedCanvas.toDataURL('image/png');
        }
        
        // カメラグリッドを描画する関数
        function renderCameraGrid() {
            const cameraCount = Object.keys(cameras).length;
            
            if (cameraCount === 0) {
                cameraGrid.innerHTML = `
                    <div class="placeholder">
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                gridHTML += `
                    <div class="camera-card" data-id="${nodeId}">
                        <div class="camera-header">
                            <h3 class="camera-title">
                                <span class="status-indicator status-${camera.status}"></span>
                                ${camera.name}
                            </h3>
                            <div class="camera-actions">
                                <button class="refresh-stream-btn" data-id="${nodeId}">リフレッシュ</button>
                                <button class="snapshot-btn" data-id="${nodeId}">スナップショット</button>
                            </div>
                        </div>
                        <div class="camera-stream" id="stream-${nodeId}" data-zoom="1" data-translate-x="0" data-translate-y="0">
                            <div class="loading">読み込み中...</div>
                            ${camera.status === 'running' 
                                ? `<img src="${camera.url}" alt="${camera.name}" onerror="handleStreamError('${nodeId}')">`
                                : `<div class="error-overlay">カメラ接続エラー</div>`
                            }
                            <div class="zoom-controls">
                                <button class="zoom-in-btn" data-id="${nodeId}">+</button>
                                <button class="zoom-out-btn" data-id="${nodeId}">-</button>
                                <button class="zoom-reset-btn" data-id="${nodeId}">↺</button>
                            </div>
                        </div>
                        <div class="camera-info">
                            <p><strong>ID</strong> ${nodeId}</p>
                            <p><strong>解像度</strong> ${camera.resolution ? camera.resolution.join(' x ') : '不明'}</p>
                            <p><strong>ステータス</strong> ${getStatusText(camera.status)}</p>
                            <p><strong>最終確認</strong> ${camera.last_seen}</p>
                        </div>
                    </div>
                `;
            }
            
            cameraGrid.innerHTML = gridHTML;
            
            // イベントリスナーを追加
            document.querySelectorAll('.refresh-stream-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    refreshStream(nodeId);
                });
            });
            
            document.querySelectorAll('.snapshot-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    takeSnapshot(nodeId);
                });
            });
            
            // ズームコントロールのイベントリスナーを設定
            setupZoomControls();
        }
        
        // ズームコントロールの設定
        function setupZoomControls() {
            // ストリーミングタブのズームコントロール
            document.querySelectorAll('.zoom-in-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.min(scale + 0.2, 3);  // 最大ズーム3倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                });
            });
            
            document.querySelectorAll('.zoom-out-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.max(scale - 0.2, 1);  // 最小ズーム等倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                    
                    // ズームアウト時に位置をリセットする場合
                    if (scale === 1) {
                        streamContainer.dataset.translateX = 0;
                        streamContainer.dataset.translateY = 0;
                    }
                });
            });
            
            document.querySelectorAll('.zoom-reset-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    streamContainer.dataset.zoom = 1;
                    streamContainer.dataset.translateX = 0;
                    streamContainer.dataset.translateY = 0;
                    img.style.transform = 'scale(1) translate(0px, 0px)';
                });
            });
            
            // ストリーミングタブでの画像ドラッグ機能
            document.querySelectorAll('.camera-stream img').forEach(img => {
                let isDragging = false;
                let startX, startY;
                let translateX = 0;
                let translateY = 0;
                
                // ドラッグ開始
                img.addEventListener('mousedown', (e) => {
                    const streamContainer = e.target.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                    
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    translateX = parseFloat(streamContainer.dataset.translateX || 0);
                    translateY = parseFloat(streamContainer.dataset.translateY || 0);
                    
                    streamContainer.classList.add('zoomed');
                });
                
                // ドラッグ中
                window.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;
                    
                    const x = e.clientX;
                    const y = e.clientY;
                    const deltaX = (x - startX) / 5;
                    const deltaY = (y - startY) / 5;
                    
                    const newTranslateX = translateX + deltaX;
                    const newTranslateY = translateY + deltaY;
                    
                    const streamContainer = img.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    
                    streamContainer.dataset.translateX = newTranslateX;
                    streamContainer.dataset.translateY = newTranslateY;
                    
                    img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                });
                
                // ドラッグ終了
                window.addEventListener('mouseup', () => {
                    isDragging = false;
                    document.querySelectorAll('.camera-stream.zoomed').forEach(el => {
                        el.classList.remove('zoomed');
                    });
                });
            });
            
            // タッチデバイス用の処理
            document.querySelectorAll('.camera-stream img').forEach(img => {
                const streamContainer = img.closest('.camera-stream');
                
                // ピンチズーム用の変数
                let initialDistance = 0;
                let initialScale = 1;
                
                // タッチ開始
                streamContainer.addEventListener('touchstart', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        initialDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        initialScale = parseFloat(streamContainer.dataset.zoom || 1);
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                        
                        const touch = e.touches[0];
                        streamContainer.dataset.startX = touch.clientX;
                        streamContainer.dataset.startY = touch.clientY;
                        streamContainer.dataset.translateStartX = parseFloat(streamContainer.dataset.translateX || 0);
                        streamContainer.dataset.translateStartY = parseFloat(streamContainer.dataset.translateY || 0);
                        streamContainer.classList.add('zoomed');
                        e.preventDefault();
                    }
                });
                
                // タッチ移動
                streamContainer.addEventListener('touchmove', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        const currentDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        
                        // スケールの計算
                        let newScale = initialScale * (currentDistance / initialDistance);
                        newScale = Math.max(1, Math.min(3, newScale));  // 1～3倍に制限
                        
                        streamContainer.dataset.zoom = newScale;
                        img.style.transform = `scale(${newScale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                        
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;
                        
                        const touch = e.touches[0];
                        const startX = parseFloat(streamContainer.dataset.startX || 0);
                        const startY = parseFloat(streamContainer.dataset.startY || 0);
                        const translateStartX = parseFloat(streamContainer.dataset.translateStartX || 0);
                        const translateStartY = parseFloat(streamContainer.dataset.translateStartY || 0);
                        
                        const deltaX = (touch.clientX - startX) / 5;
                        const deltaY = (touch.clientY - startY) / 5;
                        
                        const newTranslateX = translateStartX + deltaX;
                        const newTranslateY = translateStartY + deltaY;
                        
                        streamContainer.dataset.translateX = newTranslateX;
                        streamContainer.dataset.translateY = newTranslateY;
                        
                        img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                        
                        e.preventDefault();
                    }
                });
                
                // タッチ終了
                streamContainer.addEventListener('touchend', () => {
                    streamContainer.classList.remove('zoomed');
                });
            });
            
            // アノテーション、寸法、異常検知のズームコントロール
            ['annotation', 'dimension', 'anomaly'].forEach(tabType => {
                for (const [nodeId, camera] of Object.entries(cameras)) {
                    const zoomInBtn = document.getElementById(`zoom-in-${tabType}-${nodeId}`);
                    const zoomOutBtn = document.getElementById(`zoom-out-${tabType}-${nodeId}`);
                    const zoomResetBtn = document.getElementById(`zoom-reset-${tabType}-${nodeId}`);
                    const img = document.getElementById(`${tabType}-img-${nodeId}`);
                    const canvas = document.getElementById(`${tabType}-canvas-${nodeId}`);
                    
                    if (!zoomInBtn || !zoomOutBtn || !zoomResetBtn || !img || !canvas) continue;
                    
                    let scale = 1;
                    let translateX = 0;
                    let translateY = 0;
                    
                    // ズームイン
                    zoomInBtn.addEventListener('click', () => {
                        scale = Math.min(scale + 0.2, 3);
                        updateTransform();
                    });
                    
                    // ズームアウト
                    zoomOutBtn.addEventListener('click', () => {
                        scale = Math.max(scale - 0.2, 1);
                        if (scale === 1) {
                            translateX = 0;
                            translateY = 0;
                        }
                        updateTransform();
                    });
                    
                    // リセット
                    zoomResetBtn.addEventListener('click', () => {
                        scale = 1;
                        translateX = 0;
                        translateY = 0;
                        updateTransform();
                    });
                    
                    // 変換を更新する関数
                    function updateTransform() {
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    // ドラッグ処理
                    let isDragging = false;
                    let startX, startY;
                    
                    const container = document.getElementById(`${tabType}-container-${nodeId}`);
                    
                    container.addEventListener('mousedown', (e) => {
                        if (scale <= 1) return;
                        isDragging = true;
                        startX = e.clientX;
                        startY = e.clientY;
                    });
                    
                    container.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        
                        const x = e.clientX;
                        const y = e.clientY;
                        const deltaX = (x - startX) / 5;
                        const deltaY = (y - startY) / 5;
                        
                        translateX += deltaX;
                        translateY += deltaY;
                        
                        updateTransform();
                        
                        startX = x;
                        startY = y;
                    });
                    
                    container.addEventListener('mouseup', () => {
                        isDragging = false;
                    });
                    
                    container.addEventListener('mouseleave', () => {
                        isDragging = false;
                    });
                    
                    // タッチデバイス用
                    container.addEventListener('touchstart', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            container.dataset.initialDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            container.dataset.initialScale = scale;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            container.dataset.startX = touch.clientX;
                            container.dataset.startY = touch.clientY;
                            e.preventDefault();
                        }
                    });
                    
                    container.addEventListener('touchmove', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            const currentDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            
                            const initialDistance = parseFloat(container.dataset.initialDistance || 0);
                            const initialScale = parseFloat(container.dataset.initialScale || 1);
                            
                            if (initialDistance > 0) {
                                scale = Math.max(1, Math.min(3, initialScale * (currentDistance / initialDistance)));
                                updateTransform();
                            }
                            
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            const startX = parseFloat(container.dataset.startX || 0);
                            const startY = parseFloat(container.dataset.startY || 0);
                            
                            if (startX && startY) {
                                const deltaX = (touch.clientX - startX) / 5;
                                const deltaY = (touch.clientY - startY) / 5;
                                
                                translateX += deltaX;
                                translateY += deltaY;
                                
                                updateTransform();
                                
                                container.dataset.startX = touch.clientX;
                                container.dataset.startY = touch.clientY;
                            }
                            
                            e.preventDefault();
                        }
                    });
                }
            });
        }
        
        // スナップショットを取得する関数
        async function takeSnapshot(nodeId) {
            if (!cameras[nodeId]) return;
            
            try {
                const response = await fetch(`/api/snapshot/${nodeId}`);
                if (!response.ok) {
                    throw new Error('スナップショット取得エラー');
                }
                
                const data = await response.json();
                
                if (data.success && data.image) {
                    // スナップショットをモーダルに表示
                    snapshotTitle.textContent = `スナップショット: ${cameras[nodeId].name}`;
                    snapshotImg.src = `data:image/jpeg;base64,${data.image}`;
                    
                    // ズーム機能を追加
                    let scale = 1;
                    let translateX = 0;
                    let translateY = 0;
                    let isDragging = false;
                    let startX, startY;
                    
                    // リセット
                    snapshotImg.style.transform = 'scale(1)';
                    
                    // ダブルクリックでズームイン/アウト
                    snapshotImg.addEventListener('dblclick', (e) => {
                        if (scale === 1) {
                            scale = 2;
                            // クリック位置を中心にズーム
                            const rect = snapshotImg.getBoundingClientRect();
                            const offsetX = (e.clientX - rect.left) / rect.width;
                            const offsetY = (e.clientY - rect.top) / rect.height;
                            translateX = (0.5 - offsetX) * 100;
                            translateY = (0.5 - offsetY) * 100;
                        } else {
                            scale = 1;
                            translateX = 0;
                            translateY = 0;
                        }
                        snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    });
                    
                    // ドラッグ
                    snapshotImg.addEventListener('mousedown', (e) => {
                        if (scale === 1) return;
                        isDragging = true;
                        startX = e.clientX;
                        startY = e.clientY;
                        e.preventDefault();
                    });
                    
                    window.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        const x = e.clientX;
                        const y = e.clientY;
                        translateX += (x - startX) / scale;
                        translateY += (y - startY) / scale;
                        snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        startX = x;
                        startY = y;
                    });
                    
                    window.addEventListener('mouseup', () => {
                        isDragging = false;
                    });
                    
                    // タッチ操作
                    let initialDistance = 0;
                    let initialScale = 1;
                    
                    snapshotImg.addEventListener('touchstart', (e) => {
                        if (e.touches.length === 2) {
                            initialDistance = Math.hypot(
                                e.touches[1].clientX - e.touches[0].clientX,
                                e.touches[1].clientY - e.touches[0].clientY
                            );
                            initialScale = scale;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            isDragging = true;
                            startX = e.touches[0].clientX;
                            startY = e.touches[0].clientY;
                            e.preventDefault();
                        }
                    });
                    
                    snapshotImg.addEventListener('touchmove', (e) => {
                        if (e.touches.length === 2) {
                            const distance = Math.hypot(
                                e.touches[1].clientX - e.touches[0].clientX,
                                e.touches[1].clientY - e.touches[0].clientY
                            );
                            scale = Math.max(1, Math.min(3, initialScale * (distance / initialDistance)));
                            snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && isDragging) {
                            const x = e.touches[0].clientX;
                            const y = e.touches[0].clientY;
                            translateX += (x - startX) / scale;
                            translateY += (y - startY) / scale;
                            snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                            startX = x;
                            startY = y;
                            e.preventDefault();
                        }
                    });
                    
                    snapshotImg.addEventListener('touchend', () => {
                        isDragging = false;
                    });
                    
                    // モーダルを表示
                    showModal(snapshotModal);
                } else {
                    alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                }
            } catch (error) {
                console.error('スナップショットエラー:', error);
                alert('スナップショット取得エラー: ' + error.message);
            }
        }
        
        // ストリームをリフレッシュする関数
        function refreshStream(nodeId) {
            if (!cameras[nodeId]) return;
            
            const streamContainer = document.getElementById(`stream-${nodeId}`);
            const camera = cameras[nodeId];
            
            if (camera.status === 'running') {
                streamContainer.innerHTML = `
                    <div class="loading">読み込み中...</div>
                    <img src="${camera.url}?t=${new Date().getTime()}" alt="${camera.name}" 
                         onerror="handleStreamError('${nodeId}')">
                    <div class="zoom-controls">
                        <button class="zoom-in-btn" data-id="${nodeId}">+</button>
                        <button class="zoom-out-btn" data-id="${nodeId}">-</button>
                        <button class="zoom-reset-btn" data-id="${nodeId}">↺</button>
                    </div>
                `;
                
                // ズームコントロールを再設定
                setupZoomControls();
            }
        }
        
        // ストリームエラーを処理する関数
        function handleStreamError(nodeId) {
            if (!cameras[nodeId]) return;
            
            const streamContainer = document.getElementById(`stream-${nodeId}`);
            
            if (streamContainer) {
                streamContainer.innerHTML = `
                    <div class="error-overlay">
                        ストリーム読み込みエラー
                        <button onclick="refreshStream<!DOCTYPE html>
<html>
<head>
    <title>カメラ監視システム</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
            --primary-color: #0062cc;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --border-radius: 8px;
            --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            --transition-speed: 0.3s;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px 5px;
            border-bottom: 1px solid rgba(0,0,0,0.08);
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-radius: var(--border-radius);
        }
        
        h1 {
            margin: 0;
            color: var(--dark-color);
            font-weight: 600;
            font-size: 1.5rem;
        }
        
        .controls {
            display: flex;
            gap: 10px;
        }
        
        .button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: var(--border-radius);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .button:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .button:active {
            transform: translateY(1px);
            box-shadow: 0 1px 2px rgba(0, 98, 204, 0.3);
        }
        
        /* タブスタイル */
        .tab-container {
            width: 100%;
            margin-bottom: 24px;
        }
        
        .tabs {
            display: flex;
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            position: relative;
            z-index: 1;
        }
        
        .tab {
            flex: 1;
            padding: 16px 8px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
            position: relative;
            overflow: hidden;
        }
        
        .tab::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: var(--primary-color);
            transform: scaleX(0);
            transition: transform var(--transition-speed) ease;
        }
        
        .tab:hover {
            color: var(--primary-color);
            background-color: rgba(0, 98, 204, 0.03);
        }
        
        .tab.active {
            color: var(--primary-color);
            background-color: rgba(0, 98, 204, 0.05);
        }
        
        .tab.active::before {
            transform: scaleX(1);
        }
        
        .tab-content {
            display: none;
            padding-top: 20px;
            animation: fadeIn 0.5s ease forwards;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 24px;
            margin-bottom: 30px;
        }
        
        .camera-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
            border: 1px solid rgba(0,0,0,0.03);
        }
        
        .camera-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background-color: rgba(0,0,0,0.02);
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .camera-title {
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            font-size: 1rem;
            color: var(--dark-color);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            position: relative;
        }
        
        .status-indicator::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            border-radius: 50%;
            background-color: transparent;
            z-index: 1;
        }
        
        .status-running {
            background-color: var(--success-color);
            box-shadow: 0 0 0 2px rgba(40, 167, 69, 0.2);
        }
        
        .status-running::after {
            animation: pulse 2s infinite;
            border: 2px solid rgba(40, 167, 69, 0.4);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            70% { transform: scale(1.5); opacity: 0; }
            100% { transform: scale(1.5); opacity: 0; }
        }
        
        .status-error, .status-unreachable {
            background-color: var(--danger-color);
        }
        
        .status-initializing {
            background-color: var(--warning-color);
        }
        
        .camera-actions {
            display: flex;
            gap: 8px;
        }
        
        .camera-actions button {
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: var(--border-radius);
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .camera-actions button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .camera-stream {
            width: 100%;
            height: 300px;
            background-color: #111;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        
        .camera-stream img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            transition: transform var(--transition-speed) ease;
        }
        
        .camera-stream.zoomed {
            cursor: move;
        }
        
        .zoom-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: background-color var(--transition-speed) ease;
        }
        
        .zoom-controls button:hover {
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        .camera-info {
            padding: 15px 20px;
            font-size: 13px;
            color: var(--secondary-color);
            border-top: 1px solid rgba(0,0,0,0.05);
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        
        .camera-info p {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        
        .camera-info p strong {
            color: var(--dark-color);
            margin-right: 6px;
            display: inline-block;
            width: 80px;
        }
        
        .loading {
            color: white;
            font-size: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .loading::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-top: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .placeholder {
            text-align: center;
            padding: 30px;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .placeholder p {
            color: var(--secondary-color);
            margin: 10px 0;
        }
        
        /* モーダル用のスタイル */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
        }
        
        .modal.visible {
            opacity: 1;
        }
        
        .modal-content {
            background-color: white;
            border-radius: var(--border-radius);
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transform: scale(0.9);
            opacity: 0;
            transition: all var(--transition-speed) ease;
        }
        
        .modal.visible .modal-content {
            transform: scale(1);
            opacity: 1;
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: var(--secondary-color);
            transition: color var(--transition-speed) ease;
        }
        
        .modal-close:hover {
            color: var(--danger-color);
        }
        
        .snapshot-img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            transition: transform var(--transition-speed) ease;
        }
        
        .error-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(220, 53, 69, 0.8);
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
            text-align: center;
            padding: 20px;
        }
        
        .error-overlay button {
            margin-top: 15px;
            background-color: white;
            color: var(--danger-color);
            border: none;
            padding: 8px 16px;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 500;
            transition: all var(--transition-speed) ease;
        }
        
        .error-overlay button:hover {
            background-color: #f8f9fa;
            transform: translateY(-2px);
        }
        
        /* アノテーション用スタイル */
        .annotation-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            background-color: #111;
        }
        
        .annotation-image {
            max-width: 100%;
            max-height: 100%;
            display: block;
            margin: 0 auto;
            transform-origin: center;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
            touch-action: none;
            transition: transform var(--transition-speed) ease;
        }
        
        .annotation-controls {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 12px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .annotation-controls button {
            margin-right: 10px;
            background-color: white;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 8px 14px;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all var(--transition-speed) ease;
            color: var(--secondary-color);
        }
        
        .annotation-controls button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .color-picker {
            margin-right: 10px;
            height: 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .size-control {
            display: flex;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            padding: 6px 10px;
            border-radius: var(--border-radius);
        }
        
        .size-control label {
            margin-right: 8px;
            font-weight: 500;
            color: var(--secondary-color);
        }
        
        .size-control input[type="range"] {
            width: 80px;
            accent-color: var(--primary-color);
        }
        
        .zoom-annotation-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
            z-index: 5;
        }
        
        .zoom-annotation-controls button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: background-color var(--transition-speed) ease;
        }
        
        .zoom-annotation-controls button:hover {
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        /* 寸法検知用スタイル */
        .dimension-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            background-color: #222;
        }
        
        .dimension-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            cursor: crosshair;
        }
        
        .dimension-controls {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
        }
        
        .dimension-info {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* 異常検知用スタイル */
        .anomaly-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            background-color: #222;
        }
        
        .anomaly-controls {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
        }
        
        .anomaly-info {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .heatmap-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
            opacity: 0.6;
        }
        
        .camera-grid-function {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 24px;
            margin-top: 20px;
        }
        
        .function-card {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
            padding: 0;
            border: 1px solid rgba(0,0,0,0.03);
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
        }
        
        .function-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .function-card h3 {
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 15px 20px;
            margin: 0;
            background-color: rgba(0,0,0,0.02);
            font-weight: 600;
            font-size: 1rem;
            color: var(--dark-color);
        }
        
        .function-card-content {
            padding: 20px;
        }
        
        .capture-btn {
            display: flex;
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 600;
            transition: all var(--transition-speed) ease;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0, 98, 204, 0.2);
        }
        
        .capture-btn:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 98, 204, 0.3);
        }
        
        .capture-btn::before {
            content: "📷";
            margin-right: 8px;
            font-size: 1.2em;
        }
        
        /* iPadに最適化 */
        @media (max-width: 1024px) {
            .camera-grid, .camera-grid-function {
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            }
            
            .container {
                padding: 10px;
            }
            
            .tabs {
                flex-wrap: wrap;
            }
            
            .tab {
                flex-basis: 50%;
            }
        }
        
        @media (max-width: 768px) {
            .camera-grid, .camera-grid-function {
                grid-template-columns: 1fr;
            }
            
            .camera-stream, .annotation-container, .dimension-container, .anomaly-container {
                height: 250px;
            }
            
            header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .controls {
                margin-top: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>カメラ監視システム</h1>
            <div class="controls">
                <button id="refresh-btn" class="button">更新</button>
                <button id="grid-toggle-btn" class="button">グリッド切替</button>
            </div>
        </header>
        
        <!-- タブコンテナ -->
        <div class="tab-container">
            <div class="tabs">
                <div class="tab active" data-tab="streaming">ストリーミング</div>
                <div class="tab" data-tab="annotation">アノテーション</div>
                <div class="tab" data-tab="dimension">寸法検知</div>
                <div class="tab" data-tab="anomaly">異常検知</div>
            </div>
        </div>
        
        <!-- ストリーミングタブ -->
        <div id="streaming-tab" class="tab-content active">
            <div id="camera-grid" class="camera-grid">
                <div class="placeholder">
                    <p>カメラを読み込み中...</p>
                </div>
            </div>
        </div>
        
        <!-- アノテーションタブ -->
        <div id="annotation-tab" class="tab-content">
            <div id="annotation-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 寸法検知タブ -->
        <div id="dimension-tab" class="tab-content">
            <div id="dimension-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
        
        <!-- 異常検知タブ -->
        <div id="anomaly-tab" class="tab-content">
            <div id="anomaly-grid" class="camera-grid-function">
                <div class="placeholder">
                    <p>カメラを選択して静止画を撮影してください。</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- スナップショットモーダル -->
    <div id="snapshot-modal" class="modal">
        <div class="modal-content">
            <span class="modal-close" id="close-modal">&times;</span>
            <h2 id="snapshot-title">スナップショット</h2>
            <img id="snapshot-img" class="snapshot-img" src="" alt="スナップショット">
        </div>
    </div>
    
    <script>
        // DOM要素の取得
        const cameraGrid = document.getElementById('camera-grid');
        const annotationGrid = document.getElementById('annotation-grid');
        const dimensionGrid = document.getElementById('dimension-grid');
        const anomalyGrid = document.getElementById('anomaly-grid');
        const refreshBtn = document.getElementById('refresh-btn');
        const gridToggleBtn = document.getElementById('grid-toggle-btn');
        const snapshotModal = document.getElementById('snapshot-modal');
        const closeModalBtn = document.getElementById('close-modal');
        const snapshotImg = document.getElementById('snapshot-img');
        const snapshotTitle = document.getElementById('snapshot-title');
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // カメラリスト
        let cameras = {};
        
        // グリッド列数
        let gridColumns = 'auto-fill';
        
        // 現在選択されているタブ
        let currentTab = 'streaming';
        
        // 撮影された画像データを保存するオブジェクト
        let capturedImages = {
            annotation: {},
            dimension: {},
            anomaly: {}
        };
        
        // タブの切り替え
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 現在のアクティブなタブを非アクティブにする
                document.querySelector('.tab.active').classList.remove('active');
                document.querySelector('.tab-content.active').classList.remove('active');
                
                // クリックされたタブをアクティブにする
                tab.classList.add('active');
                currentTab = tab.dataset.tab;
                document.getElementById(`${currentTab}-tab`).classList.add('active');
                
                // タブが変更されたときに必要な処理
                if (currentTab === 'streaming') {
                    fetchCameras();
                } else if (Object.keys(cameras).length === 0) {
                    fetchCameras().then(() => {
                        updateFunctionGrid(currentTab);
                    });
                } else {
                    updateFunctionGrid(currentTab);
                }
            });
        });
        
        // カメラ情報を取得する関数
        async function fetchCameras() {
            try {
                const response = await fetch('/api/cameras');
                if (!response.ok) {
                    throw new Error('サーバーからのレスポンスエラー');
                }
                
                cameras = await response.json();
                
                if (currentTab === 'streaming') {
                    renderCameraGrid();
                } else {
                    updateFunctionGrid(currentTab);
                }
                
                return cameras;
            } catch (error) {
                console.error('カメラ情報取得エラー:', error);
                const errorMessage = `
                    <div class="placeholder">
                        <p>カメラ情報の取得に失敗しました。</p>
                        <p>${error.message}</p>
                    </div>
                `;
                
                cameraGrid.innerHTML = errorMessage;
                annotationGrid.innerHTML = errorMessage;
                dimensionGrid.innerHTML = errorMessage;
                anomalyGrid.innerHTML = errorMessage;
            }
        }
        
        // 機能タブのグリッドを更新する関数
        function updateFunctionGrid(tabName) {
            const grid = document.getElementById(`${tabName}-grid`);
            
            if (Object.keys(cameras).length === 0) {
                grid.innerHTML = `
                    <div class="placeholder">
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                let functionContent = '';
                
                // 各タブごとの特有のコンテンツ
                if (tabName === 'annotation') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-annotation-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="annotation-container-${nodeId}" class="annotation-container" style="display: none;">
                                <img id="annotation-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="annotation-canvas-${nodeId}" class="annotation-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-annotation-${nodeId}">+</button>
                                    <button id="zoom-out-annotation-${nodeId}">-</button>
                                    <button id="zoom-reset-annotation-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="annotation-controls-${nodeId}" class="annotation-controls" style="display: none;">
                                <div>
                                    <button id="clear-annotation-${nodeId}">消去</button>
                                    <button id="save-annotation-${nodeId}">保存</button>
                                    <button id="recapture-annotation-${nodeId}">再撮影</button>
                                </div>
                                <div class="size-control">
                                    <label for="color-${nodeId}">色:</label>
                                    <input type="color" id="color-${nodeId}" class="color-picker" value="#ff0000">
                                    <label for="size-${nodeId}">太さ:</label>
                                    <input type="range" id="size-${nodeId}" min="1" max="20" value="5">
                                </div>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'dimension') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-dimension-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="dimension-container-${nodeId}" class="dimension-container" style="display: none;">
                                <img id="dimension-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="dimension-canvas-${nodeId}" class="dimension-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-dimension-${nodeId}">+</button>
                                    <button id="zoom-out-dimension-${nodeId}">-</button>
                                    <button id="zoom-reset-dimension-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="dimension-controls-${nodeId}" class="dimension-controls" style="display: none;">
                                <button id="clear-dimension-${nodeId}">リセット</button>
                                <button id="recapture-dimension-${nodeId}">再撮影</button>
                            </div>
                            <div id="dimension-info-${nodeId}" class="dimension-info" style="display: none;">
                                <p>2点間の距離を測定するには、画像上で2点をクリックしてください。</p>
                                <p id="dimension-result-${nodeId}">測定結果: まだ測定されていません</p>
                            </div>
                        </div>
                    `;
                } else if (tabName === 'anomaly') {
                    functionContent = `
                        <div class="function-card-content">
                            <button id="capture-anomaly-${nodeId}" class="capture-btn">静止画を撮影</button>
                            <div id="anomaly-container-${nodeId}" class="anomaly-container" style="display: none;">
                                <img id="anomaly-img-${nodeId}" class="annotation-image" src="" alt="${camera.name}の画像" />
                                <canvas id="anomaly-canvas-${nodeId}" class="heatmap-canvas"></canvas>
                                <div class="zoom-annotation-controls">
                                    <button id="zoom-in-anomaly-${nodeId}">+</button>
                                    <button id="zoom-out-anomaly-${nodeId}">-</button>
                                    <button id="zoom-reset-anomaly-${nodeId}">↺</button>
                                </div>
                            </div>
                            <div id="anomaly-controls-${nodeId}" class="anomaly-controls" style="display: none;">
                                <button id="detect-anomaly-${nodeId}">異常検知実行</button>
                                <button id="recapture-anomaly-${nodeId}">再撮影</button>
                            </div>
                            <div id="anomaly-info-${nodeId}" class="anomaly-info" style="display: none;">
                                <p>異常検知の結果がここに表示されます。</p>
                                <p id="anomaly-result-${nodeId}">検知結果: まだ実行されていません</p>
                            </div>
                        </div>
                    `;
                }
                
                gridHTML += `
                    <div class="function-card" data-id="${nodeId}">
                        <h3>${camera.name}</h3>
                        ${functionContent}
                    </div>
                `;
            }
            
            if (gridHTML === '') {
                grid.innerHTML = `
                    <div class="placeholder">
                        <p>利用可能なカメラがありません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
            } else {
                grid.innerHTML = gridHTML;
                
                // 各タブごとのイベントリスナーを設定
                if (tabName === 'annotation') {
                    setupAnnotationEvents();
                } else if (tabName === 'dimension') {
                    setupDimensionEvents();
                } else if (tabName === 'anomaly') {
                    setupAnomalyEvents();
                }
            }
        }
        
        // アノテーションタブのイベントセットアップ
        function setupAnnotationEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-annotation-${nodeId}`);
                const container = document.getElementById(`annotation-container-${nodeId}`);
                const controls = document.getElementById(`annotation-controls-${nodeId}`);
                const clearBtn = document.getElementById(`clear-annotation-${nodeId}`);
                const saveBtn = document.getElementById(`save-annotation-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-annotation-${nodeId}`);
                const canvas = document.getElementById(`annotation-canvas-${nodeId}`);
                const img = document.getElementById(`annotation-img-${nodeId}`);
                const colorPicker = document.getElementById(`color-${nodeId}`);
                const sizeSlider = document.getElementById(`size-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.annotation[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.annotation[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupCanvas(canvas, img, nodeId);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
                
                // 保存ボタンのイベント
                saveBtn.addEventListener('click', () => {
                    const annotatedImage = combineImageAndCanvas(img, canvas);
                    // ここで保存処理（サンプルではダウンロードとして実装）
                    const link = document.createElement('a');
                    link.download = `annotation_${camera.name}_${new Date().toISOString()}.png`;
                    link.href = annotatedImage;
                    link.click();
                    alert('アノテーションを保存しました');
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 寸法検知タブのイベントセットアップ
        function setupDimensionEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-dimension-${nodeId}`);
                const container = document.getElementById(`dimension-container-${nodeId}`);
                const controls = document.getElementById(`dimension-controls-${nodeId}`);
                const infoBox = document.getElementById(`dimension-info-${nodeId}`);
                const clearBtn = document.getElementById(`clear-dimension-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-dimension-${nodeId}`);
                const canvas = document.getElementById(`dimension-canvas-${nodeId}`);
                const img = document.getElementById(`dimension-img-${nodeId}`);
                const resultText = document.getElementById(`dimension-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.dimension[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.dimension[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupDimensionCanvas(canvas, img, nodeId, resultText);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // クリアボタンのイベント
                clearBtn.addEventListener('click', () => {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '測定結果: まだ測定されていません';
                    // 測定点をリセット
                    canvas.points = [];
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            }
        }
        
        // 異常検知タブのイベントセットアップ
        function setupAnomalyEvents() {
            for (const [nodeId, camera] of Object.entries(cameras)) {
                if (camera.status !== 'running') continue;
                
                const captureBtn = document.getElementById(`capture-anomaly-${nodeId}`);
                const container = document.getElementById(`anomaly-container-${nodeId}`);
                const controls = document.getElementById(`anomaly-controls-${nodeId}`);
                const infoBox = document.getElementById(`anomaly-info-${nodeId}`);
                const detectBtn = document.getElementById(`detect-anomaly-${nodeId}`);
                const recaptureBtn = document.getElementById(`recapture-anomaly-${nodeId}`);
                const canvas = document.getElementById(`anomaly-canvas-${nodeId}`);
                const img = document.getElementById(`anomaly-img-${nodeId}`);
                const resultText = document.getElementById(`anomaly-result-${nodeId}`);
                
                if (!captureBtn) continue;
                
                // 撮影ボタンのイベント
                captureBtn.addEventListener('click', async () => {
                    try {
                        const response = await fetch(`/api/snapshot/${nodeId}`);
                        if (!response.ok) {
                            throw new Error('スナップショット取得エラー');
                        }
                        
                        const data = await response.json();
                        
                        if (data.success && data.image) {
                            // 画像データを保存
                            capturedImages.anomaly[nodeId] = `data:image/jpeg;base64,${data.image}`;
                            
                            // 画像を表示
                            img.src = capturedImages.anomaly[nodeId];
                            container.style.display = 'block';
                            controls.style.display = 'flex';
                            infoBox.style.display = 'block';
                            captureBtn.style.display = 'none';
                            
                            // 画像読み込み完了後にキャンバスをセットアップ
                            img.onload = () => {
                                setupAnomalyCanvas(canvas, img);
                            };
                        } else {
                            alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                        }
                    } catch (error) {
                        console.error('スナップショットエラー:', error);
                        alert('スナップショット取得エラー: ' + error.message);
                    }
                });
                
                // 異常検知ボタンのイベント
                detectBtn.addEventListener('click', () => {
                    // 異常検知の実行（サンプルとして擬似的なヒートマップを生成）
                    detectAnomalies(canvas, img, resultText);
                });
                
                // 再撮影ボタンのイベント
                recaptureBtn.addEventListener('click', () => {
                    container.style.display = 'none';
                    controls.style.display = 'none';
                    infoBox.style.display = 'none';
                    captureBtn.style.display = 'block';
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    resultText.textContent = '検知結果: まだ実行されていません';
                });
            }
        }
        
        // アノテーション用キャンバスのセットアップ
        function setupCanvas(canvas, img, nodeId) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let startDist = 0;
            
            // タッチでの描画
            canvas.addEventListener('touchstart', function(e) {
                if (e.touches.length === 1) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    lastX = (touch.clientX - rect.left) / scale - translateX;
                    lastY = (touch.clientY - rect.top) / scale - translateY;
                    isDrawing = true;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    startDist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                }
            });
            
            canvas.addEventListener('touchmove', function(e) {
                if (e.touches.length === 1 && isDrawing) {
                    // 1本指の場合は描画
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    const x = (touch.clientX - rect.left) / scale - translateX;
                    const y = (touch.clientY - rect.top) / scale - translateY;
                    
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(x, y);
                    ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                    ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                    ctx.lineCap = 'round';
                    ctx.stroke();
                    
                    lastX = x;
                    lastY = y;
                } else if (e.touches.length === 2) {
                    // 2本指の場合はズーム
                    e.preventDefault();
                    const touch1 = e.touches[0];
                    const touch2 = e.touches[1];
                    const dist = Math.hypot(
                        touch2.clientX - touch1.clientX,
                        touch2.clientY - touch1.clientY
                    );
                    
                    const newScale = scale * (dist / startDist);
                    if (newScale > 0.5 && newScale < 5) {  // スケール制限
                        scale = newScale;
                        
                        // 中心点を計算
                        const centerX = (touch1.clientX + touch2.clientX) / 2;
                        const centerY = (touch1.clientY + touch2.clientY) / 2;
                        const rect = canvas.getBoundingClientRect();
                        const canvasCenterX = (centerX - rect.left) / scale - translateX;
                        const canvasCenterY = (centerY - rect.top) / scale - translateY;
                        
                        // 変換を更新
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    startDist = dist;
                }
            });
            
            canvas.addEventListener('touchend', function(e) {
                if (e.touches.length === 0) {
                    isDrawing = false;
                }
            });
            
            // マウスでの描画（デスクトップ用）
            canvas.addEventListener('mousedown', function(e) {
                const rect = canvas.getBoundingClientRect();
                lastX = (e.clientX - rect.left) / scale - translateX;
                lastY = (e.clientY - rect.top) / scale - translateY;
                isDrawing = true;
            });
            
            canvas.addEventListener('mousemove', function(e) {
                if (!isDrawing) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / scale - translateX;
                const y = (e.clientY - rect.top) / scale - translateY;
                
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.strokeStyle = document.getElementById(`color-${nodeId}`).value;
                ctx.lineWidth = document.getElementById(`size-${nodeId}`).value;
                ctx.lineCap = 'round';
                ctx.stroke();
                
                lastX = x;
                lastY = y;
            });
            
            canvas.addEventListener('mouseup', function() {
                isDrawing = false;
            });
            
            canvas.addEventListener('mouseout', function() {
                isDrawing = false;
            });
        }
        
        // 寸法検知用キャンバスのセットアップ
        function setupDimensionCanvas(canvas, img, nodeId, resultText) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            const ctx = canvas.getContext('2d');
            canvas.points = [];
            
            // クリックイベント（タッチ含む）
            function handleClick(e) {
                e.preventDefault();
                
                const rect = canvas.getBoundingClientRect();
                let x, y;
                
                if (e.type === 'touchstart') {
                    x = e.touches[0].clientX - rect.left;
                    y = e.touches[0].clientY - rect.top;
                } else {
                    x = e.clientX - rect.left;
                    y = e.clientY - rect.top;
                }
                
                // 点の描画
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = 'red';
                ctx.fill();
                
                // テキスト
                ctx.fillStyle = 'white';
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;
                ctx.font = '12px Arial';
                ctx.strokeText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                ctx.fillText(`点${canvas.points.length + 1}`, x + 10, y - 10);
                
                // 点を保存
                canvas.points.push({x, y});
                
                // 2点目が追加されたら線を引く
                if (canvas.points.length === 2) {
                    const p1 = canvas.points[0];
                    const p2 = canvas.points[1];
                    
                    // 線を引く
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = 'yellow';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // 距離の計算
                    const distance = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                    
                    // 中間点に距離を表示
                    const midX = (p1.x + p2.x) / 2;
                    const midY = (p1.y + p2.y) / 2;
                    
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = 'black';
                    ctx.font = '14px Arial';
                    
                    const dimensionText = `${distance.toFixed(1)}px`;
                    ctx.strokeText(dimensionText, midX, midY - 10);
                    ctx.fillText(dimensionText, midX, midY - 10);
                    
                    // 結果テキストの更新
                    resultText.textContent = `測定結果: 2点間の距離は ${distance.toFixed(1)}px です`;
                    
                    // リセットする（次の測定のため）
                    setTimeout(() => {
                        canvas.points = [];
                    }, 500);
                }
            }
            
            canvas.addEventListener('mousedown', handleClick);
            canvas.addEventListener('touchstart', handleClick);
        }
        
        // 異常検知用キャンバスのセットアップ
        function setupAnomalyCanvas(canvas, img) {
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }
        
        // 異常検知実行関数（サンプル実装）
        function detectAnomalies(canvas, img, resultText) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // サンプルとして擬似的なヒートマップを生成
            const gradients = [
                {pos: [0.2, 0.3], radius: 30, intensity: 0.8},
                {pos: [0.5, 0.6], radius: 40, intensity: 0.6},
                {pos: [0.8, 0.2], radius: 25, intensity: 0.9}
            ];
            
            // ヒートマップの描画
            for (const grad of gradients) {
                const centerX = canvas.width * grad.pos[0];
                const centerY = canvas.height * grad.pos[1];
                
                const grd = ctx.createRadialGradient(
                    centerX, centerY, 1,
                    centerX, centerY, grad.radius
                );
                
                grd.addColorStop(0, `rgba(255, 0, 0, ${grad.intensity})`);
                grd.addColorStop(1, 'rgba(255, 0, 0, 0)');
                
                ctx.fillStyle = grd;
                ctx.beginPath();
                ctx.arc(centerX, centerY, grad.radius, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // 結果テキストの更新
            resultText.textContent = `検知結果: ${gradients.length}箇所の異常が検出されました`;
        }
        
        // 画像とキャンバスを合成する関数
        function combineImageAndCanvas(img, canvas) {
            const combinedCanvas = document.createElement('canvas');
            combinedCanvas.width = canvas.width;
            combinedCanvas.height = canvas.height;
            
            const ctx = combinedCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            ctx.drawImage(canvas, 0, 0);
            
            return combinedCanvas.toDataURL('image/png');
        }
        
        // カメラグリッドを描画する関数
        function renderCameraGrid() {
            const cameraCount = Object.keys(cameras).length;
            
            if (cameraCount === 0) {
                cameraGrid.innerHTML = `
                    <div class="placeholder">
                        <p>カメラが見つかりません。</p>
                        <p>カメラノードを起動して、このサーバーに接続してください。</p>
                    </div>
                `;
                return;
            }
            
            let gridHTML = '';
            
            for (const [nodeId, camera] of Object.entries(cameras)) {
                gridHTML += `
                    <div class="camera-card" data-id="${nodeId}">
                        <div class="camera-header">
                            <h3 class="camera-title">
                                <span class="status-indicator status-${camera.status}"></span>
                                ${camera.name}
                            </h3>
                            <div class="camera-actions">
                                <button class="refresh-stream-btn" data-id="${nodeId}">リフレッシュ</button>
                                <button class="snapshot-btn" data-id="${nodeId}">スナップショット</button>
                            </div>
                        </div>
                        <div class="camera-stream" id="stream-${nodeId}" data-zoom="1" data-translate-x="0" data-translate-y="0">
                            <div class="loading">読み込み中...</div>
                            ${camera.status === 'running' 
                                ? `<img src="${camera.url}" alt="${camera.name}" onerror="handleStreamError('${nodeId}')">`
                                : `<div class="error-overlay">カメラ接続エラー</div>`
                            }
                            <div class="zoom-controls">
                                <button class="zoom-in-btn" data-id="${nodeId}">+</button>
                                <button class="zoom-out-btn" data-id="${nodeId}">-</button>
                                <button class="zoom-reset-btn" data-id="${nodeId}">↺</button>
                            </div>
                        </div>
                        <div class="camera-info">
                            <p><strong>ID</strong> ${nodeId}</p>
                            <p><strong>解像度</strong> ${camera.resolution ? camera.resolution.join(' x ') : '不明'}</p>
                            <p><strong>ステータス</strong> ${getStatusText(camera.status)}</p>
                            <p><strong>最終確認</strong> ${camera.last_seen}</p>
                        </div>
                    </div>
                `;
            }
            
            cameraGrid.innerHTML = gridHTML;
            
            // イベントリスナーを追加
            document.querySelectorAll('.refresh-stream-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    refreshStream(nodeId);
                });
            });
            
            document.querySelectorAll('.snapshot-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    takeSnapshot(nodeId);
                });
            });
            
            // ズームコントロールのイベントリスナーを設定
            setupZoomControls();
        }
        
        // ストリームをリフレッシュする関数
        function refreshStream(nodeId) {
            if (!cameras[nodeId]) return;
            
            const streamContainer = document.getElementById(`stream-${nodeId}`);
            const camera = cameras[nodeId];
            
            if (camera.status === 'running') {
                streamContainer.innerHTML = `
                    <div class="loading">読み込み中...</div>
                    <img src="${camera.url}?t=${new Date().getTime()}" alt="${camera.name}" 
                         onerror="handleStreamError('${nodeId}')">
                `;
            }
        }
        
        // スナップショットを取得する関数
        async function takeSnapshot(nodeId) {
            if (!cameras[nodeId]) return;
            
            try {
                const response = await fetch(`/api/snapshot/${nodeId}`);
                if (!response.ok) {
                    throw new Error('スナップショット取得エラー');
                }
                
                const data = await response.json();
                
                if (data.success && data.image) {
                    // スナップショットをモーダルに表示
                    snapshotTitle.textContent = `スナップショット: ${cameras[nodeId].name}`;
                    snapshotImg.src = `data:image/jpeg;base64,${data.image}`;
                    
                    // ズーム機能を追加
                    let scale = 1;
                    let translateX = 0;
                    let translateY = 0;
                    let isDragging = false;
                    let startX, startY;
                    
                    // リセット
                    snapshotImg.style.transform = 'scale(1)';
                    
                    // ダブルクリックでズームイン/アウト
                    snapshotImg.addEventListener('dblclick', (e) => {
                        if (scale === 1) {
                            scale = 2;
                            // クリック位置を中心にズーム
                            const rect = snapshotImg.getBoundingClientRect();
                            const offsetX = (e.clientX - rect.left) / rect.width;
                            const offsetY = (e.clientY - rect.top) / rect.height;
                            translateX = (0.5 - offsetX) * 100;
                            translateY = (0.5 - offsetY) * 100;
                        } else {
                            scale = 1;
                            translateX = 0;
                            translateY = 0;
                        }
                        snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    });
                    
                    // ドラッグ
                    snapshotImg.addEventListener('mousedown', (e) => {
                        if (scale === 1) return;
                        isDragging = true;
                        startX = e.clientX;
                        startY = e.clientY;
                        e.preventDefault();
                    });
                    
                    window.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        const x = e.clientX;
                        const y = e.clientY;
                        translateX += (x - startX) / scale;
                        translateY += (y - startY) / scale;
                        snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        startX = x;
                        startY = y;
                    });
                    
                    window.addEventListener('mouseup', () => {
                        isDragging = false;
                    });
                    
                    // タッチ操作
                    let initialDistance = 0;
                    let initialScale = 1;
                    
                    snapshotImg.addEventListener('touchstart', (e) => {
                        if (e.touches.length === 2) {
                            initialDistance = Math.hypot(
                                e.touches[1].clientX - e.touches[0].clientX,
                                e.touches[1].clientY - e.touches[0].clientY
                            );
                            initialScale = scale;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            isDragging = true;
                            startX = e.touches[0].clientX;
                            startY = e.touches[0].clientY;
                            e.preventDefault();
                        }
                    });
                    
                    snapshotImg.addEventListener('touchmove', (e) => {
                        if (e.touches.length === 2) {
                            const distance = Math.hypot(
                                e.touches[1].clientX - e.touches[0].clientX,
                                e.touches[1].clientY - e.touches[0].clientY
                            );
                            scale = Math.max(1, Math.min(3, initialScale * (distance / initialDistance)));
                            snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && isDragging) {
                            const x = e.touches[0].clientX;
                            const y = e.touches[0].clientY;
                            translateX += (x - startX) / scale;
                            translateY += (y - startY) / scale;
                            snapshotImg.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                            startX = x;
                            startY = y;
                            e.preventDefault();
                        }
                    });
                    
                    snapshotImg.addEventListener('touchend', () => {
                        isDragging = false;
                    });
                    
                    // モーダルを表示
                    showModal(snapshotModal);
                } else {
                    alert('スナップショット取得エラー: ' + (data.error || '不明なエラー'));
                }
            } catch (error) {
                console.error('スナップショットエラー:', error);
                alert('スナップショット取得エラー: ' + error.message);
            }
        }
        
        // ストリームエラーを処理する関数
        function handleStreamError(nodeId) {
            if (!cameras[nodeId]) return;
            
            const streamContainer = document.getElementById(`stream-${nodeId}`);
            
            if (streamContainer) {
                streamContainer.innerHTML = `
                    <div class="error-overlay">
                        ストリーム読み込みエラー
                        <button class="button" onclick="refreshStream('${nodeId}')">
                            再試行
                        </button>
                    </div>
                `;
            }
        }
        
        // ステータスのテキストを取得する関数
        function getStatusText(status) {
            switch (status) {
                case 'running': return '正常';
                case 'error': return 'エラー';
                case 'initializing': return '初期化中';
                case 'unreachable': return '接続不可';
                default: return status;
            }
        }
        
        // グリッドの列数を切り替える関数
        function toggleGridColumns() {
            if (gridColumns === 'auto-fill') {
                gridColumns = '1';
                cameraGrid.style.gridTemplateColumns = '1fr';
                document.querySelectorAll('.camera-grid-function').forEach(grid => {
                    grid.style.gridTemplateColumns = '1fr';
                });
            } else {
                gridColumns = 'auto-fill';
                cameraGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(450px, 1fr))';
                document.querySelectorAll('.camera-grid-function').forEach(grid => {
                    grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(500px, 1fr))';
                });
            }
        }
        
        // イベントリスナー
        refreshBtn.addEventListener('click', fetchCameras);
        gridToggleBtn.addEventListener('click', toggleGridColumns);
        
        // モーダルを閉じる
        closeModalBtn.addEventListener('click', () => {
            snapshotModal.style.display = 'none';
        });
        
        // モーダル外をクリックして閉じる
        snapshotModal.addEventListener('click', (e) => {
            if (e.target === snapshotModal) {
                snapshotModal.style.display = 'none';
            }
        });
        
        // ズームコントロールの設定
        function setupZoomControls() {
            // ストリーミングタブのズームコントロール
            document.querySelectorAll('.zoom-in-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.min(scale + 0.2, 3);  // 最大ズーム3倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                });
            });
            
            document.querySelectorAll('.zoom-out-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    let scale = parseFloat(streamContainer.dataset.zoom || 1);
                    scale = Math.max(scale - 0.2, 1);  // 最小ズーム等倍
                    streamContainer.dataset.zoom = scale;
                    img.style.transform = `scale(${scale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                    
                    // ズームアウト時に位置をリセットする場合
                    if (scale === 1) {
                        streamContainer.dataset.translateX = 0;
                        streamContainer.dataset.translateY = 0;
                    }
                });
            });
            
            document.querySelectorAll('.zoom-reset-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const nodeId = e.target.dataset.id;
                    const streamContainer = document.getElementById(`stream-${nodeId}`);
                    const img = streamContainer.querySelector('img');
                    if (!img) return;
                    
                    streamContainer.dataset.zoom = 1;
                    streamContainer.dataset.translateX = 0;
                    streamContainer.dataset.translateY = 0;
                    img.style.transform = 'scale(1) translate(0px, 0px)';
                });
            });
            
            // ストリーミングタブでの画像ドラッグ機能
            document.querySelectorAll('.camera-stream img').forEach(img => {
                let isDragging = false;
                let startX, startY;
                let translateX = 0;
                let translateY = 0;
                
                // ドラッグ開始
                img.addEventListener('mousedown', (e) => {
                    const streamContainer = e.target.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                    
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    translateX = parseFloat(streamContainer.dataset.translateX || 0);
                    translateY = parseFloat(streamContainer.dataset.translateY || 0);
                    
                    streamContainer.classList.add('zoomed');
                });
                
                // ドラッグ中
                window.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;
                    
                    const x = e.clientX;
                    const y = e.clientY;
                    const deltaX = (x - startX) / 5;
                    const deltaY = (y - startY) / 5;
                    
                    const newTranslateX = translateX + deltaX;
                    const newTranslateY = translateY + deltaY;
                    
                    const streamContainer = img.closest('.camera-stream');
                    const scale = parseFloat(streamContainer.dataset.zoom || 1);
                    
                    streamContainer.dataset.translateX = newTranslateX;
                    streamContainer.dataset.translateY = newTranslateY;
                    
                    img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                });
                
                // ドラッグ終了
                window.addEventListener('mouseup', () => {
                    isDragging = false;
                    document.querySelectorAll('.camera-stream.zoomed').forEach(el => {
                        el.classList.remove('zoomed');
                    });
                });
            });
            
            // タッチデバイス用の処理
            document.querySelectorAll('.camera-stream img').forEach(img => {
                const streamContainer = img.closest('.camera-stream');
                
                // ピンチズーム用の変数
                let initialDistance = 0;
                let initialScale = 1;
                
                // タッチ開始
                streamContainer.addEventListener('touchstart', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        initialDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        initialScale = parseFloat(streamContainer.dataset.zoom || 1);
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;  // ズームインしていない場合はドラッグ無効
                        
                        const touch = e.touches[0];
                        streamContainer.dataset.startX = touch.clientX;
                        streamContainer.dataset.startY = touch.clientY;
                        streamContainer.dataset.translateStartX = parseFloat(streamContainer.dataset.translateX || 0);
                        streamContainer.dataset.translateStartY = parseFloat(streamContainer.dataset.translateY || 0);
                        streamContainer.classList.add('zoomed');
                        e.preventDefault();
                    }
                });
                
                // タッチ移動
                streamContainer.addEventListener('touchmove', (e) => {
                    if (e.touches.length === 2) {
                        // ピンチズームの場合
                        const touch1 = e.touches[0];
                        const touch2 = e.touches[1];
                        const currentDistance = Math.hypot(
                            touch2.clientX - touch1.clientX,
                            touch2.clientY - touch1.clientY
                        );
                        
                        // スケールの計算
                        let newScale = initialScale * (currentDistance / initialDistance);
                        newScale = Math.max(1, Math.min(3, newScale));  // 1～3倍に制限
                        
                        streamContainer.dataset.zoom = newScale;
                        img.style.transform = `scale(${newScale}) translate(${streamContainer.dataset.translateX || 0}px, ${streamContainer.dataset.translateY || 0}px)`;
                        
                        e.preventDefault();
                    } else if (e.touches.length === 1) {
                        // ドラッグの場合
                        const scale = parseFloat(streamContainer.dataset.zoom || 1);
                        if (scale <= 1) return;
                        
                        const touch = e.touches[0];
                        const startX = parseFloat(streamContainer.dataset.startX || 0);
                        const startY = parseFloat(streamContainer.dataset.startY || 0);
                        const translateStartX = parseFloat(streamContainer.dataset.translateStartX || 0);
                        const translateStartY = parseFloat(streamContainer.dataset.translateStartY || 0);
                        
                        const deltaX = (touch.clientX - startX) / 5;
                        const deltaY = (touch.clientY - startY) / 5;
                        
                        const newTranslateX = translateStartX + deltaX;
                        const newTranslateY = translateStartY + deltaY;
                        
                        streamContainer.dataset.translateX = newTranslateX;
                        streamContainer.dataset.translateY = newTranslateY;
                        
                        img.style.transform = `scale(${scale}) translate(${newTranslateX}px, ${newTranslateY}px)`;
                        
                        e.preventDefault();
                    }
                });
                
                // タッチ終了
                streamContainer.addEventListener('touchend', () => {
                    streamContainer.classList.remove('zoomed');
                });
            });
            
            // アノテーション、寸法、異常検知のズームコントロール
            ['annotation', 'dimension', 'anomaly'].forEach(tabType => {
                for (const [nodeId, camera] of Object.entries(cameras)) {
                    const zoomInBtn = document.getElementById(`zoom-in-${tabType}-${nodeId}`);
                    const zoomOutBtn = document.getElementById(`zoom-out-${tabType}-${nodeId}`);
                    const zoomResetBtn = document.getElementById(`zoom-reset-${tabType}-${nodeId}`);
                    const img = document.getElementById(`${tabType}-img-${nodeId}`);
                    const canvas = document.getElementById(`${tabType}-canvas-${nodeId}`);
                    
                    if (!zoomInBtn || !zoomOutBtn || !zoomResetBtn || !img || !canvas) continue;
                    
                    let scale = 1;
                    let translateX = 0;
                    let translateY = 0;
                    
                    // ズームイン
                    zoomInBtn.addEventListener('click', () => {
                        scale = Math.min(scale + 0.2, 3);
                        updateTransform();
                    });
                    
                    // ズームアウト
                    zoomOutBtn.addEventListener('click', () => {
                        scale = Math.max(scale - 0.2, 1);
                        if (scale === 1) {
                            translateX = 0;
                            translateY = 0;
                        }
                        updateTransform();
                    });
                    
                    // リセット
                    zoomResetBtn.addEventListener('click', () => {
                        scale = 1;
                        translateX = 0;
                        translateY = 0;
                        updateTransform();
                    });
                    
                    // 変換を更新する関数
                    function updateTransform() {
                        img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                        canvas.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
                    }
                    
                    // ドラッグ処理
                    let isDragging = false;
                    let startX, startY;
                    
                    const container = document.getElementById(`${tabType}-container-${nodeId}`);
                    
                    container.addEventListener('mousedown', (e) => {
                        if (scale <= 1) return;
                        isDragging = true;
                        startX = e.clientX;
                        startY = e.clientY;
                    });
                    
                    container.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        
                        const x = e.clientX;
                        const y = e.clientY;
                        const deltaX = (x - startX) / 5;
                        const deltaY = (y - startY) / 5;
                        
                        translateX += deltaX;
                        translateY += deltaY;
                        
                        updateTransform();
                        
                        startX = x;
                        startY = y;
                    });
                    
                    container.addEventListener('mouseup', () => {
                        isDragging = false;
                    });
                    
                    container.addEventListener('mouseleave', () => {
                        isDragging = false;
                    });
                    
                    // タッチデバイス用
                    container.addEventListener('touchstart', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            container.dataset.initialDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            container.dataset.initialScale = scale;
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            container.dataset.startX = touch.clientX;
                            container.dataset.startY = touch.clientY;
                            e.preventDefault();
                        }
                    });
                    
                    container.addEventListener('touchmove', (e) => {
                        if (e.touches.length === 2) {
                            // ピンチズーム
                            const touch1 = e.touches[0];
                            const touch2 = e.touches[1];
                            const currentDistance = Math.hypot(
                                touch2.clientX - touch1.clientX,
                                touch2.clientY - touch1.clientY
                            );
                            
                            const initialDistance = parseFloat(container.dataset.initialDistance || 0);
                            const initialScale = parseFloat(container.dataset.initialScale || 1);
                            
                            if (initialDistance > 0) {
                                scale = Math.max(1, Math.min(3, initialScale * (currentDistance / initialDistance)));
                                updateTransform();
                            }
                            
                            e.preventDefault();
                        } else if (e.touches.length === 1 && scale > 1) {
                            // 移動
                            const touch = e.touches[0];
                            const startX = parseFloat(container.dataset.startX || 0);
                            const startY = parseFloat(container.dataset.startY || 0);
                            
                            if (startX && startY) {
                                const deltaX = (touch.clientX - startX) / 5;
                                const deltaY = (touch.clientY - startY) / 5;
                                
                                translateX += deltaX;
                                translateY += deltaY;
                                
                                updateTransform();
                                
                                container.dataset.startX = touch.clientX;
                                container.dataset.startY = touch.clientY;
                            }
                            
                            e.preventDefault();
                        }
                    });
                }
            });
        }
        
        // スナップショットモーダルのアニメーション
        function showModal(modal) {
            modal.style.display = 'flex';
            setTimeout(() => {
                modal.classList.add('visible');
            }, 10);
        }
        
        function hideModal(modal) {
            modal.classList.remove('visible');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        }
        
        // ページロード時にカメラ情報を取得
        document.addEventListener('DOMContentLoaded', () => {
            fetchCameras();
            
            // グローバル関数の定義（handleStreamErrorはグローバルに必要）
            window.handleStreamError = handleStreamError;
            window.refreshStream = refreshStream;
            
            // 1分ごとに自動更新（ストリーミングタブがアクティブの場合のみ）
            setInterval(() => {
                if (currentTab === 'streaming') {
                    fetchCameras();
                }
            }, 60000);
            
            // モーダルのアニメーション対応
            closeModalBtn.addEventListener('click', () => {
                hideModal(snapshotModal);
            });
            
            snapshotModal.addEventListener('click', (e) => {
                if (e.target === snapshotModal) {
                    hideModal(snapshotModal);
                }
            });
        });
    </script>
</body>
</html>
    """
    
    # テンプレートを保存
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_template)
    
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