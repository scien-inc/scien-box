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
        global frame
        if frame is None:
            return jsonify({'error': 'No frame available'}), 404
        
        try:
            # 現在のフレームをJPEGとしてエンコード
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
    return render_template('dashboard.html')

if __name__ == '__main__':
    # templatesディレクトリが存在しない場合は作成
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # dashboardテンプレートの作成
    dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>カメラネットワークダッシュボード</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .camera-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .camera-header {
            background: #4a90e2;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .camera-status {
            font-size: 0.8em;
            padding: 3px 8px;
            border-radius: 12px;
            background: #e2e2e2;
        }
        .status-running {
            background: #4caf50;
            color: white;
        }
        .status-error {
            background: #f44336;
            color: white;
        }
        .camera-stream {
            width: 100%;
            height: 300px;
            background: #eee;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .camera-stream img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .camera-info {
            padding: 10px 15px;
            font-size: 0.9em;
            color: #666;
        }
        .refresh-button {
            background: #4a90e2;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .refresh-button:hover {
            background: #3a80d2;
        }
        .no-cameras {
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 1.2em;
        }
        @media (max-width: 800px) {
            .camera-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <h1>カメラネットワークダッシュボード</h1>
    
    <div style="text-align: center;">
        <button id="refreshButton" class="refresh-button">カメラ一覧を更新</button>
    </div>
    
    <div id="cameraGrid" class="camera-grid">
        <div class="no-cameras">カメラを検索中...</div>
    </div>

    <script>
        function fetchCameras() {
            fetch('/api/cameras')
                .then(response => response.json())
                .then(data => {
                    const cameraGrid = document.getElementById('cameraGrid');
                    cameraGrid.innerHTML = '';
                    
                    if (Object.keys(data).length === 0) {
                        cameraGrid.innerHTML = '<div class="no-cameras">利用可能なカメラがありません</div>';
                        return;
                    }
                    
                    for (const [id, camera] of Object.entries(data)) {
                        const card = document.createElement('div');
                        card.className = 'camera-card';
                        
                        const statusClass = camera.status === 'running' ? 'status-running' : 'status-error';
                        
                        card.innerHTML = `
                            <div class="camera-header">
                                <span>${camera.name}</span>
                                <span class="camera-status ${statusClass}">${camera.status === 'running' ? '稼働中' : 'エラー'}</span>
                            </div>
                            <div class="camera-stream">
                                <img src="${camera.url}" alt="${camera.name}" onerror="this.src='/static/offline.jpg'; this.onerror=null;">
                            </div>
                            <div class="camera-info">
                                <p>IP: ${camera.ip}:${camera.port}</p>
                                <p>解像度: ${camera.resolution ? camera.resolution.join('x') : '不明'}</p>
                                <p>最終通信: ${camera.last_seen}</p>
                            </div>
                        `;
                        
                        cameraGrid.appendChild(card);
                    }
                })
                .catch(error => {
                    console.error('Error fetching cameras:', error);
                    document.getElementById('cameraGrid').innerHTML = 
                        '<div class="no-cameras">カメラ情報の取得中にエラーが発生しました</div>';
                });
        }

        // 初期ロード
        document.addEventListener('DOMContentLoaded', fetchCameras);
        
        // 更新ボタン
        document.getElementById('refreshButton').addEventListener('click', fetchCameras);
        
        // 定期的な更新（30秒ごと）
        setInterval(fetchCameras, 30000);
    </script>
</body>
</html>
    """
    
    # テンプレートを保存
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
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