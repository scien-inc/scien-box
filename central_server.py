from flask import Flask, render_template, request, jsonify, Response
import json
import threading
import time
import logging
import os
import requests
from datetime import datetime

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flaskアプリの初期化
app = Flask(__name__)

# グローバル変数
cameras = {}  # カメラノード情報を格納する辞書
camera_lock = threading.Lock()  # スレッドセーフな操作のためのロック
HEARTBEAT_TIMEOUT = 60  # ハートビートタイムアウト（秒）

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
                # タイムアウトしたノードを特定
                timed_out_nodes = [
                    node_id for node_id, info in cameras.items()
                    if current_time - info.get('last_heartbeat', 0) > HEARTBEAT_TIMEOUT
                ]
                
                # タイムアウトしたノードを削除
                for node_id in timed_out_nodes:
                    logger.info(f"ノード {node_id} ({cameras[node_id].get('name', 'unknown')}) がタイムアウトしました")
                    cameras.pop(node_id, None)
                
                # 各ノードのステータスを更新
                for node_id, info in list(cameras.items()):
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
    if node_id not in cameras:
        return jsonify({'error': 'Camera not found'}), 404
    
    data, status = request_node(node_id, '/api/snapshot')
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': f'Failed to get snapshot: {status}'}), 500

# メインページ
@app.route('/')
def index():
    return render_template('dashboard.html')

if __name__ == '__main__':
    # templatesディレクトリが存在しない場合は作成
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # クリーンアップスレッドの開始
    cleanup_thread = threading.Thread(target=cleanup_thread)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # サーバーの開始
    PORT = int(os.environ.get('SERVER_PORT', 5001))  # デフォルトを5001に変更
    logger.info(f"中央サーバーを開始します: http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
