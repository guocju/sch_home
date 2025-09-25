import threading
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json
import logging
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device_port = {
    "CPU": 1900,
    "GPU": 2000,
    "NPU": 3000,
    "FPGA": 4000
}


class _TaskManager:
    """内部管理：维护 ui sockets 和 per-task sockets"""
    
    def __init__(self):
        self.ui_clients: Set[WebSocket] = set()
        self.task_clients: Dict[str, Set[WebSocket]] = {}
        self.tasks: Set[str] = set()
        self._lock = asyncio.Lock()  # 添加锁以防止竞态条件

    # UI (manager page) connect
    async def connect_ui(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.ui_clients.add(ws)
        # send current task list
        await self._safe_send(ws, {"event": "task_list", "tasks": list(self.tasks)})

    async def disconnect_ui(self, ws: WebSocket):
        async with self._lock:
            self.ui_clients.discard(ws)

    async def broadcast_ui(self, msg: dict):
        async with self._lock:
            clients = list(self.ui_clients)
        
        disconnected = []
        for ws in clients:
            try:
                await ws.send_json(msg)
            except Exception as e:
                logger.warning(f"Failed to send to UI client: {e}")
                disconnected.append(ws)
        
        # 清理断开的连接
        for ws in disconnected:
            await self.disconnect_ui(ws)

    async def _safe_send(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
            await self.disconnect_ui(ws)

    # task ws connect
    async def connect_task_ws(self, task: str, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            if task not in self.tasks:
                # 任务不存在，拒绝连接
                try:
                    await ws.send_json({"error": "Task not found", "task": task})
                    await ws.close()
                except:
                    pass
                return False
            self.task_clients.setdefault(task, set()).add(ws)
        return True

    async def disconnect_task_ws(self, task: str, ws: WebSocket):
        async with self._lock:
            cl = self.task_clients.get(task)
            if cl:
                cl.discard(ws)
                if not cl:
                    self.task_clients.pop(task, None)

    async def send_to_task_clients(self, task: str, msg: dict):
        async with self._lock:
            clients = list(self.task_clients.get(task, set()))
        
        disconnected = []
        for ws in clients:
            try:
                await ws.send_json(msg)
            except Exception as e:
                logger.warning(f"Failed to send to task client: {e}")
                disconnected.append(ws)
        
        # 清理断开的连接
        for ws in disconnected:
            await self.disconnect_task_ws(task, ws)

    # task lifecycle controlled by external manager
    async def add_task(self, task: str):
        async with self._lock:
            if task in self.tasks:
                return False
            self.tasks.add(task)
        
        await self.broadcast_ui({
            "event": "task_online", 
            "task": task, 
            "ws_path": f"/ws/task/{task}"
        })
        return True

    async def remove_task(self, task: str):
        async with self._lock:
            if task not in self.tasks:
                return False
            self.tasks.discard(task)
            # 获取该任务的所有连接
            clients = list(self.task_clients.get(task, set()))
            self.task_clients.pop(task, None)
        
        # notify UI
        await self.broadcast_ui({"event": "task_offline", "task": task})
        
        # notify task connections and close them
        for ws in clients:
            try:
                await ws.send_json({"status": "offline", "task": task})
                await ws.close()
            except Exception as e:
                logger.warning(f"Error closing task connection: {e}")
        
        return True

    async def get_tasks(self):
        """获取当前所有任务列表"""
        async with self._lock:
            return list(self.tasks)


class TaskPlotServer:
    """
    针对 task 的实时曲线监控服务。
    使用示例：
      s = TaskPlotServer(host='0.0.0.0', port=5000)
      s.start(background=True)
      s.add_task('train_step')
      s.push_value('train_step', time.time(), 0.123)
      s.remove_task('train_step')
      s.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.manager = _TaskManager()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        
        # FastAPI app with lifespan to capture loop
        self.app = FastAPI(lifespan=self._lifespan)
        # register routes
        self.app.get("/")(self._index)
        self.app.websocket("/ws/manager")(self._ws_manager)
        self.app.websocket("/ws/task/{task}")(self._ws_task)

    # ---------------- lifespan: 捕获 event loop，优雅关闭时清理 ----------------
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # called in server thread's event loop
        self.loop = asyncio.get_running_loop()
        self._started.set()
        logger.info(f"[task_plot] Server starting on {self.host}:{self.port}")
        try:
            yield
        finally:
            logger.info("[task_plot] Server shutting down: cleaning resources.")
            # 清理所有连接
            await self._cleanup_all_connections()
            self.loop = None
            self._stopped.set()
            logger.info("[task_plot] Shutdown complete.")

    async def _cleanup_all_connections(self):
        """清理所有WebSocket连接"""
        # 关闭所有UI客户端
        for ws in list(self.manager.ui_clients):
            try:
                await ws.close()
            except:
                pass
        
        # 关闭所有任务客户端
        for task, clients in list(self.manager.task_clients.items()):
            for ws in list(clients):
                try:
                    await ws.close()
                except:
                    pass

    # ---------------- 前端页面 (示例) ----------------
    async def _index(self):
        # 嵌入的示例页面： manager websocket + per-task websockets
        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Task Monitor (MA smoothing only)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
  <style>
    body{font-family:Arial;padding:12px;background:#f5f5f5;}
    h3{color:#333;}
    #tasks span{margin-right:8px;padding:4px 8px;background:#4CAF50;color:white;border-radius:4px;display:inline-block;}
    #status{padding:8px;margin:10px 0;border-radius:4px;font-size:14px;}
    .connected{background:#e8f5e9;color:#2e7d32;}
    .disconnected{background:#ffebee;color:#c62828;}
    canvas{background:white;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}    
    #controls{margin:10px 0;padding:8px;background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
    #controls label{margin-right:8px;font-size:13px}
  </style>
</head>
<body>
  <h3>实时任务指标监控（移动平均平滑 MA）</h3>
  <div id="status" class="disconnected">连接状态：未连接</div>
  <div id="tasks">当前任务：无</div>

  <div id="controls">
    <label><input id="smoothing_enabled" type="checkbox" checked> 启用平滑 (MA, 窗口=3s)</label>
    <small style="margin-left:12px;color:#666">平滑在客户端实时计算，默认保留最后 20s 数据。</small>
  </div>

  <canvas id="chart" width="1000" height="420"></canvas>

<script>
const ctx = document.getElementById('chart').getContext('2d');
const datasetIndex = {};
const smoothingParams = { enabled: true, ma_window: 3.0 };
const sockets = {};
let managerSocket = null;
let reconnectTimer = null;

function randColor(){
  const colors = [
    'rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 206, 86)',
    'rgb(75, 192, 192)', 'rgb(153, 102, 255)', 'rgb(255, 159, 64)'
  ];
  const used = Object.values(datasetIndex).map(i => chart.data.datasets[i]?.borderColor);
  const available = colors.filter(c => !used.includes(c));
  return available.length > 0 ? available[0] : 
    `rgb(${80+Math.floor(Math.random()*130)},${80+Math.floor(Math.random()*130)},${80+Math.floor(Math.random()*130)})`;
}

const chart = new Chart(ctx, {
  type: 'line',
  data: {datasets: []},
  options: {
    animation: false,
    parsing: false,
    normalized: true,
    spanGaps: true,
    interaction: {mode: 'nearest', axis: 'x', intersect: false},
    scales: {
      x: {type: 'linear', title: { display: true, text: '时间/s' }, ticks: { stepSize: 1 }},
      y: {title: { display: true, text: '值' }, beginAtZero: true, grace: '10%'}
    },
    plugins: {legend: {display: true}, tooltip: {callbacks: {label: function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(2); }}}}
  }
});

function updateStatus(connected) {
  const status = document.getElementById('status');
  if (connected) { status.className = 'connected'; status.textContent = '连接状态：已连接'; } 
  else { status.className = 'disconnected'; status.textContent = '连接状态：未连接'; }
}

function updateTaskUI() {
  const names = Object.keys(datasetIndex).filter(n => !n.endsWith('__raw'));
  document.getElementById('tasks').innerHTML = names.length ? '当前任务：' + names.map(n => `<span>${n}</span>`).join('') : '当前任务：无';
}

function ensureDataset(task) {
  if (datasetIndex[task] !== undefined) return datasetIndex[task];
  const color = randColor();
  chart.data.datasets.push({
    label: task,
    data: [],
    borderColor: color,
    backgroundColor: color + '20',
    pointRadius: 0,
    borderWidth: 2,
    fill: false,
    tension: 0.2,
    raw: []
  });
  datasetIndex[task] = chart.data.datasets.length - 1;
  updateTaskUI();
  chart.update();
  return datasetIndex[task];
}

function removeDataset(task) {
  const idx = datasetIndex[task];
  if (idx === undefined) return;
  chart.data.datasets.splice(idx, 1);
  delete datasetIndex[task];
  chart.data.datasets.forEach((d, i) => { datasetIndex[d.label] = i; });
  updateTaskUI();
  chart.update();
}

// MA smoothing helper
function computeMA(rawArr, windowSec) {
  const out = [];
  for (let i = 0; i < rawArr.length; ++i) {
    const t = rawArr[i].x;
    const lo = t - windowSec;
    let sum = 0, cnt = 0;
    for (let j = 0; j <= i; ++j) {
      if (rawArr[j].x >= lo) { sum += rawArr[j].y; cnt += 1; }
    }
    out.push({x: t, y: (cnt ? sum / cnt : rawArr[i].y)});
  }
  return out;
}

function recomputeDatasetForDisplay(dsIndex) {
  const ds = chart.data.datasets[dsIndex];
  if (!ds) return;
  const raw = ds.raw.slice();
  raw.sort((a,b)=>a.x-b.x);
  let display = raw;
  if (smoothingParams.enabled && raw.length > 0) {
    display = computeMA(raw, smoothingParams.ma_window);
  }
  if (display.length > 0) {
    const now = display[display.length-1].x;
    const horizon = 20;
    const filtered = display.filter(p => now - p.x <= horizon);
    ds.data = filtered.map(p => ({x: p.x, y: p.y}));
  } else {
    ds.data = [];
  }
}

function applySmoothingToAll() {
  Object.keys(datasetIndex).forEach(task => {
    const i = datasetIndex[task];
    recomputeDatasetForDisplay(i);
  });
  chart.update('none');
}

function connectTask(task) {
  if (sockets[task]) return;
  const ws = new WebSocket(`ws://${location.host}/ws/task/${encodeURIComponent(task)}`);
  sockets[task] = ws;
  ws.onopen = () => { ensureDataset(task); console.log(`Task '${task}' connected`); };
  ws.onmessage = ev => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.error) { console.error(`Task '${task}' error:`, msg.error); ws.close(); return; }
      if (msg.status === 'offline') { ws.close(); delete sockets[task]; removeDataset(task); return; }
      if (typeof msg.ts === 'number' && typeof msg.value === 'number') {
        const idx = ensureDataset(task);
        const now = msg.ts;
        const ds = chart.data.datasets[idx];
        ds.raw.push({x: now, y: msg.value});
        const horizon = 20;
        ds.raw = ds.raw.filter(p => now - p.x <= horizon + Math.max(1, smoothingParams.ma_window || 0));
        recomputeDatasetForDisplay(idx);
        chart.update('none');
      }
    } catch (e) { console.error(`Error processing message for task '${task}':`, e); }
  };
  ws.onclose = () => { console.log(`Task '${task}' disconnected`); delete sockets[task]; removeDataset(task); };
  ws.onerror = e => { console.error(`Task '${task}' WebSocket error:`, e); };
}

function disconnectTask(task) { const ws = sockets[task]; if (ws) { ws.close(); } delete sockets[task]; removeDataset(task); }

function connectManager() {
  if (managerSocket && managerSocket.readyState === WebSocket.OPEN) return;
  managerSocket = new WebSocket(`ws://${location.host}/ws/manager`);
  managerSocket.onopen = () => { updateStatus(true); console.log('Manager connected'); clearTimeout(reconnectTimer); };
  managerSocket.onmessage = ev => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.event === 'task_list') { (msg.tasks || []).forEach(task => connectTask(task)); }
      else if (msg.event === 'task_online') { connectTask(msg.task); }
      else if (msg.event === 'task_offline') { disconnectTask(msg.task); }
    } catch (e) { console.error('Error processing manager message:', e); }
  };
  managerSocket.onclose = () => {
    updateStatus(false);
    console.log('Manager disconnected, reconnecting in 3s...');
    Object.keys(sockets).forEach(task => disconnectTask(task));
    reconnectTimer = setTimeout(connectManager, 3000);
  };
  managerSocket.onerror = e => { console.error('Manager WebSocket error:', e); };
}

// controls wiring
document.getElementById('smoothing_enabled').addEventListener('change', (e)=>{ smoothingParams.enabled = e.target.checked; applySmoothingToAll(); });

// initial connect
connectManager();

// heartbeat
setInterval(()=>{ if (managerSocket && managerSocket.readyState === WebSocket.OPEN) managerSocket.send('ping'); }, 30000);
</script>
</body>
</html>

"""
        return HTMLResponse(html)

    # ---------------- WebSocket endpoints ----------------
    async def _ws_manager(self, ws: WebSocket):
        await self.manager.connect_ui(ws)
        try:
            while True:
                # 接收心跳或其他消息
                try:
                    data = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                    if data == 'ping':
                        # 可以回应pong，但这里简单忽略
                        pass
                except asyncio.TimeoutError:
                    # 发送心跳检查连接
                    try:
                        await ws.send_json({"event": "ping"})
                    except:
                        break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"Manager WebSocket error: {e}")
        finally:
            await self.manager.disconnect_ui(ws)

    async def _ws_task(self, ws: WebSocket, task: str):
        # 验证任务是否存在
        connected = await self.manager.connect_task_ws(task, ws)
        if not connected:
            return
        
        try:
            while True:
                # 保持连接打开，但设置超时
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                except asyncio.TimeoutError:
                    # 检查连接是否仍然有效
                    try:
                        await ws.send_json({"ping": True})
                    except:
                        break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"Task WebSocket error for '{task}': {e}")
        finally:
            await self.manager.disconnect_task_ws(task, ws)

    # ---------------- 对外接口（线程/进程安全） ----------------
    def start(self, background: bool = True):
        """Start uvicorn server. If background, run in a daemon thread."""
        if self._server_thread and self._server_thread.is_alive():
            logger.warning("Server is already running")
            return
        
        def _run():
            try:
                config = uvicorn.Config(
                    self.app, 
                    host=self.host, 
                    port=self.port, 
                    log_level="info",
                    access_log=False  # 减少日志噪音
                )
                server = uvicorn.Server(config)
                self._server = server
                server.run()
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                self._stopped.set()
        
        self._started.clear()
        self._stopped.clear()
        
        t = threading.Thread(target=_run, daemon=background)
        t.start()
        self._server_thread = t
        
        # 等待服务器启动
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("Server failed to start within 5 seconds")
        
        logger.info(f"Server started at http://{self.host}:{self.port}")

    def stop(self, timeout: float = 5.0):
        """Stop uvicorn server (if started via start())."""
        if not self._server_thread or not self._server_thread.is_alive():
            logger.warning("Server is not running")
            return
        
        logger.info("Stopping server...")
        
        if self._server:
            # 请求关闭
            self._server.should_exit = True
        
        # 等待线程结束
        self._server_thread.join(timeout=timeout)
        
        if self._server_thread.is_alive():
            logger.warning("Server thread did not stop within timeout")
        else:
            logger.info("Server stopped successfully")

    def add_task(self, task: str) -> bool:
        """通知前端新增任务曲线（线程安全）。"""
        if self.loop is None:
            raise RuntimeError("Server not running")
        
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.manager.add_task(task), 
                self.loop
            )
            return fut.result(timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to add task '{task}': {e}")
            return False

    def remove_task(self, task: str) -> bool:
        """移除任务（线程安全）。"""
        if self.loop is None:
            raise RuntimeError("Server not running")
        
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.manager.remove_task(task), 
                self.loop
            )
            return fut.result(timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to remove task '{task}': {e}")
            return False

    def push_value(self, task: str, ts: float, value: float):
        """推送一条数据到该任务的所有前端订阅者（线程安全）。
        
        Args:
            task: 任务名称
            ts: 时间戳（秒）
            value: 数值（如帧率）
        """
        if self.loop is None:
            raise RuntimeError("Server not running")
        
        try:
            # 不等待结果以提高性能
            asyncio.run_coroutine_threadsafe(
                self.manager.send_to_task_clients(
                    task, 
                    {"task": task, "ts": ts, "value": value}
                ),
                self.loop
            )
        except Exception as e:
            logger.error(f"Failed to push value for task '{task}': {e}")

    def get_tasks(self) -> list:
        """获取当前所有任务列表（线程安全）。"""
        if self.loop is None:
            return []
        
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.manager.get_tasks(),
                self.loop
            )
            return fut.result(timeout=1.0)
        except Exception:
            return []


# 方便直接运行调试
if __name__ == "__main__":
    import random
    
    # 创建服务器
    server = TaskPlotServer(host="127.0.0.1", port=5000)
    server.start(background=True)
    
    print(f"Server started at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop...")
    
    # 模拟数据推送
    tasks = ["yolo", "BFS"]
    
    try:
        # 添加任务
        for task in tasks:
            server.add_task(task)
            print(f"Added task: {task}")
        
        # 持续推送数据
        while True:
            for task in tasks:
                # 模拟帧率数据（20-60 FPS之间波动）
                fps = 40 + 20 * math.sin(time.time()) + random.uniform(-5, 5)
                server.push_value(task, time.time(), fps)
            
            time.sleep(0.1)  # 10Hz更新频率
            
    except KeyboardInterrupt:
        print("\nStopping server...")
        # 移除所有任务
        for task in tasks:
            server.remove_task(task)
        server.stop()
        print("Server stopped.")