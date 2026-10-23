"""OpenAI 兼容 API 层。

用标准库 ``http.server.ThreadingHTTPServer`` 实现零依赖 API：

- ``GET  /v1/models``            列出模型
- ``POST /v1/chat/completions``  对话补全
- ``POST /v1/completions``       文本补全
- ``POST /api/embeddings``       嵌入
- ``POST /api/{alias}``          按模型系列别名转发（qwen / llama / ...）
- ``GET  /api/status``           服务状态
- ``GET  /healthz``              健康检查

支持 CORS、Bearer Token 校验、流式（SSE）转发与请求统计。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import proxy
from .backends import Backend, detect_backends
from .config import Config
from .proxy import DEFAULT_ALIAS, forward_json, pick_model_for_alias

__all__ = ["ApiServer", "create_handler", "build_server", "main"]

# ---------------------------------------------------------------- 内置网页面板

_WEBUI_HTML = r"""
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ImDeploy 本地模型部署器</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2129;--border:#30363d;
--text:#e6edf3;--dim:#8b949e;--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;--err:#f85149}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.6 "Microsoft YaHei UI",system-ui,sans-serif;padding:24px}
.wrap{max-width:1080px;margin:0 auto}
header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--border)}
h1{font-size:19px;font-weight:600}
.badge{font-size:12px;padding:3px 10px;border-radius:20px;background:rgba(63,185,80,.15);color:var(--ok);border:1px solid rgba(63,185,80,.3)}
.badge.off{background:rgba(139,148,158,.15);color:var(--dim);border-color:rgba(139,148,158,.3)}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:14px;animation:fadeUp .35s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
h2{font-size:14px;color:var(--accent);margin-bottom:12px;font-weight:600}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.kv{display:grid;grid-template-columns:110px 1fr;gap:6px 12px;font-size:13px}
.kv dt{color:var(--dim)}
.kv dd{color:var(--text);word-break:break-all}
button{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:7px;
padding:8px 16px;cursor:pointer;font-size:13px;font-family:inherit;transition:.18s}
button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
button.primary{background:var(--accent);color:#06101f;border-color:var(--accent);font-weight:600}
button.primary:hover{color:#06101f;filter:brightness(1.12)}
input{background:var(--panel2);border:1px solid var(--border);border-radius:7px;padding:8px 12px;
color:var(--text);font-family:inherit;font-size:13px;flex:1;min-width:180px}
input:focus{outline:none;border-color:var(--accent)}
select{background:var(--panel2);border:1px solid var(--border);border-radius:7px;padding:8px 12px;
color:var(--text);font-family:inherit;font-size:13px;width:100%}
select:focus{outline:none;border-color:var(--accent)}
.fl{display:block;font-size:12px;color:var(--dim);margin-bottom:4px}
.sw{display:inline-flex;align-items:center;cursor:pointer}
.sw input{display:none}
.sw .track{width:38px;height:21px;border-radius:11px;background:var(--panel2);
border:1px solid var(--border);position:relative;transition:.2s;display:block}
.sw .track::after{content:'';position:absolute;width:15px;height:15px;border-radius:50%;
background:var(--dim);top:2px;left:2px;transition:.2s}
.sw input:checked + .track{background:rgba(88,166,255,.25);border-color:var(--accent)}
.sw input:checked + .track::after{left:20px;background:var(--accent)}
h3{font-weight:600}
.qsel{max-width:170px}
.btnrow{display:inline-flex;gap:6px}
button.danger{background:#3a1d22;border-color:#5c2b32;color:#ff9aa5}
button.danger:hover{background:#4a2429}
.logbox{background:#0a0d12;border:1px solid var(--border);border-radius:8px;padding:12px;
max-height:220px;overflow-y:auto;font-family:Consolas,monospace;font-size:12px;
color:var(--dim);white-space:pre-wrap;margin:0}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 10px;color:var(--dim);font-weight:500;border-bottom:1px solid var(--border);font-size:12px}
td{padding:8px 10px;border-bottom:1px solid rgba(48,54,61,.5)}
tr:hover td{background:var(--panel2)}
.mono{font-family:Consolas,monospace;font-size:12px}
#log{background:#0a0d12;border:1px solid var(--border);border-radius:8px;padding:12px;
height:180px;overflow-y:auto;font-family:Consolas,monospace;font-size:12px;color:var(--dim);white-space:pre-wrap}
.tip{color:var(--dim);font-size:12px;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>ImDeploy 本地模型部署器</h1>
  <span class="badge" id="status">检测中</span>
</header>

<div class="card">
  <h2>服务状态</h2>
  <dl class="kv" id="statusKv"><dt>加载中…</dt><dd></dd></dl>
</div>

<div class="card">
  <h2>推荐模型（按本机硬件匹配）</h2>
  <div class="row">
    <button class="primary" onclick="loadRecs(true)">重新推荐</button>
    <span class="tip" id="recTip">点「重新推荐」从模型站抓取并匹配</span>
  </div>
  <table>
    <thead><tr><th>模型</th><th>档位</th><th>参数量</th><th>上下文</th><th>体积</th><th>操作</th></tr></thead>
    <tbody id="recBody"><tr><td colspan="6" style="color:var(--dim)">尚未加载推荐列表</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>下载进度</h2>
  <div id="dlBox"><span class="tip">暂无下载任务</span></div>
</div>

<div class="card">
  <h2>设置</h2>
  <div class="row">
    <button onclick="loadConfig()">刷新</button>
    <button class="primary" onclick="saveConfig()">保存全部</button>
    <span class="tip" id="cfgTip"></span>
  </div>
  <div id="cfgBox"><span class="tip">正在读取配置…</span></div>
</div>

<div class="card">
  <h2>推理运行库
    <span class="right">
      <button onclick="installRuntime()" id="btnRuntime">一键安装</button>
    </span>
  </h2>
  <div id="runtimeBox"><span class="tip">正在检测…</span></div>
</div>

<div class="card">
  <h2>模型库</h2>
  <div class="row">
    <button onclick="loadModels()">刷新列表</button>
    <span class="tip" id="modelTip"></span>
  </div>
  <table>
    <thead><tr><th>模型</th><th>系列</th><th>参数量</th><th>上下文</th><th>量化</th><th>操作</th></tr></thead>
    <tbody id="modelBody"><tr><td colspan="6" style="color:var(--dim)">暂无已下载模型</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>请求测试</h2>
  <div class="row">
    <input id="q" placeholder="输入一句话测试…" value="你好，介绍一下你自己">
    <button class="primary" onclick="ask()">发送</button>
  </div>
  <div id="log">就绪。先在设置里加载模型，然后在这里测试对话。</div>
</div>

<div class="card">
  <h2>接口地址</h2>
  <dl class="kv">
    <dt>OpenAI 兼容</dt><dd class="mono" id="u1"></dd>
    <dt>模型列表</dt><dd class="mono" id="u2"></dd>
    <dt>多模型别名</dt><dd class="mono" id="u3"></dd>
  </dl>
</div>
</div>

<script>
const base = location.origin;
let loadedIds = {};   // 已加载模型 id 集合，删除前判断要不要先卸载
const $ = id => document.getElementById(id);

function log(msg){
  const el = $('log');
  el.textContent = msg;
  el.scrollTop = el.scrollHeight;
}

async function loadConfig(){
  const box = document.getElementById('cfgBox');
  const tip = document.getElementById('cfgTip');
  try{
    const r = await fetch(base + '/api/config');
    const d = await r.json();
    if(!d.ok){ box.innerHTML = '<span class="tip">读取失败</span>'; return; }
    const fields = d.fields || {};
    const secs = d.section_fields || {};
    const titles = d.sections || {};
    let html = '';
    for(const sec of Object.keys(secs)){
      html += '<h3 style="font-size:13px;color:var(--dim);margin:14px 0 6px;font-weight:600">'
           + (titles[sec] || sec) + '</h3>';
      html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px">';
      for(const key of secs[sec]){
        const f = fields[key];
        if(!f) continue;
        const id = 'cfg_' + key.replace(/\./g, '_');
        let ctrl;
        if(f.options && f.options.length){
          ctrl = '<select id="' + id + '" data-key="' + key + '">'
            + f.options.map(o => '<option value="' + o.value + '"'
                + (String(f.value) === String(o.value) ? ' selected' : '') + '>'
                + o.label + '</option>').join('')
            + '</select>';
        } else if(f.type === 'bool'){
          ctrl = '<label class="sw"><input type="checkbox" id="' + id + '" data-key="' + key + '"'
            + (f.value ? ' checked' : '') + '><span class="track"></span></label>';
        } else if(f.type === 'int' || f.type === 'float'){
          ctrl = '<input type="number" id="' + id + '" data-key="' + key + '" value="' + f.value
            + '"' + (f.type === 'float' ? ' step="0.1"' : ' step="1"') + '>';
        } else {
          const secret = /token|key|password|secret/i.test(key);
          ctrl = '<input type="' + (secret ? 'password' : 'text') + '" id="' + id
            + '" data-key="' + key + '" value="' + String(f.value == null ? '' : f.value) + '">';
        }
        html += '<div><label class="fl">' + f.label + '</label>' + ctrl
          + '<div class="tip">' + (f.default !== undefined && f.default !== '' && f.default !== f.value
              ? '默认 ' + f.default : '') + '</div></div>';
      }
      html += '</div>';
    }
    box.innerHTML = html;
    tip.textContent = Object.keys(fields).length + ' 项';
  }catch(e){ box.innerHTML = '<span class="tip">读取失败：' + e + '</span>'; }
}

async function saveConfig(){
  const tip = document.getElementById('cfgTip');
  const upd = {};
  document.querySelectorAll('#cfgBox [data-key]').forEach(el => {
    const key = el.getAttribute('data-key');
    if(el.type === 'checkbox') upd[key] = el.checked;
    else if(el.type === 'number'){
      const n = parseFloat(el.value);
      if(!isNaN(n)) upd[key] = Number.isInteger(n) ? n : n;
    }
    else upd[key] = el.value;
  });
  try{
    const r = await fetch(base + '/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({config: upd})
    });
    const d = await r.json();
    if(d.ok){
      tip.textContent = '已保存 ' + (d.saved || []).length + ' 项';
      setTimeout(() => tip.textContent = '', 3000);
    } else {
      tip.textContent = '保存失败：' + (d.error || '');
    }
  }catch(e){ tip.textContent = '请求失败：' + e; }
}

async function loadStatus(){
  try{
    const r = await fetch(base + '/api/status');
    const d = await r.json();
    $('status').textContent = '服务运行中';
    $('status').className = 'badge';
    const backends = (d.backends||[]).map(b=>b.name||b).join('、') || '未检测到';
    const loaded = Object.keys(d.loaded_models||{});
    $('statusKv').innerHTML =
      '<dt>监听地址</dt><dd class="mono">' + d.host + ':' + d.port + '</dd>' +
      '<dt>推理后端</dt><dd>' + backends + '</dd>' +
      '<dt>已加载模型</dt><dd>' + (loaded.length ? loaded.join('、') : '无') + '</dd>';
    $('u1').textContent = base + '/v1/chat/completions';
    $('u2').textContent = base + '/v1/models';
    $('u3').textContent = base + '/api/qwen  ' + base + '/api/llama';
  }catch(e){
    $('status').textContent = '服务异常';
    $('status').className = 'badge off';
    $('statusKv').innerHTML = '<dt>错误</dt><dd>' + e + '</dd>';
  }
}

async function loadRuntime(){
  const box = document.getElementById('runtimeBox');
  const btn = document.getElementById('btnRuntime');
  try{
    const r = await fetch(base + '/api/runtime');
    const d = await r.json();
    if(d.installed){
      box.innerHTML = '<div class="kv">'
        + '<dt>状态</dt><dd style="color:var(--ok)">已就绪</dd>'
        + '<dt>路径</dt><dd class="mono">' + (d.path || '-') + '</dd>'
        + '<dt>可用后端</dt><dd>' + ((d.backends||[]).join('、') || '-') + '</dd>'
        + '</div>';
      btn.textContent = '重新安装';
    } else {
      box.innerHTML = '<div class="kv">'
        + '<dt>状态</dt><dd style="color:var(--warn)">未安装</dd>'
        + '<dt>说明</dt><dd>点击「一键安装」自动下载并配置 llama-server，无需手动处理 PATH</dd>'
        + '</div>';
      btn.textContent = '一键安装';
    }
  }catch(e){
    box.innerHTML = '<span class="tip">检测失败：' + e + '</span>';
  }
}

async function installRuntime(){
  const btn = document.getElementById('btnRuntime');
  const box = document.getElementById('runtimeBox');
  btn.disabled = true;
  btn.textContent = '安装中…';
  try{
    const r = await fetch(base + '/api/install_runtime', {method:'POST'});
    const d = await r.json();
    if(!d.ok){ box.innerHTML = '<span class="tip">启动失败：' + (d.error||'') + '</span>'; btn.disabled=false; return; }
    pollRuntime();
  }catch(e){
    box.innerHTML = '<span class="tip">请求失败：' + e + '</span>';
    btn.disabled = false;
  }
}

async function pollRuntime(){
  const box = document.getElementById('runtimeBox');
  const btn = document.getElementById('btnRuntime');
  try{
    const r = await fetch(base + '/api/runtime_progress');
    const d = await r.json();
    const log = (d.log || []).slice(-8).map(l => '  ' + l).join('\n');
    if(d.installing){
      box.innerHTML = '<pre class="logbox" style="max-height:180px;overflow:auto">' + log + '</pre>';
      setTimeout(pollRuntime, 1500);
    } else {
      const res = d.result || {};
      if(res.ok){
        box.innerHTML = '<div class="kv">'
          + '<dt>状态</dt><dd style="color:var(--ok)">安装完成</dd>'
          + '<dt>说明</dt><dd>' + (res.message||'') + '</dd>'
          + '<dt>路径</dt><dd class="mono">' + (res.path||'') + '</dd>'
          + '</div>';
        btn.disabled = false; btn.textContent = '重新安装';
        loadStatus();
      } else {
        box.innerHTML = '<pre class="logbox">' + log + (res.message ? '\n  ' + res.message : '') + '</pre>';
        btn.disabled = false; btn.textContent = '重试安装';
      }
    }
  }catch(e){
    btn.disabled = false; btn.textContent = '一键安装';
  }
}

async function loadModels(){
  try{
    const r = await fetch(base + '/v1/models');
    const d = await r.json();
    const items = d.data || [];
    if(!items.length){
      $('modelBody').innerHTML = '<tr><td colspan="6" style="color:var(--dim)">暂无已下载模型，请把 GGUF 放进 models 目录</td></tr>';
      $('modelTip').textContent = '';
      return;
    }
    // 查哪些已加载
    let loaded = {};
    try{
      const lr = await fetch(base + '/api/loaded');
      const ld = await lr.json();
      loaded = ld.models || {};
    }catch(e){}
    loadedIds = loaded;

    $('modelBody').innerHTML = items.map(m => {
      const on = !!loaded[m.id];
      const safe = m.id.replace(/'/g, "\\'");
      const loadBtn = on
        ? '<button onclick="unloadModel(\'' + safe + '\')">卸载</button>'
        : '<button onclick="loadModel(\'' + safe + '\')">加载</button>';
      const delBtn = '<button class="danger" onclick="removeModel(\'' + safe + '\')">删除</button>';
      return '<tr><td>' + m.id + (on ? ' <span style="color:var(--ok)">● 已加载</span>' : '') + '</td>'
        + '<td>-</td><td>-</td><td>-</td><td>-</td>'
        + '<td><span class="btnrow">' + loadBtn + delBtn + '</span></td></tr>';
    }).join('');
    $('modelTip').textContent = '共 ' + items.length + ' 个';
  }catch(e){
    $('modelBody').innerHTML = '<tr><td colspan="6" style="color:var(--err)">加载失败：' + e + '</td></tr>';
  }
}

async function loadModel(mid){
  $('modelTip').textContent = '正在加载 ' + mid + '…';
  try{
    const r = await fetch(base + '/api/load_model', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid})
    });
    const d = await r.json();
    if(d.ok){
      $('modelTip').textContent = '已加载 ' + mid;
      loadModels(); loadStatus();
    } else {
      $('modelTip').textContent = '加载失败：' + (d.error || '未知错误');
    }
  }catch(e){ $('modelTip').textContent = '请求失败：' + e; }
}

async function unloadModel(mid){
  $('modelTip').textContent = '正在卸载 ' + mid + '…';
  try{
    const r = await fetch(base + '/api/unload_model', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid})
    });
    const d = await r.json();
    $('modelTip').textContent = d.ok ? ('已卸载 ' + mid) : ('卸载失败：' + (d.error || ''));
    loadModels(); loadStatus();
  }catch(e){ $('modelTip').textContent = '请求失败：' + e; }
}

// 删除模型（连文件一起删），删除前二次确认
async function removeModel(mid){
  if(!confirm('确定删除 ' + mid + ' 的模型文件？此操作不可恢复。')) return;
  if(loadedIds && loadedIds[mid]){
    $('modelTip').textContent = '模型正在加载中，先卸载再删…';
    await unloadModel(mid);
  }
  $('modelTip').textContent = '正在删除 ' + mid + '…';
  try{
    const r = await fetch(base + '/api/library_remove', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid})
    });
    const d = await r.json();
    $('modelTip').textContent = d.ok ? ('已删除 ' + mid) : ('删除失败：' + (d.error || ''));
    loadModels(); loadStatus();
  }catch(e){ $('modelTip').textContent = '请求失败：' + e; }
}

async function ask(){
  const q = $('q').value.trim();
  if(!q) return;
  log('请求中…');
  try{
    const r = await fetch(base + '/v1/chat/completions', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model:'qwen', messages:[{role:'user',content:q}]})
    });
    const d = await r.json();
    if(d.error){ log('错误：' + (d.error.message || JSON.stringify(d.error))); return; }
    const txt = d.choices?.[0]?.message?.content || JSON.stringify(d);
    log(txt);
  }catch(e){ log('请求失败：' + e); }
}

async function loadRecs(refresh){
  $('recTip').textContent = refresh ? '正在抓取模型目录，可能需要十几秒…' : '加载中…';
  try{
    const r = await fetch(base + '/api/recommend?limit=20&refresh=' + (refresh ? 1 : 0));
    const d = await r.json();
    if(!d.ok){ $('recTip').textContent = '失败：' + (d.error?.message || '未知错误'); return; }
    const ms = d.models || [];
    if(!ms.length){
      $('recBody').innerHTML = '<tr><td colspan="6" style="color:var(--dim)">没有匹配到模型</td></tr>';
      $('recTip').textContent = '没有匹配到模型';
      return;
    }
    $('recBody').innerHTML = ms.map(m => {
      const tier = m.tier || '-';
      const color = tier === '首选' ? 'var(--ok)' : tier === '备选' ? 'var(--accent)'
                  : tier === '不适合' ? 'var(--err)' : 'var(--warn)';
      const size = m.size_gb ? m.size_gb.toFixed(2) + ' GB' : '-';
      const ctx = m.ctx_len ? Math.round(m.ctx_len/1024) + 'K' : '-';
      // 量化版本下拉：同一模型可能有 Q4_K_M / Q5_K_M / Q8_0 多份
      const vars = m.variants || [];
      let picker = '<button onclick="startDl(\'' + m.id.replace(/'/g,"\\'") + '\')">下载</button>';
      if(vars.length > 1){
        const opts = vars.map(v =>
          '<option value="' + v.quant + '"'
          + (v.quant === m.quant ? ' selected' : '') + '>'
          + v.quant + (v.size_gb ? ' (' + v.size_gb.toFixed(1) + 'G)' : '')
          + '</option>').join('');
        picker = '<select class="qsel" id="q_' + m.id.replace(/[^a-zA-Z0-9]/g,"_") + '">'
          + opts + '</select>'
          + '<button onclick="startDl2(\'' + m.id.replace(/'/g,"\\'") + '\')">下载</button>';
      }
      return '<tr><td>' + m.name + '</td>'
        + '<td style="color:' + color + '">' + tier + '</td>'
        + '<td>' + (m.params_b ? m.params_b + 'B' : '-') + '</td>'
        + '<td>' + ctx + '</td><td>' + size + '</td><td>' + picker + '</td></tr>';
    }).join('');
    $('recTip').textContent = '共 ' + ms.length + ' 个候选';
  }catch(e){ $('recTip').textContent = '请求失败：' + e; }
}

async function startDl(mid){
  try{
    const r = await fetch(base + '/api/download', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid})
    });
    const d = await r.json();
    if(!d.ok){ $('recTip').textContent = '下载失败：' + (d.error || ''); return; }
    $('recTip').textContent = '已开始下载：' + mid;
    pollDownloads();
  }catch(e){ $('recTip').textContent = '请求失败：' + e; }
}

// 带量化版本下载
async function startDl2(mid){
  const sel = document.getElementById('q_' + mid.replace(/[^a-zA-Z0-9]/g,"_"));
  const quant = sel ? sel.value : '';
  $('recTip').textContent = '开始下载 ' + mid + ' (' + (quant || '默认') + ')…';
  try{
    const r = await fetch(base + '/api/download', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid, quant: quant})
    });
    const d = await r.json();
    if(!d.ok){ $('recTip').textContent = '下载失败：' + (d.error || ''); return; }
    $('recTip').textContent = '已开始下载：' + mid + ' (' + (quant || '默认') + ')';
    pollDownloads();
  }catch(e){ $('recTip').textContent = '请求失败：' + e; }
}

async function pollDownloads(){
  try{
    const r = await fetch(base + '/api/downloads');
    const d = await r.json();
    const tasks = Object.values(d.tasks || {});
    if(!tasks.length){ $('dlBox').innerHTML = '<span class="tip">暂无下载任务</span>'; return; }
    $('dlBox').innerHTML = tasks.map(t => {
      const pct = t.percent || 0;
      const color = t.status === 'done' ? 'var(--ok)'
                  : t.status === 'error' ? 'var(--err)' : 'var(--accent)';
      const bar = '<div style="background:var(--panel2);border-radius:4px;height:6px;margin-top:6px;overflow:hidden">'
        + '<div style="width:' + pct + '%;height:100%;background:' + color + ';transition:width .3s"></div></div>';
      const label = t.status === 'done' ? '已完成'
                  : t.status === 'error' ? ('失败：' + (t.error||''))
                  : (pct.toFixed(1) + '%');
      // 体积 / 速度 / 剩余时间
      let detail = '';
      if(t.status !== 'error'){
        const parts = [];
        if(t.total){
          parts.push(fmtGB(t.downloaded) + ' / ' + fmtGB(t.total));
        } else if(t.downloaded){
          parts.push(fmtGB(t.downloaded));
        }
        if(t.speed_bps && t.status !== 'done'){
          parts.push(fmtSpeed(t.speed_bps));
        }
        if(t.eta && t.status !== 'done'){
          parts.push('剩余 ' + fmtETA(t.eta));
        }
        if(parts.length) detail = '<span class="tip">' + parts.join(' · ') + '</span>';
      }
      let extra = '';
      if(t.status === 'done') extra = ' <button onclick="loadModels()">刷新模型库</button>';
      if(t.status === 'error') extra = ' <button onclick="startDl(\'' + t.model_id.replace(/'/g,"\\'") + '\')">重试</button>';
      return '<div style="margin-bottom:10px"><b>' + t.model_id
        + (t.quant ? ' <span class="tip">(' + t.quant + ')</span>' : '') + '</b> '
        + '<span style="color:' + color + '">' + label + '</span> ' + extra
        + (detail ? '<br>' + detail : '') + bar + '</div>';
    }).join('');
  }catch(e){}
}

function fmtGB(bytes){
  if(!bytes) return '0 MB';
  const mb = bytes / 1048576;
  return mb >= 1024 ? (mb/1024).toFixed(2) + ' GB' : mb.toFixed(0) + ' MB';
}
function fmtSpeed(bps){
  const mb = bps / 1048576;
  return mb >= 1 ? mb.toFixed(1) + ' MB/s' : (bps/1024).toFixed(0) + ' KB/s';
}
function fmtETA(sec){
  if(sec < 60) return sec + ' 秒';
  if(sec < 3600) return Math.round(sec/60) + ' 分钟';
  return (sec/3600).toFixed(1) + ' 小时';
}

loadStatus(); loadModels(); pollDownloads(); loadConfig(); loadRuntime();
setInterval(loadStatus, 5000);
setInterval(pollDownloads, 2000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 统计


class Stats:
    """线程安全的请求/Token 统计。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.started_at = time.time()

    def count_request(self, failed: bool = False) -> None:
        with self._lock:
            self.requests_total += 1
            if failed:
                self.requests_failed += 1

    def count_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self.tokens_in += int(tokens_in)
            self.tokens_out += int(tokens_out)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }


# ---------------------------------------------------------------- API 服务


class ApiServer:
    """聚合后端、模型目录、别名映射与统计信息。"""

    def __init__(self, cfg: Config | None = None, backends: list[Backend] | None = None,
                 host: str = "127.0.0.1", port: int = 8080, token: str = ""):
        self.cfg = cfg or Config.default()
        self.backends = backends if backends is not None else detect_backends(self.cfg)
        self.host = host
        self.port = int(port)
        self.token = str(token or self.cfg.get("app.token", "") or "")
        self.stats = Stats()

        # 已加载模型登记: id -> {"backend": Backend, "base_url": str, "family": str}
        self._loaded: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        # family -> alias 映射（catalog family 字段推断）
        self._family_map: dict[str, str] = {}

        # 显式别名: alias -> model_id
        self._explicit_aliases: dict[str, str] = {
            str(k): str(v)
            for k, v in (self.cfg.get("api.aliases", {}) or {}).items()
        }

        # 模型目录 / 推荐结果 / 下载任务 缓存
        self._catalog_cache: list[Any] | None = None
        self._rec_cache: list[Any] | None = None
        self._hw_cache: Any = None
        self._downloads: dict[str, dict[str, Any]] = {}

        # 推理运行库安装状态
        self._runtime_installing = False
        self._runtime_log: list[str] = []
        self._runtime_info: dict[str, Any] = {}

        # 本地模型库缓存
        self._library_cache: Any = None

    # ---------------- 后端 ----------------

    @property
    def active_backend(self) -> Backend | None:
        """当前激活后端（第一个可用的）。"""
        for b in self.backends:
            if b.is_available():
                return b
        return self.backends[0] if self.backends else None

    def register_loaded(self, model_id: str, backend: Backend, base_url: str,
                        family: str = "") -> None:
        """登记一个已加载的模型。"""
        with self._lock:
            self._loaded[str(model_id)] = {
                "backend": backend,
                "base_url": base_url,
                "family": family,
                "loaded_at": time.time(),
            }

    def unregister_loaded(self, model_id: str) -> None:
        with self._lock:
            self._loaded.pop(str(model_id), None)

    def set_family_map(self, family_map: dict[str, str]) -> None:
        """设置 ``family → alias`` 映射（自动从 catalog 推断后传入）。"""
        with self._lock:
            self._family_map = dict(family_map or {})

    def refresh_family_map(self, catalog_models: list[dict[str, Any]] | None = None) -> dict[str, str]:
        """从 catalog 模型列表推断 family → alias 映射。"""
        source = catalog_models
        if source is None:
            source = []
            for b in self.backends:
                try:
                    source += b.list_models()
                except Exception:  # noqa: BLE001 - 列模型失败不应影响启动
                    pass
        fmap = proxy.infer_family_map(source)
        self.set_family_map(fmap)
        return fmap

    # ---------------- 别名 ----------------

    def aliases(self) -> dict[str, Any]:
        """返回「别名 → 模型 id」视图。"""
        with self._lock:
            out: dict[str, Any] = {}
            for model_id, info in self._loaded.items():
                fam = str(info.get("family") or "").lower()
                alias = self._family_map.get(fam) or (fam if fam else DEFAULT_ALIAS)
                out.setdefault(alias, model_id)
            for alias, mid in self._explicit_aliases.items():
                out[alias] = mid
            if DEFAULT_ALIAS not in out and self._loaded:
                first = next(iter(self._loaded))
                out[DEFAULT_ALIAS] = first
            return out

    def resolve_alias(self, alias: str) -> str | None:
        """解析别名为模型 id（失败返回 None）。"""
        alias_key = str(alias or "").strip().lower()
        with self._lock:
            # 1) 显式配置
            if alias_key in self._explicit_aliases:
                mid = self._explicit_aliases[alias_key]
                if not self._loaded or mid in self._loaded:
                    return mid
            # 2) 已登记别名
            amap = self.aliases()
            if alias_key in amap:
                return amap[alias_key]
            # 3) 从已加载列表挑
            if self._loaded:
                return pick_model_for_alias(
                    alias_key,
                    [{"id": mid, "family": info.get("family", "")}
                     for mid, info in self._loaded.items()],
                    self._family_map,
                )
        return None

    # ---------------- 状态 ----------------

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            loaded = {
                mid: {
                    "backend": info["backend"].name,
                    "family": info.get("family", ""),
                    "base_url": info.get("base_url", ""),
                    "loaded_at": info.get("loaded_at", 0),
                }
                for mid, info in self._loaded.items()
            }
            family_map = dict(self._family_map)
            explicit = dict(self._explicit_aliases)
        return {
            "service": "imdeploy",
            "host": self.host,
            "port": self.port,
            "loaded_models": loaded,
            "backends": [b.status() if hasattr(b, "status") else b.info() for b in self.backends],
            "active_backend": self.active_backend.name if self.active_backend else None,
            "family_map": family_map,
            "explicit_aliases": explicit,
            "aliases": self.aliases(),
            "stats": self.stats.snapshot(),
            "token_required": bool(self.token),
        }

    # ---------------- 转发 ----------------

    def backend_url_for(self, model_id: str) -> str:
        """取指定模型对应后端的 base_url。"""
        with self._lock:
            info = self._loaded.get(str(model_id))
            if info:
                return str(info.get("base_url") or "")
        backend = self.active_backend
        if backend is None:
            return ""
        try:
            return backend.base_url() or ""
        except Exception:  # noqa: BLE001
            return ""

    def forward_for_model(self, model_id: str, path: str, payload: dict[str, Any],
                          timeout: float = 120.0) -> tuple[int, dict[str, Any]]:
        """把请求转发到模型对应的后端。"""
        base = self.backend_url_for(model_id)
        if not base:
            return 0, {"error": {
                "message": f"模型未加载或后端不可用：{model_id}",
                "type": "api_error",
            }}
        return forward_json(base, path, payload, timeout=timeout)

    # ---------------- 模型目录 / 推荐 / 下载 ----------------

    def get_recommendations(self, limit: int = 20, min_ctx: int = 4096,
                            refresh: bool = False) -> list[dict[str, Any]]:
        """返回按本机硬件排序的推荐模型。结果缓存，避免每次重复抓取。"""
        from .catalog import build_catalog
        from .hardware import detect
        from .recommend import recommend

        with self._lock:
            cat = self._catalog_cache
            recs = self._rec_cache

        if refresh or cat is None:
            try:
                cat = build_catalog(use_cache=not refresh)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"抓取模型目录失败：{exc}") from exc
            with self._lock:
                self._catalog_cache = cat

        hw = self._hw_cache
        if hw is None:
            hw = detect()
            with self._lock:
                self._hw_cache = hw

        recs = recommend(cat, hw, limit=limit, min_ctx=min_ctx)
        with self._lock:
            self._rec_cache = recs

        # 同一仓库的多个量化版本聚合成 variants，让前端能一次看清体积差异
        try:
            from .catalog import group_by_model  # noqa: PLC0415

            variants_map = {g["id"]: g["variants"] for g in group_by_model(cat)}
        except Exception:  # noqa: BLE001
            variants_map = {}

        # 把变体挂到 entry.meta 上，下载时才能按量化挑具体文件
        with self._lock:
            for r in (self._rec_cache or []):
                vs = variants_map.get(r.entry.id)
                if vs:
                    try:
                        r.entry.meta["variants"] = vs
                    except Exception:  # noqa: BLE001
                        pass

        out: list[dict[str, Any]] = []
        for r in recs:
            e = r.entry
            out.append({
                "id": e.id,
                "name": e.name,
                "family": e.family,
                "params_b": e.params_b,
                "quant": e.quant,
                "ctx_len": e.ctx_len,
                "size_gb": r.estimated_size_gb or e.size_gb,
                "tier": r.tier,
                "score": r.score,
                "fit": r.fit,
                "reasons": r.reasons,
                "download_url": e.download_url,
                "repo_url": e.repo_url,
                "variants": variants_map.get(e.id, []),
            })
        return out

    def start_download(self, model_id: str, quant: str = "") -> dict[str, Any]:
        """后台下载一个模型。

        ``quant`` 可指定量化版本（如 Q5_K_M）；留空则用推荐版本。
        """
        from .downloader import Downloader

        key = f"{model_id}::{quant}" if quant else model_id
        with self._lock:
            if key in self._downloads and self._downloads[key].get("status") == "downloading":
                return {"ok": False, "error": "该模型正在下载中", "model_id": model_id}

        url = ""
        size_hint = 0.0
        filename = ""
        with self._lock:
            for r in (self._rec_cache or []):
                if r.entry.id != model_id:
                    continue
                size_hint = r.estimated_size_gb
                url = r.entry.download_url
                filename = (url or "").rsplit("/", 1)[-1]
                # 指定了量化版本时，从变体里挑对应的那份
                if quant:
                    for v in (r.entry.meta.get("variants") or []):
                        if str(v.get("quant", "")).upper() == quant.upper():
                            url = v.get("download_url") or url
                            filename = v.get("filename") or filename
                            size_hint = v.get("size_gb") or size_hint
                            break
                break
        if not url:
            return {"ok": False, "error": f"找不到模型的下载地址：{model_id}"}

        safe = "".join(c for c in (filename or f"{model_id}.gguf") if c.isalnum()
                       or c in "-_.")[:120]
        dest = self.cfg.models_dir / safe
        state: dict[str, Any] = {
            "model_id": model_id, "quant": quant, "status": "downloading",
            "percent": 0.0, "downloaded": 0, "total": 0, "error": "",
            "dest": str(dest), "size_hint_gb": size_hint, "filename": safe,
        }
        with self._lock:
            self._downloads[key] = state

        def worker() -> None:
            dl = Downloader()

            def progress(task) -> None:
                now = time.time()
                with self._lock:
                    prev_t = state.get("_t") or 0.0
                    prev_b = state.get("_b") or 0
                    # 用滑动窗口算速度，避免瞬时抖动
                    if prev_t and now - prev_t >= 0.4 and task.downloaded >= prev_b:
                        inst = (task.downloaded - prev_b) / (now - prev_t)
                        old = state.get("speed_bps") or 0.0
                        # 一阶低通，让读数稳定
                        state["speed_bps"] = inst if not old else old * 0.6 + inst * 0.4
                        state["_t"] = now
                        state["_b"] = task.downloaded
                    elif not prev_t:
                        state["_t"] = now
                        state["_b"] = task.downloaded

                    state["percent"] = round(task.percent, 1)
                    state["downloaded"] = task.downloaded
                    state["total"] = task.size

                    speed = state.get("speed_bps") or 0.0
                    remain = (task.size - task.downloaded) if task.size else 0
                    state["eta"] = int(remain / speed) if (speed > 1024 and remain > 0) else 0

            try:
                task = dl.download(url, dest, progress=progress)
                with self._lock:
                    state["status"] = "done" if task.status == "done" else "error"
                    state["error"] = task.error or ""
                    state["percent"] = 100.0 if task.status == "done" else state["percent"]
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    state["status"] = "error"
                    state["error"] = str(exc)

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "model_id": model_id, "dest": str(dest)}

    def download_status(self) -> dict[str, Any]:
        with self._lock:
            # 内部计时用的下划线字段不对外暴露
            return {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in self._downloads.items()
            }

    # ---------------- 模型加载 / 卸载 ----------------

    def start_model(self, model_id: str) -> dict[str, Any]:
        """加载一个本地模型到推理后端（llama-server / Ollama）。"""
        from . import backends as backends_mod  # noqa: PLC0415

        with self._lock:
            if model_id in self._loaded:
                return {"ok": True, "model_id": model_id, "note": "模型已加载"}

        backends = backends_mod.detect_backends(self.cfg)
        if not backends:
            return {"ok": False, "error": "没有可用的推理后端（找不到 llama-server？）"}

        backend = next((b for b in backends if b.is_available()), backends[0])
        try:
            info = backend.start_model(
                model_id,
                ctx_size=int(self.cfg.get("backend.context_size", 8192)),
                n_gpu_layers=int(self.cfg.get("backend.gpu_layers", 0)),
                threads=int(self.cfg.get("backend.threads", 0)) or None,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        base_url = info.get("base_url", "") if isinstance(info, dict) else ""
        with self._lock:
            self._loaded[model_id] = {
                "backend": backend,
                "base_url": base_url,
                "family": info.get("family", "") if isinstance(info, dict) else "",
                "loaded_at": time.time(),
            }
        return {"ok": True, "model_id": model_id, "base_url": base_url,
                "info": {k: v for k, v in info.items() if k != "command"}
                if isinstance(info, dict) else {}}

    def stop_model(self, model_id: str = "") -> dict[str, Any]:
        """卸载模型。model_id 为空则停掉全部。"""
        from . import backends as backends_mod  # noqa: PLC0415

        targets: list[Any] = []
        with self._lock:
            if model_id:
                entry = self._loaded.get(model_id)
                if entry:
                    targets.append(entry["backend"])
                else:
                    # 可能是同一后端在不同 key 下登记
                    targets = list({id(b): b for b in backends_mod.detect_backends(self.cfg)}.values())
            else:
                targets = list({id(b): b for b in backends_mod.detect_backends(self.cfg)}.values())

        stopped = []
        for b in targets:
            try:
                if b.stop_model():
                    stopped.append(getattr(b, "name", str(b)))
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            if model_id:
                self._loaded.pop(model_id, None)
            else:
                self._loaded.clear()
        return {"ok": True, "stopped": stopped}

    def loaded_models(self) -> dict[str, Any]:
        """已加载模型的详细信息。"""
        with self._lock:
            return {
                mid: {
                    "base_url": v.get("base_url", ""),
                    "backend": getattr(v.get("backend"), "name", ""),
                    "loaded_at": v.get("loaded_at", 0),
                }
                for mid, v in self._loaded.items()
            }

    # ---------------- 本地模型库 ----------------

    def _library(self):
        from .library import ModelLibrary  # noqa: PLC0415

        with self._lock:
            if self._library_cache is None:
                self._library_cache = ModelLibrary(self.cfg)
            return self._library_cache

    def library_list(self) -> list[dict[str, Any]]:
        try:
            lib = self._library()
        except Exception:  # noqa: BLE001
            return []
        out = []
        for m in lib.list():
            out.append({
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "params_b": m.params_b,
                "quant": m.quant,
                "ctx_len": m.ctx_len,
                "size_gb": m.size_gb,
                "path": m.path,
                "instruct": m.instruct,
                "long_context": m.long_context,
                "description": m.description,
                "source": m.source,
                "added_at": m.added_at,
            })
        return out

    def library_scan(self) -> dict[str, Any]:
        try:
            lib = self._library()
            added = lib.scan()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "added": len(added), "models": self.library_list()}

    def library_remove(self, mid: str, delete_file: bool = True) -> dict[str, Any]:
        try:
            lib = self._library()
            ok = lib.remove(mid, delete_file=delete_file)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": bool(ok), "model_id": mid}

    def runtime_status(self) -> dict[str, Any]:
        """检测推理运行库是否就绪。"""
        from . import backends as backends_mod  # noqa: PLC0415

        backends = backends_mod.detect_backends(self.cfg)
        llama = next((b for b in backends if getattr(b, "name", "") == "llama.cpp"), None)
        available = bool(llama and llama.is_available())
        path = getattr(llama, "server_path", "") if llama else ""
        return {
            "ok": True,
            "installed": available,
            "path": path or "",
            "backends": [getattr(b, "name", str(b)) for b in backends],
        }

    def install_runtime(self) -> dict[str, Any]:
        """后台下载并安装 llama.cpp 运行库。"""
        from .runtime_install import detect_accel, install_llama_runtime

        with self._lock:
            if self._runtime_installing:
                return {"ok": False, "error": "运行库正在安装中，请稍候"}

            self._runtime_installing = True
            self._runtime_log = []

        def on_log(msg: str) -> None:
            with self._lock:
                self._runtime_log.append(str(msg))
                del self._runtime_log[:-50]

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                on_log(f"下载中 {done * 100.0 / total:.0f}%（{done / 1048576:.1f}/"
                       f"{total / 1048576:.1f} MB）")
            else:
                on_log(f"下载中 {done / 1048576:.1f} MB")

        def worker() -> None:
            try:
                accel = detect_accel()
                on_log(f"检测到加速后端：{accel}")
                info = install_llama_runtime(
                    self.cfg.base_dir / "bin",
                    proxy=str(self.cfg.get("app.proxy", "") or "") or None,
                    progress=on_progress,
                    log=on_log,
                )
                with self._lock:
                    self._runtime_info = {
                        "ok": info.ok,
                        "message": info.message,
                        "path": info.exe_path,
                        "backend": info.backend,
                        "version": info.version,
                    }
                    if info.ok:
                        self.cfg.set("backend.llama_server", info.exe_path)
                        try:
                            self.cfg.save()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._runtime_info = {"ok": False, "message": str(exc)}
            finally:
                with self._lock:
                    self._runtime_installing = False

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "message": "已开始安装"}

    def runtime_progress(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "installing": bool(self._runtime_installing),
                "log": list(self._runtime_log),
                "result": dict(self._runtime_info or {}),
            }

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """更新配置（WebUI 表单提交），只接受已存在的键。"""
        if not isinstance(updates, dict):
            return {"ok": False, "error": "config 必须是对象"}

        changed: list[str] = []
        for dotted, value in updates.items():
            if not isinstance(dotted, str) or "." not in dotted:
                continue
            # 只允许写已存在的配置项，避免拼错键名写出一堆垃圾
            current = self.cfg.get(dotted, None)
            if current is None and not _config_key_exists(self.cfg, dotted):
                continue
            if current == value:
                continue
            self.cfg.set(dotted, value)
            changed.append(dotted)

        if changed:
            try:
                self.cfg.save()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"保存失败：{exc}", "saved": []}
        return {"ok": True, "saved": changed}


# ---------------------------------------------------------------- 配置辅助


def _config_key_exists(cfg, dotted: str) -> bool:
    """检查点分键是否存在（只按配置本身的结构判断，不看默认值）。"""
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    return True


def _config_payload(cfg) -> dict[str, Any]:
    """给前端渲染表单用的配置负载。

    每个字段带上类型信息，前端据此决定用 input / number / select / switch。
    """
    from .config import DEFAULT_CONFIG, FIELD_LABELS, SECTION_TITLES, VALUE_HINTS

    def walk(node: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not isinstance(node, dict):
            return out
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                nested = walk(value, f"{dotted}.")
                if nested:
                    out.update(nested)
            else:
                out[dotted] = value
        return out

    values = walk(getattr(cfg, "_data", {}) or {})
    fields: dict[str, Any] = {}
    for dotted, value in values.items():
        default = None
        node: Any = DEFAULT_CONFIG
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None and not isinstance(node, dict):
            default = node
        ftype = ("bool" if isinstance(value, bool)
                 else "int" if isinstance(value, int)
                 else "float" if isinstance(value, float)
                 else "text")
        entry: dict[str, Any] = {
            "value": value,
            "type": ftype,
            "label": FIELD_LABELS.get(dotted, dotted.split(".")[-1]),
        }
        # 枚举型：hints 里登记了候选值就用下拉
        opts = VALUE_HINTS.get(dotted)
        if opts:
            entry["options"] = [{"value": k, "label": v, "hint": ""}
                                for k, v in opts.items()]
        elif default is not None and not isinstance(default, (dict, list)):
            entry["default"] = default
        fields[dotted] = entry

    sections: dict[str, list[str]] = {}
    for dotted in fields:
        sections.setdefault(dotted.split(".")[0], []).append(dotted)

    return {
        "ok": True,
        "fields": fields,
        "sections": {k: SECTION_TITLES.get(k, k) for k in sections},
        "section_fields": sections,
    }


# ---------------------------------------------------------------- Handler


def create_handler(server: ApiServer) -> type[BaseHTTPRequestHandler]:
    """闭包绑定一个 ApiServer，生成请求处理器类。"""

    class Handler(BaseHTTPRequestHandler):
        # 关闭默认 stderr 日志刷屏
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            pass

        # ---------------- 通用 ----------------

        def _send_json(self, data: Any, status: int = 200, extra_headers: dict | None = None) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

        def _send_error_json(self, message: str, err_type: str = "invalid_request_error",
                             status: int = 400, code: Any = None) -> None:
            err: dict[str, Any] = {"message": message, "type": err_type}
            if code is not None:
                err["code"] = code
            self._send_json({"error": err}, status=status)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8", "replace"))

        # ---------------- 鉴权 ----------------

        def _check_token(self) -> bool:
            """token 非空时要求 Authorization: Bearer 或 ?token=。"""
            tok = server.token
            if not tok:
                return True
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer ") and auth[len("Bearer "):].strip() == tok:
                return True
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            got = (qs.get("token") or [""])[0]
            return got == tok

        # ---------------- OPTIONS ----------------

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors_headers()
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()

        # ---------------- GET ----------------

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if not self._check_token():
                server.stats.count_request(failed=True)
                self._send_error_json("访问令牌校验失败", "invalid_request_error", 401, "invalid_token")
                return
            if path in ("/healthz", "/api/status/health"):
                self._send_json({"ok": True, "service": "imdeploy"})
                server.stats.count_request()
                return
            if path == "/api/status":
                self._send_json(server.status_payload())
                server.stats.count_request()
                return
            if path in ("/v1/models", "/api/models"):
                self._handle_models()
                return
            if path in ("/api/recommend", "/api/catalog"):
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                try:
                    limit = int((q.get("limit") or ["20"])[0])
                    min_ctx = int((q.get("min_ctx") or ["4096"])[0])
                    refresh = (q.get("refresh") or ["0"])[0] in ("1", "true", "yes")
                    data = server.get_recommendations(limit=limit, min_ctx=min_ctx,
                                                      refresh=refresh)
                except Exception as exc:  # noqa: BLE001
                    self._send_error_json(f"获取推荐失败：{exc}", "api_error", 500, "recommend_failed")
                    return
                self._send_json({"ok": True, "count": len(data), "models": data})
                server.stats.count_request()
                return
            if path == "/api/downloads":
                self._send_json({"ok": True, "tasks": server.download_status()})
                server.stats.count_request()
                return
            if path == "/api/config":
                self._send_json(_config_payload(server.cfg))
                server.stats.count_request()
                return
            if path == "/api/loaded":
                self._send_json({"ok": True, "models": server.loaded_models()})
                server.stats.count_request()
                return
            if path == "/api/library":
                self._send_json({"ok": True, "models": server.library_list()})
                server.stats.count_request()
                return
            if path == "/api/library_scan":
                res = server.library_scan()
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
                return
            if path == "/api/runtime":
                self._send_json(server.runtime_status())
                server.stats.count_request()
                return
            if path == "/api/runtime_progress":
                self._send_json(server.runtime_progress())
                server.stats.count_request()
                return
            if path in ("/", "/index.html", "/ui", "/docs"):
                self._send_html(_WEBUI_HTML)
                server.stats.count_request()
                return
            self._send_error_json(f"未找到路径：{path}", "invalid_request_error", 404, "not_found")
            server.stats.count_request(failed=True)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_models(self) -> None:
            data: list[dict[str, Any]] = []
            with server._lock:
                model_ids = list(server._loaded.keys())
            for b in server.backends:
                try:
                    models = b.list_models()
                except Exception:  # noqa: BLE001
                    models = []
                for m in models:
                    if m.get("id") not in model_ids:
                        data.append({
                            "id": m.get("id", ""),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        })
            for mid in model_ids:
                data.append({
                    "id": mid,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                })
            # 去重
            seen: set[str] = set()
            unique: list[dict[str, Any]] = []
            for item in data:
                if item["id"] and item["id"] not in seen:
                    seen.add(item["id"])
                    unique.append(item)
            self._send_json({"object": "list", "data": unique})
            server.stats.count_request()

        # ---------------- POST ----------------

        def do_POST(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if not self._check_token():
                server.stats.count_request(failed=True)
                self._send_error_json("访问令牌校验失败", "invalid_request_error", 401, "invalid_token")
                return
            try:
                payload = self._read_body()
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是 JSON 对象")
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"请求体 JSON 解析失败：{exc}", "invalid_request_error", 400, "invalid_json")
                return

            if path == "/v1/chat/completions":
                self._handle_chat(payload)
            elif path == "/v1/completions":
                self._handle_completion(payload)
            elif path == "/api/embeddings" or path == "/v1/embeddings":
                self._handle_embeddings(payload)
            elif path == "/api/download":
                mid = str(payload.get("model_id") or payload.get("id") or "").strip()
                quant = str(payload.get("quant") or "").strip()
                if not mid:
                    server.stats.count_request(failed=True)
                    self._send_error_json("缺少 model_id", "invalid_request_error", 400, "missing_model_id")
                    return
                res = server.start_download(mid, quant)
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path == "/api/config":
                res = server.update_config(payload.get("config") or {})
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path == "/api/load_model":
                mid = str(payload.get("model_id") or payload.get("id") or "").strip()
                if not mid:
                    server.stats.count_request(failed=True)
                    self._send_error_json("缺少 model_id", "invalid_request_error", 400, "missing_model_id")
                    return
                res = server.start_model(mid)
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path == "/api/unload_model":
                mid = str(payload.get("model_id") or payload.get("id") or "").strip()
                res = server.stop_model(mid)
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path == "/api/install_runtime":
                res = server.install_runtime()
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path == "/api/library_remove":
                mid = str(payload.get("model_id") or payload.get("id") or "").strip()
                if not mid:
                    server.stats.count_request(failed=True)
                    self._send_error_json("缺少 model_id", "invalid_request_error", 400, "missing_model_id")
                    return
                keep = bool(payload.get("keep_file"))
                res = server.library_remove(mid, delete_file=not keep)
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path.startswith("/api/") and len(path) > 5:
                self._handle_alias(path[len("/api/"):], payload)
            else:
                server.stats.count_request(failed=True)
                self._send_error_json(f"未找到路径：{path}", "invalid_request_error", 404, "not_found")

        # ---------------- 各端点实现 ----------------

        def _pick_model(self, payload: dict[str, Any], alias: str = "") -> str | None:
            """从请求体或别名解析目标模型。"""
            explicit = str(payload.get("model") or "").strip()
            if explicit:
                return explicit
            if alias:
                resolved = server.resolve_alias(alias)
                if resolved:
                    return resolved
            return None

        def _forward_or_error(self, model_id: str, upstream_path: str,
                             payload: dict[str, Any]
                             ) -> tuple[int, dict[str, Any]] | None:
            """转发并处理通用错误；返回 None 表示已写回响应。

            llama-server 在模型加载完成前，``/v1/models`` 已经返回 200，
            但推理接口会回 503 ``Loading model``。这里对这种情况做退避重试，
            用户不必自己掐时间再点一次。
            """
            status, data = server.forward_for_model(model_id, upstream_path, payload)
            tries = 0
            while status == 503 and _is_loading_model(data) and tries < 90:
                tries += 1
                time.sleep(1.0 if tries < 30 else 2.0)
                status, data = server.forward_for_model(model_id, upstream_path, payload)

            if status == 0:
                err = (data or {}).get("error") or {}
                server.stats.count_request(failed=True)
                self._send_error_json(
                    str(err.get("message") or "推理后端不可用"),
                    "api_error", 502, "backend_unavailable",
                )
                return None
            if status >= 400:
                err = (data or {}).get("error") or {}
                server.stats.count_request(failed=True)
                self._send_error_json(
                    str(err.get("message") or f"上游返回错误状态 {status}"),
                    str(err.get("type") or "api_error"), status, err.get("code"),
                )
                return None
            return status, data

        def _handle_chat(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            # 流式
            if payload.get("stream"):
                self._handle_chat_stream(payload, model_id)
                return
            status_data = self._forward_or_error(model_id, "/v1/chat/completions", payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_chat(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

        def _handle_chat_stream(self, payload: dict[str, Any], model_id: str) -> None:
            base = server.backend_url_for(model_id)
            if not base:
                server.stats.count_request(failed=True)
                self._send_error_json(f"模型未加载：{model_id}", "api_error", 502, "backend_unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors_headers()
            self.end_headers()
            try:
                for line in proxy.forward_stream(base, "/v1/chat/completions", payload):
                    if line:
                        self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.write(b"\n")
                    self.wfile.flush()
            except (RuntimeError, OSError) as exc:
                try:
                    self.wfile.write(f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    pass
                server.stats.count_request(failed=True)
                return
            server.stats.count_request()

        def _handle_completion(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            status_data = self._forward_or_error(model_id, "/v1/completions", payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_completion(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

        def _handle_embeddings(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            if not server.cfg.get("api.expose_embedding", True):
                server.stats.count_request(failed=True)
                self._send_error_json("嵌入接口未开放（api.expose_embedding=False）", "invalid_request_error", 404, "embedding_disabled")
                return
            status_data = self._forward_or_error(model_id, "/v1/embeddings", payload)
            if status_data is None:
                return
            _, data = status_data
            out = _to_openai_embedding(data, model_id, payload)
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), 0)
            self._send_json(out)
            server.stats.count_request()

        def _handle_alias(self, alias: str, payload: dict[str, Any]) -> None:
            """POST /api/{alias} —— 按模型系列别名转发。"""
            model_id = self._pick_model(payload, alias=alias)
            if not model_id:
                server.stats.count_request(failed=True)
                amap = server.aliases()
                self._send_error_json(
                    f"别名「{alias}」没有对应的已加载模型（已注册别名：{', '.join(sorted(amap)) or '无'}）",
                    "invalid_request_error", 404, "alias_not_found",
                )
                return
            upstream_path = "/v1/chat/completions"
            status_data = self._forward_or_error(model_id, upstream_path, payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_chat(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

    return Handler


# ---------------------------------------------------------------- 响应归一化


def _is_loading_model(data: Any) -> bool:
    """判断上游 503 是不是「模型还在加载中」。

    llama-server 加载权重期间会返回 ``{"error": {"message": "Loading model"}}``，
    这种情况等一会儿就好，不是真的故障。
    """
    if not isinstance(data, dict):
        return False
    err = data.get("error")
    if isinstance(err, dict):
        msg = str(err.get("message") or "").lower()
        code = str(err.get("code") or "").lower()
    elif isinstance(err, str):
        msg, code = err.lower(), ""
    else:
        msg, code = "", ""
    if "loading" in msg or "loading" in code:
        return True
    # 有些版本放在顶层 message 里
    top = str(data.get("message") or "").lower()
    return "loading model" in top


def _to_openai_chat(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """把后端响应转成 OpenAI chat.completion 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    # 已经是 OpenAI 格式：直接补齐字段
    choices = data.get("choices")
    message = None
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
    if message is not None:
        return _fill_chat_fields(data, model_id)

    # Ollama /api/generate 风格: {"response": "..."}
    if "response" in data and "choices" not in data:
        return {
            "id": f"chatcmpl-{_now_ms()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": str(data.get("response") or "")},
                "finish_reason": str(data.get("done_reason") or "stop"),
            }],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }

    # Ollama /api/chat 风格: {"message": {"role","content"}}
    msg = data.get("message")
    if isinstance(msg, dict):
        return {
            "id": f"chatcmpl-{_now_ms()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{
                "index": 0,
                "message": {"role": msg.get("role", "assistant"),
                            "content": str(msg.get("content") or "")},
                "finish_reason": "stop" if data.get("done", True) else "length",
            }],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }

    raise KeyError("上游响应中没有 choices / response / message 字段")


def _fill_chat_fields(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """补齐 OpenAI 格式缺的字段。"""
    out = dict(data)
    out.setdefault("id", f"chatcmpl-{_now_ms()}")
    out["object"] = "chat.completion"
    out.setdefault("created", int(time.time()))
    out.setdefault("model", model_id)
    choices = out.get("choices") or [{}]
    for ch in choices:
        if isinstance(ch, dict):
            ch.setdefault("index", 0)
            ch.setdefault("finish_reason", "stop")
            msg = ch.get("message")
            if not isinstance(msg, dict):
                ch["message"] = {"role": "assistant", "content": ch.get("text") or ""}
            else:
                msg.setdefault("role", "assistant")
                msg.setdefault("content", "")
    out["choices"] = choices
    out.setdefault("usage", _usage(data))
    return out


def _to_openai_completion(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """把后端响应转成 OpenAI text_completion 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    if data.get("choices") and isinstance(data["choices"][0], dict):
        out = dict(data)
        out.setdefault("id", f"cmpl-{_now_ms()}")
        out["object"] = "text_completion"
        out.setdefault("created", int(time.time()))
        out.setdefault("model", model_id)
        for ch in out["choices"]:
            if isinstance(ch, dict):
                ch.setdefault("index", 0)
                ch.setdefault("finish_reason", "stop")
                ch.setdefault("text", ch.get("text") or "")
        out.setdefault("usage", _usage(data))
        return out
    # Ollama generate 风格
    if "response" in data:
        return {
            "id": f"cmpl-{_now_ms()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{"index": 0, "text": str(data.get("response") or ""),
                         "finish_reason": "stop"}],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }
    raise KeyError("上游响应中没有 choices / response 字段")


def _to_openai_embedding(data: dict[str, Any], model_id: str,
                         payload: dict[str, Any]) -> dict[str, Any]:
    """把后端响应转成 OpenAI embedding 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    if data.get("data"):
        out = dict(data)
        out.setdefault("object", "list")
        out.setdefault("model", model_id)
        out.setdefault("usage", _usage(data))
        return out
    # Ollama /api/embeddings 风格: {"embedding": [...]}
    if "embedding" in data:
        emb = data.get("embedding") or []
        return {
            "object": "list",
            "model": str(data.get("model") or model_id),
            "data": [{"object": "embedding", "index": 0, "embedding": emb}],
            "usage": _usage(data),
        }
    raise KeyError("上游响应中没有 embedding / data 字段")


def _usage(data: dict[str, Any], prompt_eval: Any = None, eval_count: Any = None) -> dict[str, int]:
    """从各种字段里凑出 usage。"""
    u = data.get("usage") or {}
    if isinstance(u, dict):
        pt = u.get("prompt_tokens")
        ct = u.get("completion_tokens")
        if pt is not None or ct is not None:
            return {
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "total_tokens": int(u.get("total_tokens") or (int(pt or 0) + int(ct or 0))),
            }
    pe = prompt_eval if prompt_eval is not None else data.get("prompt_eval_count")
    ec = eval_count if eval_count is not None else data.get("eval_count")
    if pe is not None or ec is not None:
        p, c = int(pe or 0), int(ec or 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------- 组装


def build_server(cfg: Config | None = None, host: str | None = None,
                 port: int | None = None, token: str | None = None,
                 backends: list[Backend] | None = None) -> tuple[ThreadingHTTPServer, ApiServer]:
    """创建 HTTP 服务器与对应的 ApiServer。"""
    cfg = cfg or Config.load()
    api = ApiServer(
        cfg,
        backends=backends,
        host=host or str(cfg.get("app.host", "127.0.0.1")),
        port=int(port if port is not None else cfg.get("app.port", 8080)),
        token=token if token is not None else "",
    )
    handler = create_handler(api)
    httpd = ThreadingHTTPServer((api.host, api.port), handler)
    httpd.daemon_threads = True
    api.port = httpd.server_address[1]
    return httpd, api


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``python -m imdeploy.api_server --host --port --token``。"""
    parser = argparse.ArgumentParser(
        prog="feiyu-api-server",
        description="imdeploy：本地大模型部署器 OpenAI 兼容 API 服务",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    parser.add_argument("--token", default="", help="访问口令（留空=不校验）")
    parser.add_argument("--config", default="data/config.yaml", help="配置文件路径")
    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except Exception:  # noqa: BLE001 - 配置加载失败时用默认配置
        cfg = Config.default()

    httpd, api = build_server(cfg, host=args.host, port=args.port, token=args.token)
    print(f"imdeploy API 服务已启动：http://{api.host}:{api.port}")
    print(f"  后端：{', '.join(b.name for b in api.backends) or '未检测到'}")
    print(f"  口令：{'已启用' if api.token else '未启用（免鉴权）'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
