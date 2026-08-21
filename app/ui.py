"""极简问答页面：GET / 直接可用，零构建零依赖。"""

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>repo-rag · 本地知识库问答</title>
<style>
:root{--bg:#F2F3F5;--card:#fff;--ink:#16191E;--muted:#6A7280;--line:#D3D7DD;--go:#0E7C8B;--go-soft:#D9EDF0;--warn:#8A5A00;--warn-soft:#F6EBD4}
@media (prefers-color-scheme:dark){:root{--bg:#14171B;--card:#1C2026;--ink:#E6E9ED;--muted:#98A1AE;--line:#2E343C;--go:#4FBACB;--go-soft:#123034;--warn:#E0AE55;--warn-soft:#332811}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:40px 18px 80px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
.ask{display:flex;gap:8px}
input{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);font-size:15px}
button{padding:11px 22px;border:0;border-radius:6px;background:var(--go);color:#fff;font-size:15px;cursor:pointer}
button:disabled{opacity:.5}
.hint{color:var(--muted);font-size:12px;margin-top:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:18px;margin-top:20px}
.answer{white-space:pre-wrap}
.badge{display:inline-block;font-size:11px;padding:2px 9px;border-radius:3px;margin-bottom:10px;font-family:ui-monospace,monospace}
.ok{background:var(--go-soft);color:var(--go)}.ref{background:var(--warn-soft);color:var(--warn)}
.src{border-top:1px dashed var(--line);margin-top:14px;padding-top:10px}
.src summary{cursor:pointer;font-size:13px;color:var(--muted)}
.src pre{white-space:pre-wrap;font-size:12.5px;color:var(--muted);background:var(--bg);padding:10px;border-radius:6px;max-height:200px;overflow:auto}
.loading{color:var(--muted);font-size:14px}
.examples{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
.examples span{font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:20px;color:var(--muted);cursor:pointer;background:var(--card)}
</style>
</head>
<body><div class="wrap">
<h1>repo-rag</h1>
<div class="sub">问答范围：jnMetaCode 的 10 个开源仓库文档 · bge-m3 检索 + 本地 claude 生成 · 数据不出机</div>
<div class="ask"><input id="q" placeholder="问点关于你开源项目的问题…" autofocus>
<button id="go" onclick="ask()">提问</button></div>
<div class="examples">
<span onclick="fill(this)">agency-orchestrator 是做什么的？</span>
<span onclick="fill(this)">superpowers-zh 有多少个 skills？</span>
<span onclick="fill(this)">OpenShorts 的质量门禁查什么？</span>
<span onclick="fill(this)">红烧肉怎么做？（测拒答）</span>
</div>
<div class="hint">生成端是本地 claude CLI，约 10–30 秒出结果</div>
<div id="out"></div>
</div>
<script>
function fill(el){document.getElementById('q').value=el.textContent;ask()}
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const btn=document.getElementById('go'), out=document.getElementById('out');
  btn.disabled=true;
  out.innerHTML='<div class="card loading">检索中 → 生成中…（本地推理，稍等）</div>';
  try{
    const r=await fetch('/v1/query',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q})});
    const d=await r.json();
    let h='<div class="card">';
    h+=d.refused?'<span class="badge ref">检索闸门拒答 · top_score '+d.top_score+'</span>'
               :'<span class="badge ok">top_score '+d.top_score+' · '+d.sources.length+' 条引用</span>';
    h+='<div class="answer"></div>';
    if(d.sources.length){
      h+='<div class="src"><details><summary>📎 引用原文（'+d.sources.length+'）</summary>';
      for(const s of d.sources) h+='<p style="font-size:12px;color:var(--muted);margin:8px 0 2px">['+s.ref+'] '+s.source+' #'+s.chunk_index+' · score '+s.score+'</p><pre></pre>';
      h+='</details></div>';
    }
    h+='</div>';
    out.innerHTML=h;
    out.querySelector('.answer').textContent=d.answer;
    const pres=out.querySelectorAll('pre');
    d.sources.forEach((s,i)=>{if(pres[i])pres[i].textContent=s.content});
  }catch(e){out.innerHTML='<div class="card">请求失败：'+e+'</div>'}
  btn.disabled=false;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script>
</body></html>"""
