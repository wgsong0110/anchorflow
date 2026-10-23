#!/usr/bin/env python3
"""출력 전처리 세 판 + 기준판 러닝 커브 인터랙티브 페이지 (사용자 10-23, ipcurve/ip_curve.html 형식).

  python3 exe/precond_curve.py --rec 이름=records.json ... --frames <프레임 jpg 디렉토리 이름(페이지 기준 상대)> --out precond_curve.html
스텝 슬라이더로 프레임 t 를 고르면 네 판의 증분 포텐셜 곡선(Adam 반복)과 각 판의 E*(같은 판 상태의 출력만 최적화 해, 점선)를
같은 로그 그래프에 겹치고, 그 스텝 시작·끝 영상 프레임(정답 | 기준판 | 세 판)을 보여준다.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--rec", action="append", required=True, help="표시이름=records.json (첫 번째가 기준판)")
ap.add_argument("--frames", required=True, help="프레임 jpg 디렉토리 (페이지와 같은 디렉토리 안, 001.jpg = 영상 프레임 0)")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="증분 포텐셜 한 프레임씩 · 출력 전처리 세 판 vs 기준판 러닝 커브 (Adam 2000 회)")
a = ap.parse_args()

D, names = {}, []
for kv in a.rec:
    nm, p = kv.split("=", 1)
    r = json.load(open(p))
    names.append(nm)
    D[nm] = [dict(t=q["t"], c=q["obj_curve"], es=q.get("E_star"), ratio=q.get("ratio_E_Estar"), pe=100 * q["pos_err_model"],
                  pre=(q.get("precond") or {}).get("sec")) for q in r]
T = min(len(v) for v in D.values())
cols = ["#222222", "#1f77b4", "#2ca02c", "#d62728"]
base = "/api/files/raw?path=" + os.path.dirname(os.path.abspath(a.out))
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{a.title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>body{{font-family:sans-serif;margin:12px}} .row{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}} img{{width:900px;border:1px solid #ccc}} .cap{{font-size:13px;color:#444}} #info{{font-size:14px;margin:6px 0;line-height:1.5}}</style></head><body>
<h3>{a.title}</h3>
<div>L000 &nbsp; 스텝 t <input type="range" id="sl" min="0" max="{T - 1}" value="0" style="width:600px"> <b id="tv">0</b>
 <button onclick="mv(-1)">◀</button><button onclick="mv(1)">▶</button>
 &nbsp; <label><input type="checkbox" id="lg" checked> 세로축 로그</label></div>
<div id="info"></div>
<div class="row"><div style="width:720px;height:480px"><canvas id="cv"></canvas></div>
<div><div class="cap">스텝 시작 (영상 프레임 <span id="f0"></span>) · 정답 | 기준판 | 대각 | 다중 해상도 | L-BFGS</div><img id="i0">
<div class="cap">스텝 끝 (영상 프레임 <span id="f1"></span>)</div><img id="i1"></div></div>
<script>
const N={json.dumps(names, ensure_ascii=False)}; const C={json.dumps(cols[:len(names)])};
const D={json.dumps(D, ensure_ascii=False)}; const B="{base}/{a.frames}";
const ds=[]; N.forEach((n,i)=>{{ds.push({{label:n+' E',data:[],borderColor:C[i],pointRadius:0,borderWidth:1.5}});
 ds.push({{label:n+' E*',data:[],borderColor:C[i],borderDash:[6,4],pointRadius:0,borderWidth:1}});}});
const ch=new Chart(document.getElementById('cv'),{{type:'line',data:{{datasets:ds}},
 options:{{animation:false,parsing:false,scales:{{x:{{type:'linear',title:{{display:true,text:'Adam 반복'}}}},y:{{type:'logarithmic',title:{{display:true,text:'증분 포텐셜 (정규화)'}}}}}}}}}});
function pad(n){{return String(n).padStart(3,'0')}}
function draw(){{const t=+document.getElementById('sl').value; document.getElementById('tv').textContent=t;
 ch.options.scales.y.type=document.getElementById('lg').checked?'logarithmic':'linear';
 let s=`스텝 t=${{t}} (영상 프레임 ${{t}} → ${{t+1}})<br>`;
 N.forEach((n,i)=>{{const q=D[n][t]; const xm=q.c[q.c.length-1][0];
  ch.data.datasets[2*i].data=q.c.map(p=>({{x:p[0],y:p[1]}}));
  ch.data.datasets[2*i+1].data=q.es?[{{x:0,y:q.es}},{{x:xm,y:q.es}}]:[];
  s+=`<span style="color:${{C[i]}}">■</span> ${{n}}: 시작 E ${{q.c[0][1].toExponential(3)}} → 끝 E ${{q.c[q.c.length-1][1].toExponential(3)}} · E* ${{q.es?q.es.toExponential(3):'-'}} · E/E* ${{q.ratio?q.ratio.toFixed(3):'-'}} · 위치 오차 ${{q.pe.toFixed(2)}}%`+(q.pre!=null?` · 전처리 구성 ${{q.pre.toFixed(1)}} 초`:'')+'<br>';}});
 ch.update(); document.getElementById('info').innerHTML=s;
 document.getElementById('f0').textContent=t; document.getElementById('f1').textContent=t+1;
 document.getElementById('i0').src=`${{B}}/${{pad(t+1)}}.jpg`; document.getElementById('i1').src=`${{B}}/${{pad(t+2)}}.jpg`;}}
function mv(d){{const s=document.getElementById('sl');s.value=Math.min({T - 1},Math.max(0,+s.value+d));draw();}}
document.getElementById('sl').oninput=draw; document.getElementById('lg').onchange=draw; draw();
</script></body></html>"""
open(a.out, "w").write(html)
print(f"[곡선 페이지] {a.out} 프레임 {T} 판 {names}")
