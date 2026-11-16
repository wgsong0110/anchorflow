#!/bin/bash
# 절 CM192: final_fix 인스턴스 상태 한 줄씩 (로컬 조회용)
vastai show instances-v1 --raw 2>&1 | python3 -c "
import json,sys,time
t=sys.stdin.read(); d=json.loads(t[t.find('['):]) if t.strip()[0]!='{' else json.loads(t[t.find('{'):])
if isinstance(d,dict): d=d.get('instances',d)
for i in d:
  if 'final_fix' in (i.get('label') or ''): print(i['id'],i['label'],i['actual_status'],i['num_gpus'],i['gpu_name'],round(i['dph_total'],3),i.get('public_ipaddr'),json.dumps([p['HostPort'] for p in ((i.get('ports') or {}).get('22/tcp') or [])][:1]),(i.get('status_msg') or '')[:50].replace('\n',' '),'age',round((time.time()-i['start_date'])/60) if i.get('start_date') else None,'util',i.get('gpu_util'))
"
