import json, pathlib, time
p = pathlib.Path(r'E:\ai知识库\nlp大赛\中间结果\enhanced\translation_progress.json')
d1 = json.loads(p.read_text(encoding='utf-8'))
b1 = len(d1['completed'].get('bg', {}))
f1 = len(d1['failed'].get('bg', []))
print(f'BG now: completed={b1}, failed={f1}, pending={834-b1-f1}')
print('Waiting 7s...')
time.sleep(7)
d2 = json.loads(p.read_text(encoding='utf-8'))
b2 = len(d2['completed'].get('bg', {}))
if b2 == b1:
    print('No change - processes stopped OK')
else:
    print(f'Still changing: {b1} -> {b2} - process still running!')
