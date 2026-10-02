import os
import re

def fix_file(path, replacements):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

fix_file(r'dataset\formatting\sft.py', [('import json\n', '')])
fix_file(r'deploy\space\app.py', [('f"### Generation failed"', '"### Generation failed"')])
fix_file(r'scripts\ablation_report.py', [('for l\n', 'for line\n'), ('for l in', 'for line in'), ('loads(l)', 'loads(line)'), ('if l.', 'if line.')])
fix_file(r'scripts\build_showcase.py', [('import shutil\n', ''), ('for l in', 'for line in'), ('loads(l)', 'loads(line)'), ('if l.', 'if line.')])
fix_file(r'scripts\export_eval_pack.py', [('encode()); combined', 'encode())\n        combined')])
fix_file(r'scripts\export_repair_pack.py', [('for l in', 'for line in'), ('loads(l)', 'loads(line)'), ('if l.', 'if line.')])
fix_file(r'scripts\failure_analysis.py', [('key = f["bucket"]', '_key = f["bucket"]')])
