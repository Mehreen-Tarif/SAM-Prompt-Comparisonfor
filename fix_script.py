with open('SAM_FINAL_EXPERIMENT.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('"""\n', 'r"""\n', 1)
with open('SAM_FINAL_EXPERIMENT.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed successfully!')
