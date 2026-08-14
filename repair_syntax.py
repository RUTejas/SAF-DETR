import os
import ast

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    modified = False
    
    # If the file has escaped double quotes
    if r'"' in content:
        content = content.replace(r'"', '"')
        modified = True
    
    # If the file has literal \n from bad string serialization
    if '\\n' in content and '\n' not in content:
        content = content.replace('\\n', '\n').replace('\\t', '\t')
        modified = True
        
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")

    # Check syntax
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"OK: {filepath}")
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {filepath}: {e}")

def main():
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'venv' in root or '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                fix_file(os.path.join(root, f))

if __name__ == '__main__':
    main()
