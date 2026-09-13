import re
import sys
from pathlib import Path

import pymupdf

PDF = Path(sys.argv[1])
CHAPTER = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = Path('data') / f'chapter_{CHAPTER:02d}.txt'

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII']
HEADER = re.compile(rf'^CHAPTER\s+({"|".join(ROMAN)})\s*$', re.M)

doc = pymupdf.open(PDF)
pages = [p.get_text() for p in doc]
full = '\n'.join(pages)

starts = [(m.group(1), m.start()) for m in HEADER.finditer(full)]
start = next(pos for num, pos in starts if num == ROMAN[CHAPTER - 1])
end = next((pos for num, pos in starts if num == ROMAN[CHAPTER]), len(full)) if CHAPTER < len(ROMAN) else len(full)
body = full[start:end]

body = HEADER.sub('', body, count=1)
lines = [ln.strip() for ln in body.split('\n')]
first_blank = next(i for i, ln in enumerate(lines) if ln == '' and any(lines[:i]))
head = [ln for ln in lines[:first_blank] if ln]
attributions = [i for i, ln in enumerate(head) if ln.startswith('- ')]
title = head[0]
body_lines = head[attributions[-1] + 1:] + lines[first_blank:] if attributions else head[1:] + lines[first_blank:]

ENDS_SENTENCE = re.compile(r'[.!?\u2019\u201d"\u2026)\]]$')
paragraphs, current = [], []
for ln in body_lines:
    if ln == '':
        if current:
            paragraphs.append(' '.join(current))
            current = []
    else:
        current.append(ln)
if current:
    paragraphs.append(' '.join(current))

merged = []
for para in paragraphs:
    if merged and not ENDS_SENTENCE.search(merged[-1]):
        merged[-1] = merged[-1] + ' ' + para
    else:
        merged.append(para)
paragraphs = merged
footnotes = [para for para in paragraphs if para.startswith('*')]
paragraphs = [para.replace('*', '') for para in paragraphs if not para.startswith('*')]

OUT.parent.mkdir(exist_ok=True)
OUT.write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
words = sum(len(p.split()) for p in paragraphs)
print(f'chapter {ROMAN[CHAPTER - 1]} "{title}": {len(paragraphs)} paragraphs, {words} words, {len(attributions)} epigraphs and {len(footnotes)} footnotes dropped -> {OUT}')
