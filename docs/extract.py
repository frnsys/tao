"""
Splits the raw Tao Te Ching translations into individual parts.
"""

import os
import shutil
from glob import glob

if os.path.exists('output'):
    shutil.rmtree('output')
os.makedirs('output')

for file in glob('sources/*.txt'):
    # should be 81 chapters
    chapters = []
    chapter = []
    ch_num = 0

    version = os.path.basename(file).split('.')[0]

    for line in open(file, 'r').readlines():
        line = line.strip()

        # text for links to go to top of page
        if line == 'up':
            continue

        try:
            # new chapter
            ch_num = int(line.strip('.'))
            if chapter:
                chapters.append('\n'.join(chapter).strip())
            chapter = []
        except ValueError:
            chapter.append(line)

    # add the last chapter
    chapters.append('\n'.join(chapter))

    try:
        assert(len(chapters) == 81)
        for chapter in chapters:
            assert(len(chapter) > 0)
    except AssertionError:
        print('failed on', version)
        print('n chapters', len(chapters))
        raise


    output = os.path.join('output', version)
    os.makedirs(output)
    for i, chapter in enumerate(chapters):
        ch_output = os.path.join(output, '{}.txt'.format(i+1))
        with open(ch_output, 'w') as f:
            f.write(chapter)