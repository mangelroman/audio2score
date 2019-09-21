import re
import numpy as np

from pathlib import Path
from itertools import cycle

classic_tempos = {
    "grave" : 32,
    "largoassai" : 40,
    "largo" : 50,
    "pocolargo" : 60,
    "adagio" : 71,
    "pocoadagio" : 76,
    "andante" : 92,
    "andantino" : 100,
    "menuetto" : 112,
    "moderato" : 114,
    "pocoallegretto" : 116,
    "allegretto" : 118,
    "allegromoderato" : 120,
    "pocoallegro" : 124,
    "allegro" : 130,
    "moltoallegro" : 134,
    "allegroassai" : 138,
    "vivace" : 140,
    "vivaceassai" : 150,
    "allegrovivace" : 160,
    "allegrovivaceassai" : 170,
    "pocopresto" : 180,
    "presto" : 186,
    "prestoassai" : 200,
}

class Labels(object): # 38 symbols
    def __init__(self):
        self.labels = [
            "+", # ctc blank
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "C", "D", "E", "F", "G", "A", "B", "c", "d", "e", "f", "g", "a", "b",
            "r", "#", "-", "=", ".", "[", "_", "]", ";", "\t", "\n",
            "<", ">", # seq2seq <sos> and <eos> delimiters
        ]
        self.labels_map     = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c) for (i, c) in enumerate(self.labels)])

    def ctclen(self, tokens):
        count = len(tokens)
        count += sum([tokens[i - 1] == tokens[i] for i in range(1, count)])
        return count

    def encode(self, chars):
        tokens = []
        for c in chars:
            tokens.append(self.labels_map[c])
        return tokens

    def decode(self, tokens):
        return list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))

class LabelsMulti(object): # 148 symbols
    def __init__(self, chorales=False):

        self.labels = [
            "+", # ctc blank
            "1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.","3","6","12","24","48","96",
            "BBB#","CC","CC#","DD-","DD","DD#","EE-","EE","EE#","FF-","FF","FF#","GG-","GG","GG#","AA-","AA","AA#","BB-","BB","BB#",
            "C-","C","C#","D-","D","D#","E-","E","E#","F-","F","F#","G-","G","G#","A-","A","A#","B-","B","B#",
            "c-","c","c#","d-","d","d#","e-","e","e#","f-","f","f#","g-","g","g#","a-","a","a#","b-","b","b#",
            "cc-","cc","cc#","dd-","dd","dd#","ee-","ee","ee#","ff-","ff","ff#","gg-","gg","gg#","aa-","aa","aa#",
            "bb-","bb","bb#",
            "ccc-","ccc","ccc#","ddd-","ddd","ddd#","eee-","eee","eee#","fff-","fff","fff#","ggg-","ggg","ggg#","aaa-","aaa","aaa#",
            "bbb-","bbb","bbb#",
            "cccc-","cccc","cccc#","dddd-","dddd","dddd#","eeee-","eeee","eeee#","ffff-","ffff",
            "r", "=", ".", "[", "_", "]", ";", "\t", "\n", 
            "<sos>", "<eos>", # seq2seq delimiters
        ]
        self.labels_map     = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c) for (i, c) in enumerate(self.labels)])

    def ctclen(self, tokens):
        return len(tokens)

    def encode(self, chars):
        tokens = []
        for line in chars.splitlines():
            items = line.split('\t')
            for item in items:
                if len(item) == 1:
                    tokens.append(self.labels_map[item])
                else:
                    matchobj = re.fullmatch(r'(\[?)(\d+\.*)([a-gA-Gr]{1,4}[\-#]*)(;?)([\]_]?)', item)
                    if not matchobj:
                        raise Exception(f'Item {item} in {line} does not match')
                    for m in [matchobj[1], matchobj[2], matchobj[3], matchobj[4], matchobj[5]]:
                        if m:
                            tokens.append(self.labels_map[m])
                tokens.append(self.labels_map['\t'])
            tokens[-1] = self.labels_map['\n']
        tokens.pop(-1)
        return tokens

    def decode(self, tokens):
        return list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))

class LabelsMulti2(object): # 9147 symbols
    def __init__(self):
        durations = ["1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.","3","6","12","24","48","96"]
        notes = ["BBB#","CC","CC#","DD-","DD","DD#","EE-","EE","EE#","FF-","FF","FF#","GG-","GG","GG#","AA-","AA","AA#","BB-","BB","BB#",
            "C-","C","C#","D-","D","D#","E-","E","E#","F-","F","F#","G-","G","G#","A-","A","A#","B-","B","B#",
            "c-","c","c#","d-","d","d#","e-","e","e#","f-","f","f#","g-","g","g#","a-","a","a#","b-","b","b#",
            "cc-","cc","cc#","dd-","dd","dd#","ee-","ee","ee#","ff-","ff","ff#","gg-","gg","gg#","aa-","aa","aa#",
            "bb-","bb","bb#",
            "ccc-","ccc","ccc#","ddd-","ddd","ddd#","eee-","eee","eee#","fff-","fff","fff#","ggg-","ggg","ggg#","aaa-","aaa","aaa#",
            "bbb-","bbb","bbb#",
            "cccc-","cccc","cccc#","dddd-","dddd","dddd#","eeee-","eeee","eeee#"]
        ties = ["[", "_", "]"]
        self.labels = ['+'] # ctc blank
        for d in durations:
            for n in notes:
                self.labels.append(d + n)
                self.labels.append('[' + d + n)
                self.labels.append(d + n + '_')
                self.labels.append(d + n + ']')
            self.labels.append(d + 'r')
        self.labels.extend([
            "=", ".", "\t", "\n",
            "<sos>", "<eos>", # seq2seq delimiters
        ])
        self.labels_map     = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c) for (i, c) in enumerate(self.labels)])

    def ctclen(self, tokens):
        return len(tokens)

    def encode(self, chars):
        tokens = []
        for line in chars.splitlines():
            items = line.split('\t')
            for item in items:
                tokens.append(self.labels_map[item])
                tokens.append(self.labels_map['\t'])
            tokens[-1] = self.labels_map['\n']
        tokens.pop(-1)
        return tokens

    def decode(self, tokens):
        return list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))

class Humdrum(object):
    def __init__(self, path=None, data=None):
        if path:
            data = path.read_text(encoding='iso-8859-1')
        lines = data.splitlines()
        body_begin = 0
        body_end = 0
        for i, line in enumerate(lines):
            if line.startswith('**'):
                body_begin = i + 1
            if line.startswith('*-'):
                body_end = i
                break
        self.header = lines[:body_begin]
        self.footer = lines[body_end:]
        self.body = lines[body_begin:body_end]
        self.spine_types = self.header[-1].split('\t')

    def save(self, path):
        return path.write_text(self.dump(), encoding='iso-8859-1')

    def dump(self):
        return '\n'.join(self.header + self.body + self.footer)

class SpineInfo(object):
    def __init__(self, spine_types):
        self.spines = []
        for stype in spine_types:
            self.spines.append({'type' : stype,
                                'instrument' : '*',
                                'clef' : '*',
                                'keysig' : '*',
                                'tonality' : '*',
                                'timesig' : '*',
                                'metronome' : '*',
                               })

    def update(self, line):
        for i, item in enumerate(line.split('\t')):
            if item.startswith('*k['):
                self.spines[i]['keysig'] = item
            elif item.startswith('*clef'):
                self.spines[i]['clef'] = item
            elif item.startswith('*I'):
                self.spines[i]['instrument'] = item
            elif item.startswith('*MM'):
                self.spines[i]['metronome'] = item
            elif item.startswith('*M'):
                self.spines[i]['timesig'] = item
            elif item.startswith('*CT'):
                item = f'*MM{classic_tempos[item[3:]]}'
                self.spines[i]['metronome'] = item
            elif item.endswith(':'):
                self.spines[i]['tonality'] = item

    def override_instruments(self, instruments):
        pool = cycle(instruments)
        inst = instruments[0]
        for i in range(len(self.spines)):
            if self.spines[i]['type'] == '**kern':
                inst = next(pool)
            self.spines[i]['instrument'] = f'*I{inst}'

    def dump(self):
        header = []
        for v in ['type', 'instrument', 'clef', 'keysig', 'tonality', 'timesig', 'metronome']:
            header.append('\t'.join([x[v] for x in self.spines]))
        footer = ['\t'.join(['*-' for x in self.spines])]
        return header, footer

    def clone(self):
        spine_types = [s['type'] for s in self.spines]
        spineinfo = SpineInfo(spine_types)
        spineinfo.spines = self.spines.copy()
        return spineinfo

class Kern(Humdrum):
    def __init__(self, path=None, data=None):
        super(Kern, self).__init__(path, data)

        self.spines = SpineInfo(self.spine_types)
        self.first_line = 0 
        for i, line in enumerate(self.body):
            if not line.startswith('*'):
                self.first_line = i
                break
            self.spines.update(line)

    def clean(self, remove_pauses=True):
        spine_types = self.spine_types.copy()
        newbody = []

        for line in self.body[self.first_line:]:
            if re.search(r'\*[+x^v]', line):
                i = 0
                remove_spine = False
                for item in line.split('\t'):
                    if item.startswith(('*+', '*x')):
                        print('Unsupported variable spines')
                        return False
                    elif item == '*^':
                        spine_types.insert(i, '**kern')
                        i += 1
                        spine_types[i] = '**split'
                    elif item == '*v':
                        if remove_spine:
                            spine_types.pop(i)
                            i -= 1
                            remove_spine = False
                        else:
                            remove_spine = True
                    i += 1
                continue

            if line.startswith('!'):
                newbody.append(line)
                continue

            # Remove unwanted symbols
            newline = []
            note_found = False
            grace_note_found = False
            for i, item in enumerate(line.split('\t')):
                if spine_types[i] == '**split':
                    # Remove spline split
                    continue

                if spine_types[i] == '**kern' and not item.startswith(('*', '=')):
                    item = item.split()[0] # Take the first note of the chord
                    item = re.sub(r'[pTtMmWwS$O:]', r'', item) # Remove ornaments
                    if remove_pauses:
                        item = re.sub(r';', r'', item) # Remove pauses
                    item = re.sub(r'[JKkL\\/]', r'', item) # Remove beaming and stems
                    item = re.sub(r'[(){}xXyY&]', r'', item) # Remove slurs, phrases, elisions and editorial marks
                    if re.search('[qQP]', item):
                        grace_note_found = True
                    elif re.search('[A-Ga-g]', item):
                        note_found = True
                newline.append(item)

            # Remove grace note lines unless they contain a non-grace note in the same time line
            if grace_note_found and not note_found:
                continue

            if grace_note_found and note_found:
                print(f'Unremovable grace notes {line}')
                return False

            if not all([x == '.' for x in newline]) and not all([x == '!' for x in newline]):
                newbody.append('\t'.join(newline))

        header, footer = self.spines.dump()
        self.body = header[1:] + newbody
        self.first_line = len(header) - 1
        return True

    def split(self, chunk_sizes, stride=None):
        chunks = []
        spines = self.spines.clone()

        measures = [self.first_line]
        for i, line in enumerate(self.body[self.first_line:]):
            if re.match(r'^=(\d+|=)[^-]*', line):
                measures.append(i + self.first_line + 1)
        i = 0
        while i < len(measures) - 1:
            chunk_size = min(np.random.choice(chunk_sizes), len(measures) - i - 1)
            m_begin = measures[i]
            m_end = measures[i + chunk_size]

            header, footer = spines.dump()

            i += stride if stride else chunk_size
            if len(measures) - i - 1 < min(chunk_sizes):
                body = self.body[m_begin:]
                chunk = Kern(data='\n'.join(header + body + footer))
                chunks.append(chunk)
                break

            body = self.body[m_begin:m_end]
            chunk = Kern(data='\n'.join(header + body + footer))
            chunks.append(chunk)

            for line in self.body[m_begin:measures[i]]:
                if line.startswith('*'):
                    spines.update(line)

        return chunks

    def tosequence(self):
        krn = []
        for line in self.body[self.first_line:]:
            newline = []
            if line.startswith('='):
                if not re.match(r'^=(\d+|=)[^-]*', line):
                    continue
                newline.append('=')
            elif line.startswith(('*', '!')):
                continue
            else:
                line = re.sub(r'[^rA-Ga-g0-9.\[_\]#\-;\t]', r'', line) # Remove undefined symbols
                for i, item in enumerate(line.split('\t')):
                    if self.spine_types[i] == '**kern':
                        newline.append(item)

            krn.append('\t'.join(newline))

        krnseq = '\n'.join(krn)

        if re.search(r'(#|-|\.){2,}', krnseq):
            # Discard double sharps/flats/dots
            return None

        return krnseq
