# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
## 다른 데이터 셋을 위해 수정한 코드 ##

import os
import re
import json
from pathlib import Path

def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


# def preprocess(dataset_path, mode='phonetic'):
#     print('preprocess started..')

#     audio_paths = list()
#     transcripts = list()

#     percent_files = {
#         '087797': '퍼센트',
#         '215401': '퍼센트',
#         '284574': '퍼센트',
#         '397184': '퍼센트',
#         '501006': '프로',
#         '502173': '프로',
#         '542363': '프로',
#         '581483': '퍼센트'
#     }

#     for folder in os.listdir(dataset_path):
#         # folder : {KsponSpeech_01, ..., KsponSpeech_05}
#         if not folder.startswith('KsponSpeech'):
#             continue
#         path = os.path.join(dataset_path, folder)
#         for idx, subfolder in enumerate(os.listdir(path)):
#             path = os.path.join(dataset_path, folder, subfolder)

#             for jdx, file in enumerate(os.listdir(path)):
#                 if file.endswith('.txt'):
#                     with open(os.path.join(path, file), "r", encoding='cp949') as f:
#                         raw_sentence = f.read()
#                         if file[12:18] in percent_files.keys():
#                             new_sentence = sentence_filter(raw_sentence, mode, percent_files[file[12:18]])
#                         else:
#                             new_sentence = sentence_filter(raw_sentence, mode=mode)

#                     audio_paths.append(os.path.join(folder, subfolder, file))
#                     transcripts.append(new_sentence)

#                 else:
#                     continue

#     return audio_paths, transcripts


# def preprocess(dataset_path, mode='phonetic'):
#     print('preprocess started..')

#     audio_paths, transcripts = [], []

#     # 그대로 두어도 됨(Kspon 전용 특례). 파일명이 다르면 자동 무시됨.
#     percent_files = {
#         '087797': '퍼센트',
#         '215401': '퍼센트',
#         '284574': '퍼센트',
#         '397184': '퍼센트',
#         '501006': '프로',
#         '502173': '프로',
#         '542363': '프로',
#         '581483': '퍼센트'
#     }

#     root = Path(dataset_path).resolve()

#     # dataset_path 하위의 모든 폴더/파일을 재귀적으로 순회
#     for dirpath, _, filenames in os.walk(root):
#         dirpath = Path(dirpath)
#         for name in filenames:
#             # 지금은 .txt만 처리(요청대로 JSON 처리는 나중에)
#             if not name.lower().endswith('.txt'):
#                 continue

#             file_path = dirpath / name

#             # 안전 인코딩(cp949 -> 안되면 utf-8 시도)
#             try:
#                 raw = file_path.read_text(encoding='cp949')
#             except UnicodeDecodeError:
#                 raw = file_path.read_text(encoding='utf-8')

#             # Kspon 특례(파일명 길이 짧으면 무시)
#             file_id = name[12:18] if len(name) >= 18 else None
#             rep = percent_files.get(file_id)

#             # 기존 규칙 적용(괄호/특수문자 등)
#             new_sentence = sentence_filter(raw, mode, rep)

#             # 원래 코드는 .txt 경로를 그대로 넣었음(지금은 그 동작 유지)
#             rel_dir = dirpath.relative_to(root)
#             audio_paths.append(str(rel_dir / name))
#             transcripts.append(new_sentence)

#     return audio_paths, transcripts




def preprocess(dataset_path, mode='phonetic'):
    """
    - dataset_path 하위의 모든 .json을 재귀 순회
    - Transcript 전처리 (기존 sentence_filter 재사용)
    - 오디오 경로는 존재확인 없이 문자열만 생성:
        1) File_id가 있으면 <same_dir>/<File_id>
        2) 없으면 <same_dir>/<same_stem>.wav
    - 반환: (audio_paths, transcripts)  # audio_paths는 dataset 루트 기준 상대경로 (posix)
    """
    print('preprocess started..')

    root = Path(dataset_path).resolve()
    audio_paths, transcripts = [], []

    for json_path in root.rglob('*.json'):
        # JSON 읽기
        try:
            obj = json.loads(json_path.read_text(encoding='utf-8'))
        except UnicodeDecodeError:
            obj = json.loads(json_path.read_text(encoding='cp949'))
        except Exception as e:
            print(f'[WARN] JSON parse failed: {json_path} ({e})')
            continue

        # Transcript 추출
        raw = obj.get('Transcript', '')
        if not isinstance(raw, str) or not raw.strip():
            print(f'[WARN] Missing Transcript in: {json_path.name}')
            continue

        # 전처리 (괄호/특수기호 규칙 동일)
        new_sentence = sentence_filter(raw, mode)

        # WAV 경로 문자열 생성 (존재 확인 X)
        file_id = obj.get('File_id')
        if isinstance(file_id, str) and file_id.strip():
            wav_path = json_path.parent / file_id
        else:
            wav_path = json_path.with_suffix('.wav')

        # 루트 기준 상대경로, POSIX 문자열로 저장 (학습 파이프라인 호환 좋음)
        rel = wav_path.relative_to(root).as_posix()
        audio_paths.append(rel)
        transcripts.append(new_sentence)

    print(f'[INFO] collected items: {len(audio_paths)}')
    return audio_paths, transcripts

