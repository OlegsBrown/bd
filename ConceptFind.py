import os
import shutil
import hashlib
import multiprocessing
import re
import pandas as pd
import sys
import zipfile
import cv2
import csv
import nltk
import numpy as np
import pytesseract
import unicodedata
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from node2vec import Node2Vec
from collections import Counter
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QLabel, QListWidget, QFormLayout, QGroupBox, QProgressBar,
    QPushButton, QSlider, QSpinBox, QMessageBox, QTableWidget,
    QTableWidgetItem)
from PyQt5.QtGui import QColor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Tesseract OCR ceļš - pielāgot pēc vajadzības
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def full_text_preprocessing(text, stop_words=None):
    # Normalizē un noņem diakritiskās zīmes
    normalized = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in normalized if not unicodedata.combining(c)])
    text = text.lower()
    # Noņem simbolus un pieturzīmes
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    wordset = text.split()
    # Ja vajag, tad noņem stopvārdus
    if stop_words:
        wordset = [word for word in wordset if word not in stop_words]
    return ' '.join(wordset)

# Funkcija teksta faila nolasīšanai
def read_file_content(f_path):
    with open(f_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# No teksta datiem izveido DataFrame ar konceptiem un frāzēm
def parse_txt_file(data):
    lines = data.split('\n')
    frame_list = []
    for l in lines:
        l = l.strip().lower()
        if '\t' in l:
            element = l.split('\t')
        elif ',' in l:
            element = l.split(',')
        else:
            element = l.split()
        element = [p.strip() for p in element]
        if len(element) < 3:
            element += [''] * (3 - len(element))
        frame_list.append([element[0], element[1], ','.join(element[2:])])
    return pd.DataFrame(frame_list, columns=['Concept1', 'LinkLabel', 'Concept2'])

# Ģenerē n-grammas  - simbolu vai vārdu
def generate_ngrams(text, n, ngram_type='char'):
    if ngram_type == 'char':
        text = text.replace(' ', '')
        if len(text) < n:
            return []
        return [text[s:s+n] for s in range(len(text)-n+1)]
    elif ngram_type == 'word':
        t = nltk.RegexpTokenizer(r'\w+', flags=re.UNICODE)
        words = t.tokenize(text)
        words = [w for w in words if '_' not in w]
        if len(words) < n:
            return []
        return [' '.join(gram) for gram in nltk.ngrams(words, n)]
    return []

# Aprēķina kosinusa līdzību starp divām kolonnām
def calculate_cosine_similarity(col1, col2):
    terms = set(col1.keys()) | set(col2.keys())
    res = sum(col1.get(k, 0)*col2.get(k, 0) for k in terms)
    sim = sum(v**2 for v in col1.values())**0.5
    sim2 = sum(v**2 for v in col2.values())**0.5
    if sim == 0 or sim2 == 0:
        return 0.0
    return res / (sim*sim2)

# Izveido SHA-256 hešus no vārdu n-grammām
def create_hashes(text, n):
    ngrams_list = generate_ngrams(text, n, 'word')
    return [int(hashlib.sha256(gram.encode('utf-8')).hexdigest(), 16) for gram in ngrams_list]

# Winnowing algoritms pirkstu nospiedumu noteikšanai
def apply_winnowing(hashes, win_size):
    fingerprints = set()
    if len(hashes) < win_size:
        fingerprints.update(hashes)
        return fingerprints
    for i in range(len(hashes)-win_size+1):
        fingerprints.add(min(hashes[i:i+win_size]))
    return fingerprints

# Salīdzina divu pirkstu nospiedumu kopas (Jacard)
def compare_fingerprints(f1, f2):
    inter = f1 & f2
    union = f1 | f2
    if not union:
        return 0.0
    return len(inter)/len(union)

# Konfigurācijas klase attēlu apstrādei
class Config:
    NODE_AREA_THRESHOLD = 25000
    THRESHOLD_VALUE = 90
    NODE2VEC_DIMENSIONS = 80
    NODE2VEC_WALK_LENGTH = 50
    NODE2VEC_NUM_WALKS = 500
    OCR_LANGUAGE = 'lav'
    OCR_CONFIG = '--psm 10'
    NUM_WORKERS = multiprocessing.cpu_count()

# Veic attēla binarizāciju
def binarize(img, threshold=Config.THRESHOLD_VALUE, adaptive=False):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if adaptive:
        node_thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return node_thresh
    else:
        g = cv2.medianBlur(g, 5)
        _, thresh = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY_INV)
        return thresh

# Attēlu apstrāde un teksta iegūšana
class ImageHandler:
    def __init__(self, ocr_language='lav', ocr_config="--psm 10"):
        self.ocr_language = ocr_language
        self.ocr_config = ocr_config

    # Pirmsapstrāde attēlam
    def preprocess_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Neizdevās ielādēt attēlu: {path}")
        thresh = binarize(img, threshold=Config.THRESHOLD_VALUE, adaptive=False)
        return img, thresh

    # Atrod karšu mezglus un salidzina ar ievadīto laukumu
    def detect_nodes_and_edges(self, thresh):
        cont, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nodes, edges = [], []
        for c in cont:
            if cv2.contourArea(c) > Config.NODE_AREA_THRESHOLD:
                nodes.append(c)
            else:
                edges.append(c)
        return nodes, edges

    # No katra mezgla kontūras izgriež attēlu un nolasa tekstu ar OCR
    def extract_text_from_nodes(self, image, nodes):
        node_texts = {}
        for idx, num in enumerate(nodes):
            x, y, x1, y2 = cv2.boundingRect(num)
            node_img = image[y:y+y2, x:x+x1]
            node_thresh = binarize(node_img, adaptive=True)
            try:
                text = pytesseract.image_to_string(node_thresh, lang=self.ocr_language, config=self.ocr_config).strip()
                node_texts[idx] = text if text else "empty"
            except Exception as e:
                print(f"Neizdevās nolasīt tekstu no mezgla: {e}")
                node_texts[idx] = "error"
        return node_texts

    # Izveido grafu no karšu konceptiem un frāzēm
    def build_graph(self, node_texts, edge, node):
        G = nx.DiGraph()
        G.add_nodes_from([(idx, {'label': txt}) for idx, txt in node_texts.items()])
        for e in edge:
            M = cv2.moments(e)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            distances = []
            for indx, cnt in enumerate(node):
                x, y, x1, y1 = cv2.boundingRect(cnt)
                dist = np.hypot(cX - (x + x1 // 2), cY - (y + y1 // 2))
                distances.append((dist, indx))
            distances.sort(key=lambda x: x[0])
            if len(distances) >= 2:
                G.add_edge(distances[0][1], distances[1][1])
        return G

    # Apstrādā karti, lai iegūtu grafu un nolasītu tekstu no mezgliem
    def process_image(self, image_path):
        image, thresh = self.preprocess_image(image_path)
        nodes, edges = self.detect_nodes_and_edges(thresh)
        node_texts = self.extract_text_from_nodes(image, nodes)
        G = self.build_graph(node_texts, edges, nodes)
        return G, node_texts

# Grafa iegulumu veidošana, izmantojot Node2Vec un TF-IDF uz mezgla teksta
class GraphVectorizer:
    def __init__(self, vectorizer, dimensions=80, walk_length=50, num_walks=500, window=12):
        self.num_walks = num_walks
        self.vectorizer = vectorizer
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.window = window

    def generate_embeddings(self, G):
        if not G.nodes():
            return None
        node_texts = [data['label'] for _, data in G.nodes(data=True)]
        node_ind = list(G.nodes())
        text_emb = self.vectorizer.transform(node_texts).toarray()
        vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length,
                       num_walks=self.num_walks, workers=1, quiet=True, seed=50)
        emb_model = vec.fit(window=self.window, min_count=1, seed=50)
        graph_emb = np.array([emb_model.wv[str(node)] for node in node_ind])
        comb = np.hstack((graph_emb, text_emb))
        return {str(node): comb[i] for i, node in enumerate(node_ind)}

    @staticmethod
    def compare_embeddings(e1, e2):
        all_nodes = set(e1.keys()) | set(e2.keys())
        emb_dim = len(next(iter(e1.values())))
        m1 = [e1.get(n, np.zeros(emb_dim)) for n in all_nodes]
        m2 = [e2.get(n, np.zeros(emb_dim)) for n in all_nodes]
        sim = cosine_similarity([np.mean(m1, axis=0)], [np.mean(m2, axis=0)])
        return sim[0][0]

# Pogu stils - teksts un attēls
class BaseChecker(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.button_size = QtCore.QSize(150, 40)

    def style_button(self, button, color):
        button.setFixedHeight(40)
        button.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: bold;"
        )

# Plūsma teksta pārbaudei
class AnalysisWorker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, files, mode, n_value, window_size=None, exclude_stopwords=False,
                 top_n_stopwords=0, weight_phrases=0.5, weight_concepts=0.5):
        super().__init__()
        self.files = files
        self.mode = mode
        self.n_value = n_value
        self.window_size = window_size
        self.exclude_stopwords = exclude_stopwords
        self.top_n_stopwords = top_n_stopwords
        self.weight_phrases = weight_phrases
        self.weight_concepts = weight_concepts

    def run(self):
        file_data = []
        all_words = []
        # Apstrādāt ievadītos failus
        for f in self.files:
            if zipfile.is_zipfile(f):
                with zipfile.ZipFile(f, 'r') as z:
                    ex_dir = os.path.join(os.path.dirname(f), f"extracted_{os.path.basename(f)}")
                    z.extractall(ex_dir)
                    for base, _, fs in os.walk(ex_dir):
                        for zf in fs:
                            fp = os.path.join(base, zf)
                            if fp.endswith('.txt'):
                                self.process_file(fp, file_data, all_words)
            else:
                if f.endswith('.txt'):
                    self.process_file(f, file_data, all_words)

        # Ja nepieciešams, mēklē stopvārdus
        stop_words_set = set()
        if self.exclude_stopwords and self.top_n_stopwords > 0:
            wc = Counter(all_words)
            common = wc.most_common(self.top_n_stopwords)
            stop_words_set = {w for w, _ in common}
            for fd in file_data:
                fd['phrase_text'] = full_text_preprocessing(fd['phrase_text'], stop_words=stop_words_set)
                fd['concept_text'] = full_text_preprocessing(fd['concept_text'], stop_words=stop_words_set)
        else:
            for fd in file_data:
                fd['phrase_text'] = full_text_preprocessing(fd['phrase_text'])
                fd['concept_text'] = full_text_preprocessing(fd['concept_text'])

        # Atkarībā no izvēlētās metodes ģenerē n-grammas vai hešus
        if self.mode in ['Simbolu N-Gramas', 'Vārdu N-Gramas']:
            ngram_type = 'char' if self.mode == 'Simbolu N-Gramas' else 'word'
            for fd in file_data:
                ng_ph = generate_ngrams(fd['phrase_text'], self.n_value, ngram_type)
                ng_co = generate_ngrams(fd['concept_text'], self.n_value, ngram_type)
                fd['ngram_counter_phrases'] = Counter(ng_ph)
                fd['ngram_counter_concepts'] = Counter(ng_co)
        elif self.mode == 'Heša metode':
            for fd in file_data:
                h_ph = create_hashes(fd['phrase_text'], self.n_value)
                h_co = create_hashes(fd['concept_text'], self.n_value)
                f_ph = apply_winnowing(h_ph, self.window_size)
                f_co = apply_winnowing(h_co, self.window_size)
                fd['fingerprints_phrases'] = f_ph
                fd['fingerprints_concepts'] = f_co

        results = []
        # Salīdzina katru failu pāri
        for i in range(len(file_data)):
            for j in range(i+1, len(file_data)):
                f1 = file_data[i]
                f2 = file_data[j]
                if self.mode in ['Simbolu N-Gramas', 'Vārdu N-Gramas']:
                    sim_p = calculate_cosine_similarity(f1.get('ngram_counter_phrases', {}),
                                                        f2.get('ngram_counter_phrases', {}))
                    sim_c = calculate_cosine_similarity(f1.get('ngram_counter_concepts', {}),
                                                        f2.get('ngram_counter_concepts', {}))
                else:
                    sim_p = compare_fingerprints(f1.get('fingerprints_phrases', set()),
                                                 f2.get('fingerprints_phrases', set()))
                    sim_c = compare_fingerprints(f1.get('fingerprints_concepts', set()),
                                                 f2.get('fingerprints_concepts', set()))

                results.append({
                    'file1': os.path.splitext(f1['file_name'])[0],
                    'file2': os.path.splitext(f2['file_name'])[0],
                    'similarity_phrases': sim_p,
                    'similarity_concepts': sim_c,
                    'analysis_method': self.mode,
                    'n_value': str(self.n_value),
                    'window_size': str(self.window_size) if self.window_size else '',
                    'weight_phrases': str(self.weight_phrases),
                    'weight_concepts': str(self.weight_concepts),
                    'num_stopwords': str(self.top_n_stopwords),
                })
        self.result_ready.emit(results)

    def process_file(self, fp, file_data, all_words):
        content = read_file_content(fp)
        if content is None:
            return
        df = parse_txt_file(content)
        if df.empty:
            return
        # Izvelk izejas tekstus
        phrases_text = ' '.join(df['LinkLabel'].str.lower().str.strip().tolist())
        concepts_text = ' '.join(df[['Concept1', 'Concept2']]
                                 .apply(lambda x: ' '.join(x), axis=1)
                                 .str.lower().str.strip().tolist())

        all_words.extend(phrases_text.split())
        all_words.extend(concepts_text.split())

        file_data.append({
            'file_name': os.path.basename(fp),
            'phrase_text': phrases_text,
            'concept_text': concepts_text,
        })

# --------------------------------------------
# Teksta pārbaudes interfeiss
# --------------------------------------------
class TextChecker(BaseChecker):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konceptu karšu līdzības noteikšana tekstam")
        self.files_paths = []
        self.results = []
        self.analysis_results = []
        self.current_parameters = {}
        self.init_ui()

    def init_ui(self):
        self.mode_list = ['Simbolu N-Gramas', 'Vārdu N-Gramas', 'Heša metode']

        main_window = QtWidgets.QVBoxLayout(self)
        up_lay = QtWidgets.QHBoxLayout()
        left_lay = QtWidgets.QVBoxLayout()

        file_list_label = QLabel("Pievienotie faili:")
        self.file_list_widget = QListWidget()
        upload_layout = QtWidgets.QHBoxLayout()
        self.txt_button = QPushButton("Augšupielādēt TXT")
        self.zip_button = QPushButton("Augšupielādēt ZIP")
        upload_layout.addWidget(self.txt_button)
        upload_layout.addWidget(self.zip_button)
        for b in [self.txt_button, self.zip_button]:
            self.style_button(b, "#2196F3")

        left_lay.addWidget(file_list_label)
        left_lay.addWidget(self.file_list_widget)
        left_lay.addLayout(upload_layout)

        settings_layout = QtWidgets.QVBoxLayout()
        mode_group = QGroupBox("Pārbaudes režīms")
        mode_form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.mode_list)
        mode_form.addRow("Metode:", self.mode_combo)
        self.n_label = QLabel("N vērtība:")
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setRange(1, 10)
        self.n_spinbox.setValue(3)
        mode_form.addRow(self.n_label, self.n_spinbox)
        self.window_size_label = QLabel("Heša loga izmērs:")
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(1, 20)
        self.window_size_spinbox.setValue(4)
        mode_form.addRow(self.window_size_label, self.window_size_spinbox)
        mode_group.setLayout(mode_form)
        settings_layout.addWidget(mode_group)

        weight_group = QGroupBox("Svara iestatījumi")
        weight_form = QFormLayout()
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(50)
        self.weight_slider.setTickInterval(10)
        self.weight_slider.valueChanged.connect(self.update_weights)
        self.weight_display = QLabel("50% / 50%")
        weight_form.addRow("Frāzes / Koncepti:", self.weight_slider)
        weight_form.addRow("Svari:", self.weight_display)
        weight_group.setLayout(weight_form)
        settings_layout.addWidget(weight_group)

        stopwords_group = QGroupBox("Stopvārdi")
        stopwords_form = QFormLayout()
        self.stopwords_checkbox = QCheckBox("Izslēgt stopvārdus")
        self.stopwords_checkbox.stateChanged.connect(self.toggle_stopwords)
        self.stopwords_spinbox = QSpinBox()
        self.stopwords_spinbox.setRange(0, 20)
        self.stopwords_spinbox.setValue(0)
        self.stopwords_spinbox.setEnabled(False)
        stopwords_form.addRow(self.stopwords_checkbox)
        stopwords_form.addRow("Stopvārdu skaits:", self.stopwords_spinbox)
        stopwords_group.setLayout(stopwords_form)
        settings_layout.addWidget(stopwords_group)

        threshold_group = QGroupBox("Līdzības klasifikators")
        threshold_form = QFormLayout()
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.01, 1.00)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(0.80)
        threshold_form.addRow("Nomarķēt, ja sakritība >:", self.threshold_spinbox)
        threshold_group.setLayout(threshold_form)
        settings_layout.addWidget(threshold_group)

        left_lay.addLayout(settings_layout)

        self.analyze_button = QPushButton("Pārbaudīt")
        self.save_button = QPushButton("Saglabāt rezultātus")
        self.clear_button = QPushButton("Notīrīt rezultātus")
        self.style_button(self.analyze_button, "#4CAF50")
        self.style_button(self.save_button, "#2196F3")
        self.style_button(self.clear_button, "#f44336")

        button_lay = QtWidgets.QVBoxLayout()
        button_lay.addWidget(self.analyze_button)
        sub_button_lay = QtWidgets.QHBoxLayout()
        sub_button_lay.addWidget(self.save_button)
        sub_button_lay.addWidget(self.clear_button)
        button_lay.addLayout(sub_button_lay)
        left_lay.addLayout(button_lay)

        right_lay = QtWidgets.QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Fails 1", "Fails 2", "Sakritības procents",
            "Konceptu sakritība", "Frāžu sakritība", "Klasifikācija"
        ])
        right_lay.addWidget(QLabel("Rezultāti:"))
        right_lay.addWidget(self.results_table)
        self.progress_bar = QProgressBar()
        right_lay.addWidget(self.progress_bar)

        up_lay.addLayout(left_lay)
        up_lay.addLayout(right_lay)
        main_window.addLayout(up_lay)

        self.analyze_button.clicked.connect(self.start_analysis)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear)
        self.txt_button.clicked.connect(lambda: self.load_files('txt'))
        self.zip_button.clicked.connect(lambda: self.load_files('zip'))
        self.stopwords_checkbox.setChecked(False)
        self.update_mode()
        self.update_weights()
        self.mode_combo.currentIndexChanged.connect(self.update_mode)

    def toggle_stopwords(self, state):
        self.stopwords_spinbox.setEnabled(state == Qt.Checked)
        if state != Qt.Checked:
            self.stopwords_spinbox.setValue(0)

    def load_files(self, ftype):
        filt = "TXT faili (*.txt)" if ftype == 'txt' else "ZIP faili (*.zip)"
        files, _ = QFileDialog.getOpenFileNames(self, "Izvēlieties failus", "", filt)
        if files:
            for f in files:
                if f not in self.files_paths:
                    self.file_list_widget.addItem(os.path.basename(f))
                    self.files_paths.append(f)
            self.disable_load_buttons(ftype)

    def disable_load_buttons(self, sel_type):
        self.txt_button.setEnabled(sel_type != 'txt')
        self.zip_button.setEnabled(sel_type != 'zip')

    def update_mode(self):
        m = self.mode_combo.currentText()
        if m in ['Simbolu N-Gramas', 'Vārdu N-Gramas']:
            self.n_label.setEnabled(True)
            self.n_spinbox.setEnabled(True)
            self.window_size_label.setEnabled(False)
            self.window_size_spinbox.setEnabled(False)
        elif m == 'Heša metode':
            self.n_label.setEnabled(True)
            self.n_spinbox.setEnabled(True)
            self.window_size_label.setEnabled(True)
            self.window_size_spinbox.setEnabled(True)
        else:
            self.n_label.setEnabled(False)
            self.n_spinbox.setEnabled(False)
            self.window_size_label.setEnabled(False)
            self.window_size_spinbox.setEnabled(False)

    def update_weights(self):
        v = self.weight_slider.value()
        self.weight_display.setText(f"{v}% / {100 - v}%")

    def start_analysis(self):
        if not self.files_paths:
            QMessageBox.warning(self, "Brīdinājums", "Nav failu pārbaudei!")
            return
        m = self.mode_combo.currentText()
        n_val = self.n_spinbox.value()
        w_size = self.window_size_spinbox.value() if self.window_size_spinbox.isEnabled() else None
        excl_sw = self.stopwords_checkbox.isChecked()
        top_sw = self.stopwords_spinbox.value() if excl_sw else 0
        w_p = self.weight_slider.value()/100.0
        w_c = (100 - self.weight_slider.value())/100.0

        self.current_parameters = {
            'analysis_method': m,
            'n_value': str(n_val),
            'window_size': str(w_size) if w_size else '',
            'exclude_stopwords': excl_sw,
            'top_n_stopwords': str(top_sw),
            'weight_phrases': f"{w_p:.2f}",
            'weight_concepts': f"{w_c:.2f}",
            'threshold': self.threshold_spinbox.value(),
        }

        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.results = []
        self.analyze_button.setEnabled(False)
        self.thread = AnalysisWorker(self.files_paths, m, n_val, w_size, excl_sw, top_sw, w_p, w_c)
        self.thread.result_ready.connect(self.display_results)
        self.thread.start()

    def display_results(self, results):
        self.analyze_button.setEnabled(True)
        if not results:
            QMessageBox.information(self, "Informācija", "Analīze pabeigta, sakritības nav.")
            self.results_table.setRowCount(0)
            return
        w_p = float(self.current_parameters['weight_phrases'])
        w_c = float(self.current_parameters['weight_concepts'])
        th = float(self.current_parameters['threshold'])

        for r in results:
            perc = (r['similarity_phrases']*w_p + r['similarity_concepts']*w_c)
            r['percentage'] = perc
            r['classification'] = 1 if perc >= th else 0

        results.sort(key=lambda x: x['percentage'], reverse=True)
        top_results = results[:1000]
        self.analysis_results.append({'results': top_results, 'parameters': self.current_parameters.copy()})
        self.results = top_results

        self.results_table.setRowCount(len(top_results))
        for row, r in enumerate(top_results):
            f1, f2 = r['file1'], r['file2']
            perc = r['percentage']*100
            sc = r['similarity_concepts']*100
            sp = r['similarity_phrases']*100
            c = r['classification']

            i1 = QTableWidgetItem(f1)
            i2 = QTableWidgetItem(f2)
            i3 = QTableWidgetItem(f"{perc:.2f}%")
            i4 = QTableWidgetItem(f"{sc:.2f}%")
            i5 = QTableWidgetItem(f"{sp:.2f}%")
            i6 = QTableWidgetItem(str(c))

            if perc >= 80:
                color = QColor(255, 150, 150)
            elif perc >= 65:
                color = QColor(255, 200, 150)
            elif perc >= 55:
                color = QColor(200, 230, 255)
            else:
                color = QColor(200, 255, 200)

            for it in [i1, i2, i3, i4, i5, i6]:
                it.setBackground(color)

            self.results_table.setItem(row, 0, i1)
            self.results_table.setItem(row, 1, i2)
            self.results_table.setItem(row, 2, i3)
            self.results_table.setItem(row, 3, i4)
            self.results_table.setItem(row, 4, i5)
            self.results_table.setItem(row, 5, i6)
            self.progress_bar.setValue(int((row+1)/len(top_results)*100))

    def save_results(self):
        if not self.analysis_results:
            QMessageBox.warning(self, "Brīdinājums", "Nav rezultātu, ko saglabāt.")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Saglabāt rezultātus", "", "CSV faili (*.csv);;Visi faili (*)")
        if fp:
            all_dfs = []
            for ad in self.analysis_results:
                rs = ad['results']
                if not rs:
                    continue
                df = pd.DataFrame(rs)
                df.rename(columns={
                    'analysis_method': 'metode', 'file1': 'fails1', 'file2': 'fails2',
                    'percentage': 'sakritibas_procents', 'similarity_concepts': 'konceptu_sakritiba',
                    'similarity_phrases': 'frazu_sakritiba', 'n_value': 'n_vertiba',
                    'window_size': 'loga_izmers', 'num_stopwords': 'stopvardu_skaits',
                    'weight_concepts': 'konceptu_svars', 'weight_phrases': 'frazu_svars',
                    'classification': 'klasifikacija'
                }, inplace=True)
                cols = [
                    'metode', 'fails1', 'fails2', 'sakritibas_procents',
                    'konceptu_sakritiba', 'frazu_sakritiba', 'klasifikacija',
                    'n_vertiba', 'loga_izmers', 'stopvardu_skaits',
                    'konceptu_svars', 'frazu_svars'
                ]
                df = df[cols]
                df['sakritibas_procents'] = df['sakritibas_procents'].astype(float).round(4)
                df['konceptu_sakritiba'] = df['konceptu_sakritiba'].astype(float).round(4)
                df['frazu_sakritiba'] = df['frazu_sakritiba'].astype(float).round(4)
                df['n_vertiba'] = pd.to_numeric(df['n_vertiba'], errors='coerce').fillna(0).astype(int)
                df['loga_izmers'] = pd.to_numeric(df['loga_izmers'], errors='coerce').fillna(0).astype(int)
                df['stopvardu_skaits'] = pd.to_numeric(df['stopvardu_skaits'], errors='coerce').fillna(0).astype(int)
                df['konceptu_svars'] = df['konceptu_svars'].astype(float).round(2)
                df['frazu_svars'] = df['frazu_svars'].astype(float).round(2)
                df['klasifikacija'] = df['klasifikacija'].astype(int)
                all_dfs.append(df)
            if not all_dfs:
                QMessageBox.information(self, "Informācija", "Nav rezultātu, ko saglabāt.")
                return
            fr = pd.concat(all_dfs, ignore_index=True)
            fr.to_csv(fp, index=False, encoding='utf-8', sep=',', decimal='.', float_format='%.4f')
            QMessageBox.information(self, "Veiksmīgi", "Rezultāti veiksmīgi saglabāti.")

    def clear(self):
        self.file_list_widget.clear()
        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.results = []
        self.files_paths = []
        self.analysis_results = []
        self.txt_button.setEnabled(True)
        self.zip_button.setEnabled(True)
        self.stopwords_checkbox.setChecked(False)
        self.stopwords_spinbox.setValue(0)
        self.stopwords_spinbox.setEnabled(False)

# --------------------------------------------
# Attēlu pārbaudes interfeiss
# --------------------------------------------
class ImageChecker(BaseChecker):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konceptu karšu līdzības noteikšana attēliem")
        self.images = []
        self.embeddings = {}
        self.results = []
        self.current_results = []
        self.initUI()

    def initUI(self):
        main_window = QtWidgets.QVBoxLayout(self)
        up_lay = QtWidgets.QHBoxLayout()
        left_lay = QtWidgets.QVBoxLayout()
        self.create_file_list_layout(left_lay)
        self.create_settings_layout(left_lay)
        right_lay = QtWidgets.QVBoxLayout()
        self.create_results_layout(right_lay)
        up_lay.addLayout(left_lay)
        up_lay.addLayout(right_lay)
        main_window.addLayout(up_lay)

    def create_file_list_layout(self, layout):
        layout.addWidget(QtWidgets.QLabel("Pievienotie faili:"))
        self.file_list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.file_list_widget)
        upload_layout = QtWidgets.QHBoxLayout()
        self.upload_images_button = QPushButton("Augšupielādēt attēlus")
        self.upload_zip_button = QPushButton("Augšupielādēt ZIP")
        upload_layout.addWidget(self.upload_images_button)
        upload_layout.addWidget(self.upload_zip_button)
        layout.addLayout(upload_layout)
        for b in [self.upload_images_button, self.upload_zip_button]:
            self.style_button(b, "#2196F3")
        self.upload_images_button.clicked.connect(self.load_images)
        self.upload_zip_button.clicked.connect(self.load_zip)

    def create_settings_layout(self, layout):
        layout.addWidget(QtWidgets.QLabel("Iestatījumi:"))
        s_layout = QtWidgets.QVBoxLayout()

        n2v_group = QGroupBox("Node2Vec parametri")
        n2v_form = QFormLayout()
        self.node2vec_dimensions_input = self.create_spin_box(Config.NODE2VEC_DIMENSIONS, 1, 1000, "Dimensija")
        self.node2vec_walk_length_input = self.create_spin_box(Config.NODE2VEC_WALK_LENGTH, 1, 1000, "Soļu garums")
        self.node2vec_num_walks_input = self.create_spin_box(Config.NODE2VEC_NUM_WALKS, 1, 1000, "Soļu skaits")
        self.node2vec_window_input = self.create_spin_box(12, 1, 100, "Loga izmērs (window param. Node2Vec)")
        n2v_form.addRow("Dimensija:", self.node2vec_dimensions_input)
        n2v_form.addRow("Soļu garums:", self.node2vec_walk_length_input)
        n2v_form.addRow("Soļu skaits:", self.node2vec_num_walks_input)

        n2v_form.addRow("Loga izmērs:", self.node2vec_window_input)

        n2v_group.setLayout(n2v_form)
        s_layout.addWidget(n2v_group)

        ocr_group = QGroupBox("OCR iestatījumi")
        ocr_form = QFormLayout()
        self.ocr_language_input = QComboBox()
        self.ocr_language_input.addItems(['eng', 'lav', 'rus'])
        self.ocr_language_input.setCurrentText(Config.OCR_LANGUAGE)
        self.ocr_language_input.setToolTip("OCR valoda")
        self.ocr_config_input = QComboBox()
        self.ocr_config_input.addItems([f'--psm {i}' for i in range(1, 14)])
        self.ocr_config_input.setCurrentText(Config.OCR_CONFIG)
        self.ocr_config_input.setToolTip("OCR konfigurācija")
        ocr_form.addRow("OCR valoda:", self.ocr_language_input)
        ocr_form.addRow("OCR konfigurācija:", self.ocr_config_input)
        ocr_group.setLayout(ocr_form)
        s_layout.addWidget(ocr_group)

        other_group = QGroupBox("Papildu iestatījumi")
        other_form = QFormLayout()
        self.node_area_threshold_input = self.create_spin_box(Config.NODE_AREA_THRESHOLD, 0, 100000, "Mezglu laukuma slieksnis", step=100)
        self.threshold_value_input = self.create_spin_box(Config.THRESHOLD_VALUE, 0, 255, "Binarizācijas slieksnis")
        other_form.addRow("Mezglu slieksnis:", self.node_area_threshold_input)
        other_form.addRow("Binarizācijas slieksnis:", self.threshold_value_input)
        other_group.setLayout(other_form)
        s_layout.addWidget(other_group)

        layout.addLayout(s_layout)
        self.check_button = QPushButton("Pārbaudīt")
        self.save_button = QPushButton("Saglabāt rezultātus")
        self.clear_button = QPushButton("Notīrīt rezultātus")
        self.style_button(self.check_button, "#4CAF50")
        self.style_button(self.save_button, "#2196F3")
        self.style_button(self.clear_button, "#f44336")
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.addWidget(self.check_button)
        sb_layout = QtWidgets.QHBoxLayout()
        sb_layout.addWidget(self.save_button)
        sb_layout.addWidget(self.clear_button)
        btn_layout.addLayout(sb_layout)
        layout.addLayout(btn_layout)

        self.check_button.clicked.connect(self.check_plagiarism)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear_results)

    def create_spin_box(self, val, mini, maxi, tt, step=1):
        sb = QtWidgets.QSpinBox()
        sb.setRange(mini, maxi)
        sb.setValue(val)
        sb.setSingleStep(step)
        sb.setToolTip(tt)
        return sb

    def create_results_layout(self, layout):
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Fails 1", "Fails 2", "Sakritības procents"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QtWidgets.QLabel("Rezultāti:"))
        layout.addWidget(self.results_table)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def load_images(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Izvēlieties attēlus", "", "Attēli (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if fs:
            self.images.extend(fs)
            for f in fs:
                self.file_list_widget.addItem(os.path.basename(f))
            QMessageBox.information(self, "Ielāde", f"Ielādēti {len(fs)} attēli.")

    def load_zip(self):
        zp, _ = QFileDialog.getOpenFileName(self, "Izvēlieties ZIP failu", "", "ZIP faili (*.zip)")
        if zp:
            temp = "temp_extracted_images"
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(temp)
            ef = [os.path.join(temp, f) for f in os.listdir(temp)
                  if f.lower().endswith(('.png','.bmp', '.tiff', '.jpg', '.jpeg'))]
            self.images.extend(ef)
            for f in ef:
                self.file_list_widget.addItem(os.path.basename(f))
            QMessageBox.information(self, "Ielāde", f"No ZIP faila ielādēti {len(ef)} attēli.")

    def check_plagiarism(self):
        if not self.images:
            QMessageBox.warning(self, "Kļūda", "Nav attēlu analīzei.")
            return
        Config.NODE_AREA_THRESHOLD = self.node_area_threshold_input.value()
        Config.THRESHOLD_VALUE = self.threshold_value_input.value()
        Config.NODE2VEC_DIMENSIONS = self.node2vec_dimensions_input.value()
        Config.NODE2VEC_WALK_LENGTH = self.node2vec_walk_length_input.value()
        Config.NODE2VEC_NUM_WALKS = self.node2vec_num_walks_input.value()
        Config.OCR_LANGUAGE = self.ocr_language_input.currentText()
        Config.OCR_CONFIG = self.ocr_config_input.currentText()
        node2vec_window = self.node2vec_window_input.value()

        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.embeddings = {}
        self.current_results = []
        ih = ImageHandler(Config.OCR_LANGUAGE, Config.OCR_CONFIG)
        image_results, all_node_texts = {}, []

        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as ex:
            futures = {ex.submit(ih.process_image, img): img for img in self.images}
            total = len(futures)
            for i, fut in enumerate(futures):
                ip = futures[fut]
                try:
                    G, node_texts = fut.result()
                    if G:
                        image_results[ip] = (G, node_texts)
                        all_node_texts.extend(node_texts.values())
                except:
                    pass
                self.progress_bar.setValue(int((i+1)/total*50))

        if not image_results:
            QMessageBox.warning(self, "Kļūda", "Kļūda apstrādājot attēlus.")
            return

        vectorizer = TfidfVectorizer()
        vectorizer.fit(all_node_texts or [""])

        for ip, (G, _) in image_results.items():
            gv = GraphVectorizer(
                vectorizer,
                Config.NODE2VEC_DIMENSIONS,
                Config.NODE2VEC_WALK_LENGTH,
                Config.NODE2VEC_NUM_WALKS,
                window=node2vec_window
            )
            emb = gv.generate_embeddings(G)
            if emb:
                self.embeddings[ip] = emb

        total_pairs = len(self.embeddings)*(len(self.embeddings)-1)//2
        if total_pairs == 0:
            QMessageBox.warning(self, "Kļūda", "Nepietiek datu salīdzināšanai.")
            return

        current_pair = 0
        keys = list(self.embeddings.keys())
        for i, img1 in enumerate(keys):
            for img2 in keys[i+1:]:
                e1, e2 = self.embeddings[img1], self.embeddings[img2]
                sim = GraphVectorizer.compare_embeddings(e1, e2)
                self.current_results.append((os.path.basename(img1), os.path.basename(img2), sim,
                                             Config.NODE2VEC_DIMENSIONS, Config.NODE2VEC_WALK_LENGTH,
                                             Config.NODE2VEC_NUM_WALKS, Config.NODE_AREA_THRESHOLD,
                                             Config.THRESHOLD_VALUE, int(Config.OCR_CONFIG.split()[-1])))
                current_pair += 1
                self.progress_bar.setValue(50+int(current_pair/total_pairs*50))
        self.update_results_table()
        self.results.extend(self.current_results)

    def update_results_table(self):
        self.current_results.sort(key=lambda x: x[2], reverse=True)
        self.results_table.setRowCount(len(self.current_results))
        for row, (img1, img2, perc, *_) in enumerate(self.current_results):
            i1, i2, i3 = QTableWidgetItem(img1), QTableWidgetItem(img2), QTableWidgetItem(f"{perc*100:.2f}%")
            if perc >= 0.8:
                for it in [i1, i2, i3]:
                    it.setBackground(QtGui.QColor(255, 150, 150))
            self.results_table.setItem(row, 0, i1)
            self.results_table.setItem(row, 1, i2)
            self.results_table.setItem(row, 2, i3)

    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "Kļūda", "Nav rezultātu, ko saglabāt.")
            return
        sp, _ = QFileDialog.getSaveFileName(self, "Saglabāt rezultātus", "", "CSV faili (*.csv)")
        if sp:
            with open(sp, 'w', newline='', encoding='utf-8') as csvfile:
                fn = ['fails_1', 'fails_2', 'sakritiba_procentos', 'dimensija', 'solu_garums',
                      'solu_skaits', 'mezglu_slieksnis', 'binarizacijas_slieksnis', 'ocr_rezims']
                w = csv.DictWriter(csvfile, fieldnames=fn)
                w.writeheader()
                for r in self.results:
                    w.writerow({
                        'fails_1': r[0],
                        'fails_2': r[1],
                        'sakritiba_procentos': f"{r[2]:.4f}",
                        'dimensija': r[3],
                        'solu_garums': r[4],
                        'solu_skaits': r[5],
                        'mezglu_slieksnis': r[6],
                        'binarizacijas_slieksnis': r[7],
                        'ocr_rezims': r[8]
                    })
            QMessageBox.information(self, "Saglabāšana", "Rezultāti veiksmīgi saglabāti.")

    def clear_results(self):
        self.results_table.setRowCount(0)
        self.results.clear()
        self.current_results.clear()
        self.images.clear()
        self.embeddings.clear()
        self.file_list_widget.clear()
        self.progress_bar.setValue(0)
        self.upload_images_button.setEnabled(True)
        self.upload_zip_button.setEnabled(True)
        tmp = "temp_extracted_images"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)

# Galvenā programma teksta un attēlu pārbaudei
class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konceptu karšu līdzības noteikšana")
        self.setGeometry(100, 100, 1200, 700)
        self.initUI()

    def initUI(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.text_checker = TextChecker()
        self.image_checker = ImageChecker()
        self.tabs.addTab(self.text_checker, "Teksta pārbaude")
        self.tabs.addTab(self.image_checker, "Attēlu pārbaude")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(245, 245, 245))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.black)
    app.setPalette(palette)
    app.setStyle('Fusion')
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


