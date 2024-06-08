import numpy as np
import torch

# Bert Model
class BERT:
    max_sequence_length = 32

# News Model
class news_model:
    input_size = 768
    hidden_size = 256
    num_layers = 1
    num_classes = 39
    output_labels = [
        'olahraga',
        'ekonomi',
        'kecelakaan',
        'kriminalitas',
        'bencana',
        'bulutangkis',
        'voli',
        'basket',
        'tenis',
        'pembunuhan',
        'pencurian',
        'gempa',
        'kebakaran',
        'tsunami',
        'gunung_meletus',
        'banjir',
        'puting_beliung',
        'kekeringan',
        'abrasi',
        'longsor',
        'pendidikan',
        'teknologi',
        'politik',
        'kesehatan',
        'sains',
        'bisnis',
        'bisnis kecil',
        'media',
        'pasar',
        'seni',
        'desain',
        'musik',
        'tari',
        'film',
        'teater',
        'golf',
        'sepakbola',
        'baseball',
        'hoki'
    ]

    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False