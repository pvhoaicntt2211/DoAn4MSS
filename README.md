# 🎵 4-Stem Music Source Separation

Deep Learning project để tách nhạc thành 4 stems sử dụng U-Net architecture: **Vocals, Drums, Bass, Other**

## ✨ Tính năng chính

- 🎤 **4-Stem Separation**: Tách vocals, drums, bass, và other instruments
- 🤖 **U-Net Architecture**: Deep learning model tối ưu cho audio separation
- 🚀 **Auto Dataset Download**: Tự động tải MUSDB18-HQ từ Zenodo
- 🌐 **Web Interface**: Giao diện web thân thiện để tách nhạc
- 📊 **Per-Stem Metrics**: Theo dõi loss riêng cho từng stem
- ⚡ **GPU Training**: Hỗ trợ train trên Colab với GPU miễn phí
- 🎛️ **Selective Separation**: Chọn stems cần tách (không bắt buộc tách tất cả)

## 🎯 Demo

### Input
```
song.mp3 (mixture)
```

### Output
```
song_vocals.wav  → 🎤 Giọng hát
song_drums.wav   → 🥁 Trống
song_bass.wav    → 🎸 Bass
song_other.wav   → 🎹 Nhạc cụ khác
```

## 📋 Yêu cầu hệ thống

### Minimum
- Python 3.8+
- 8GB RAM (CPU inference)
- 5GB disk space

### Recommended
- Python 3.8+
- 16GB RAM
- NVIDIA GPU với 8GB+ VRAM (training)
- 50GB disk space (dataset + checkpoints)

## 🚀 Cài đặt nhanh

### 1. Clone repository
```bash
git clone https://github.com/pvhoaicntt2211/DoAn4MSS.git
cd DoAn4MSS
```

### 2. Tạo virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 📥 Download Dataset

### Tự động (Khuyến nghị)
```bash
python scripts/download_musdb18.py --output data/
```

Script sẽ:
- ✅ Tải MUSDB18-HQ từ Zenodo (~30GB)
- ✅ Giải nén tự động
- ✅ Tổ chức cấu trúc thư mục
- ✅ Hiển thị progress bar
- ✅ Skip nếu data đã tồn tại

### Manual
Nếu muốn tải thủ công:
1. Download MUSDB18-HQ: https://zenodo.org/record/3338373
2. Giải nén vào `data/`
3. Đảm bảo cấu trúc:
```
data/
├── train/
│   ├── Song1/
│   │   ├── vocals.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   ├── other.wav
│   │   └── mixture.wav
│   └── ...
└── test/
    └── ...
```

## 🎓 Training

### Local (GPU recommended)
```bash
python train.py \
    --train-dir data/train \
    --valid-dir data/test \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4
```

### Google Colab (Khuyến nghị cho người không có GPU)
1. Upload `train_colab.ipynb` lên Colab
2. Kích hoạt GPU (Runtime → Change runtime type → GPU)
3. Chạy từng cell theo thứ tự
4. Dataset sẽ tự động download

Chi tiết: [COLAB_SETUP.md](COLAB_SETUP.md)

### Training Options

```bash
# Điều chỉnh batch size nếu OOM
python train.py --batch-size 4

# Train nhiều workers (Linux)
python train.py --num-workers 4

# Custom learning rate
python train.py --lr 5e-5

# Custom checkpoint directory
python train.py --checkpoint-dir my_checkpoints/
```

### Monitoring Training

Loss được report riêng cho từng stem:
```
Epoch 1/50
Training Loss: 0.0234
  vocals: 0.0198
  drums: 0.0256
  bass: 0.0243
  other: 0.0239
Validation Loss: 0.0187
  vocals: 0.0165
  drums: 0.0201
  bass: 0.0189
  other: 0.0193
✅ New best model saved
```

## 🎵 Inference

### Command Line

```bash
# Tách tất cả 4 stems
python inference.py song.mp3

# Output vào thư mục khác
python inference.py song.mp3 --outdir my_outputs/

# Tách chỉ vocals và drums
python inference.py song.mp3 --stems vocals drums

# Sử dụng checkpoint khác
python inference.py song.mp3 --checkpoint my_model.pth
```

### Python API

```python
from inference import separate_file

# Tách tất cả stems
output_paths = separate_file(
    input_path="song.mp3",
    output_dir="outputs/",
    checkpoint_path="checkpoints/best_model.pth"
)

# output_paths = {
#     'vocals': 'outputs/song_vocals.wav',
#     'drums': 'outputs/song_drums.wav',
#     'bass': 'outputs/song_bass.wav',
#     'other': 'outputs/song_other.wav'
# }

# Tách chỉ một số stems
output_paths = separate_file(
    input_path="song.mp3",
    stems_to_separate=['vocals', 'drums']
)
```

## 🌐 Web Application

### Chạy web app
```bash
python app.py
```

Truy cập: http://localhost:5000

### Tính năng Web UI

- 📤 Upload audio files (wav, mp3, m4a, flac, ogg)
- ☑️ Chọn stems cần tách (checkboxes)
- 🎧 Nghe trực tiếp từng stem trong browser
- 💾 Download riêng từng stem
- 🎨 Giao diện đơn giản, dễ sử dụng

### Web App Screenshots

*Upload và chọn stems:*
- Chọn file audio
- Tick các stems muốn tách
- Click "Tách ngay"

*Kết quả:*
- 4 audio players cho từng stem
- Buttons download riêng
- Option quay lại tách bài khác

## ⚙️ Configuration

File `config.py` chứa các tham số:

```python
# Model & Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50

# Audio Processing
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SECONDS = 5

# 4-Stem Configuration
STEMS = ['vocals', 'drums', 'bass', 'other']
STEM_WEIGHTS = {
    'vocals': 1.0,
    'drums': 1.0,
    'bass': 1.0,
    'other': 1.0
}
```

### Điều chỉnh Stem Weights

Nếu muốn ưu tiên một stem hơn các stem khác:

```python
# Ưu tiên vocals và drums
STEM_WEIGHTS = {
    'vocals': 2.0,  # 2x loss weight
    'drums': 1.5,   # 1.5x loss weight
    'bass': 1.0,
    'other': 0.5    # 0.5x loss weight
}
```

## 📊 Model Architecture

```
U-Net Architecture for 4-Stem Separation

Input: Mixture Spectrogram (1, F, T)
       ↓
[Encoder]
  Conv + BN + ReLU (64)
  ↓ MaxPool
  Conv + BN + ReLU (128)
  ↓ MaxPool
  Conv + BN + ReLU (256)
  ↓ MaxPool
  Conv + BN + ReLU (512)

[Decoder]
  ↑ UpConv + Skip Connection
  Conv + BN + ReLU (256)
  ↑ UpConv + Skip Connection
  Conv + BN + ReLU (128)
  ↑ UpConv + Skip Connection
  Conv + BN + ReLU (64)
       ↓
Output: 4 Masks (4, F, T) → Sigmoid
       ↓
4 Separated Stems: vocals, drums, bass, other
```

## 📁 Project Structure

```
DoAn4MSS/
├── app.py                  # Flask web application
├── config.py               # Configuration constants
├── train.py                # Training script
├── inference.py            # Inference script
├── gui.py                  # Optional GUI (if exists)
├── requirements.txt        # Python dependencies
├── train_colab.ipynb      # Google Colab notebook
├── COLAB_SETUP.md         # Colab setup guide
├── README.md              # This file
│
├── scripts/
│   └── download_musdb18.py  # Auto dataset downloader
│
├── src/
│   ├── __init__.py
│   ├── model.py            # U-Net model definition
│   ├── dataset.py          # MUSDB dataset loader
│   └── utils.py            # Utility functions
│
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Results page (4 stems)
│
├── checkpoints/            # Saved model weights
├── data/                   # MUSDB18-HQ dataset
│   ├── train/
│   └── test/
└── outputs/                # Inference outputs
```

## 🧪 Testing

### Test Dataset Download
```bash
python scripts/download_musdb18.py --output test_data/ --force
```

### Test Inference
```bash
# Tải 1 file audio mẫu và test
python inference.py test_audio.mp3 --outdir test_output/
```

### Test Web App
```bash
python app.py
# Mở browser, upload file, verify 4 stems
```

## 🔍 Troubleshooting

### Out of Memory (OOM)
```bash
# Giảm batch size
python train.py --batch-size 4  # hoặc 2

# Giảm chunk duration trong config.py
CHUNK_SECONDS = 3  # thay vì 5
```

### Slow Training
```bash
# Tăng workers (Linux/Mac)
python train.py --num-workers 4

# Giảm validation frequency
# (sửa trong train.py, validate mỗi N epochs thay vì mỗi epoch)
```

### Poor Separation Quality
1. Train thêm epochs (100+)
2. Điều chỉnh stem weights
3. Thử learning rate khác nhau
4. Thêm data augmentation

### Web App không chạy
```bash
# Check port conflicts
python app.py  # Default: port 5000

# Or specify port
flask run --port 8080
```

## 📈 Performance Benchmarks

### Training Time (50 epochs)

| Hardware | Time/Epoch | Total | Memory |
|----------|------------|-------|---------|
| CPU (16-core) | 60-90 min | ~50-75h | 8GB |
| GPU T4 (Colab) | 10-15 min | 8-12h | 15GB |
| GPU V100 | 5-8 min | 4-7h | 16GB |
| GPU A100 | 3-5 min | 2-4h | 20GB |

### Inference Time (per song ~3-4 min)

| Hardware | Time |
|----------|------|
| CPU | 30-45s |
| GPU T4 | 3-5s |
| GPU V100 | 2-3s |

## 🎓 Technical Details

### Loss Function
```python
# Multi-stem L1 Loss
loss = Σ (STEM_WEIGHTS[stem] × L1(predicted_stem, target_stem))
```

### Mask Application
```python
# Each stem: mask × mixture
vocals = vocals_mask × mixture_spectrogram
drums = drums_mask × mixture_spectrogram
bass = bass_mask × mixture_spectrogram
other = other_mask × mixture_spectrogram
```

### Audio Processing
- Sample Rate: 44.1kHz
- STFT: n_fft=2048, hop_length=512
- Training chunks: 5 seconds
- Phase reconstruction: Original mixture phase

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📝 License

This project is for educational purposes.

MUSDB18-HQ dataset: CC BY-NC-SA 4.0 License

## 🙏 Acknowledgments

- MUSDB18-HQ dataset by Rafii et al.
- U-Net architecture inspired by Ronneberger et al.
- PyTorch framework
- librosa for audio processing

## 📧 Contact

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/pvhoaicntt2211/DoAn4MSS/issues)
- Project maintainer: pvhoaicntt2211

## 🔗 References

1. [MUSDB18-HQ Dataset](https://zenodo.org/record/3338373)
2. [U-Net Paper](https://arxiv.org/abs/1505.04597)
3. [Music Source Separation Survey](https://arxiv.org/abs/2010.10671)

---

**Made with ❤️ using PyTorch and librosa**

⭐ Star this repo if you find it useful!
