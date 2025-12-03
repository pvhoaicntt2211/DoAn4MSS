# Hướng dẫn train 4-stem model trên Google Colab

## Tổng quan

Hướng dẫn này giúp bạn train model U-Net để tách nhạc thành **4 stems**: Vocals, Drums, Bass, Other trên Google Colab với GPU miễn phí.

## Chuẩn bị

### 1. Yêu cầu
- Tài khoản Google (để dùng Colab và Drive)
- Repository này đã push lên GitHub
- Kết nối internet ổn định (để download dataset ~30GB)

### 2. Dataset MUSDB18-HQ

**Tin tốt:** Notebook hiện hỗ trợ **tự động download** MUSDB18-HQ từ Zenodo!

Có 2 cách để có dữ liệu:

**Cách 1: Tự động download trong Colab (Khuyến nghị)**
- Notebook sẽ tự động tải và giải nén MUSDB18-HQ
- Không cần chuẩn bị gì trước
- Tốn ~30-60 phút tải lần đầu

**Cách 2: Upload từ Drive (nếu đã có sẵn)**
- Nếu đã download MUSDB18-HQ trước đó
- Upload lên Google Drive: `MyDrive/MUSDB18_data/`
- Uncomment cell copy từ Drive trong notebook

### 3. Upload code lên GitHub (Khuyến nghị)

```bash
git init
git add .
git commit -m "4-stem MSS project"
git remote add origin https://github.com/<username>/DoAn4MSS.git
git push -u origin main
```

## Sử dụng Colab

### Bước 1: Mở Google Colab
1. Truy cập: https://colab.research.google.com
2. File → Upload notebook
3. Chọn `train_colab.ipynb` từ project

### Bước 2: Kích hoạt GPU
1. Menu: **Runtime → Change runtime type**
2. Hardware accelerator: **GPU** (T4 hoặc cao hơn)
3. Save

### Bước 3: Chạy từng cell theo thứ tự

#### Cell 1: Kiểm tra GPU
```python
!nvidia-smi
```
Verify bạn có GPU T4 hoặc tốt hơn.

#### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Cho phép lưu checkpoint về Drive.

#### Cell 3: Clone code
```python
!git clone https://github.com/<username>/DoAn4MSS.git
%cd DoAn4MSS
```

#### Cell 4: Cài thư viện
```python
!pip install -q torch numpy librosa tqdm soundfile requests
```

#### Cell 5: **Tải MUSDB18-HQ tự động** 🎵
```python
!python scripts/download_musdb18.py --output data/
```
⏱️ Mất 30-60 phút. Progress bar sẽ hiển thị tiến trình.

**Lưu ý:** 
- Nếu bị ngắt kết nối, script sẽ tự động tiếp tục từ chỗ dừng
- Có thể backup data về Drive sau khi download xong để dùng lại

#### Cell 6: Kiểm tra dữ liệu
Verify có đủ 4 stems (vocals, drums, bass, other) + mixture cho mỗi bài.

#### Cell 7: **Train model 4-stem** 🚀
```python
!python train.py \
    --train-dir data/train \
    --valid-dir data/test \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 2
```

**Thời gian:** ~10-15 phút/epoch với GPU T4 → 50 epochs = **8-12 giờ**

**Lưu ý:**
- Loss được tính cho từng stem riêng biệt
- Monitor loss của từng stem: vocals, drums, bass, other
- Giảm batch-size xuống 4 nếu gặp OOM error

#### Cell 8: Backup checkpoint
```python
!cp -r checkpoints /content/drive/MyDrive/MSS_4stem_checkpoints
```
**Quan trọng:** Chạy cell này thường xuyên để không mất checkpoint!

#### Cell 9-11: Test và visualize
- Test tách 4 stems từ 1 bài mẫu
- Nghe từng stem riêng biệt
- Visualize spectrograms

### Bước 4: Download checkpoint về máy

Sau khi train xong:
1. Vào Google Drive: `MyDrive/MSS_4stem_checkpoints/`
2. Download `best_model.pth`
3. Copy vào `checkpoints/` trong project local

## Chạy inference với model đã train

### CLI
```bash
# Tách tất cả 4 stems
python inference.py song.mp3 --checkpoint checkpoints/best_model.pth

# Tách chỉ vocals và drums
python inference.py song.mp3 --stems vocals drums

# Output: song_vocals.wav, song_drums.wav, song_bass.wav, song_other.wav
```

### Web App
```bash
python app.py
```
Truy cập: http://localhost:5000

**Tính năng web app:**
- Upload audio file
- Chọn stems muốn tách (checkboxes)
- Nghe trực tiếp từng stem
- Download riêng từng stem

## So sánh hiệu năng

| Thiết bị | Thời gian/epoch | Tổng 50 epoch | Memory |
|----------|----------------|---------------|---------|
| Laptop CPU (16GB RAM) | 60-90 phút | 50-75 giờ | ~8GB |
| Colab GPU T4 | 10-15 phút | 8-12 giờ | ~15GB |
| Colab GPU V100 | 5-8 phút | 4-7 giờ | ~16GB |

**Lợi ích GPU:** Nhanh hơn **6-9 lần**, miễn phí, không làm nóng laptop.

## Lưu ý quan trọng

### Giới hạn Colab Free
- ⏱️ **12h GPU/session** (tối đa)
- 🔌 **Ngắt sau 90 phút idle** (không có hoạt động)
- 💾 **RAM:** 12-15GB (đủ cho batch_size=8)
- 📊 **Disk:** 100GB temporary storage

### Best Practices

1. **Backup thường xuyên**
   - Chạy cell backup checkpoint sau mỗi 10 epochs
   - Copy data về Drive sau khi download xong

2. **Tránh timeout**
   - Đừng để Colab idle quá lâu
   - Có thể chạy cell đơn giản để giữ session active
   - Sử dụng Colab Pro nếu cần train lâu hơn

3. **Optimize memory**
   - Giảm batch_size nếu gặp OOM: 8 → 4 → 2
   - Không train với batch_size > 16 trên T4
   - Monitor GPU memory: `!nvidia-smi`

4. **Resume training**
   - Nếu bị ngắt, có thể resume từ checkpoint:
   ```python
   # Thêm vào train.py
   --resume checkpoints/best_model.pth
   ```

## Xử lý lỗi thường gặp

### 1. "Runtime disconnected"
**Nguyên nhân:** Session timeout hoặc quá tải
**Giải pháp:**
- Refresh trang, reconnect
- Chạy lại từ cell Mount Drive
- Resume training từ checkpoint backup

### 2. "Out of Memory"
**Nguyên nhân:** Batch size quá lớn hoặc 4-stem tốn nhiều RAM
**Giải pháp:**
```python
# Giảm batch size
!python train.py --batch-size 4  # thay vì 8
```

### 3. "Data not found"
**Nguyên nhân:** Download chưa hoàn thành hoặc path sai
**Giải pháp:**
- Verify data với cell kiểm tra
- Re-run download script với `--force`
- Check path: `data/train/` và `data/test/`

### 4. "Download failed"
**Nguyên nhân:** Kết nối internet không ổn định
**Giải pháp:**
- Re-run download script (sẽ resume từ chỗ dừng)
- Hoặc download manual và upload lên Drive

### 5. "Model checkpoint mismatch"
**Nguyên nhân:** Dùng checkpoint 2-stem cho model 4-stem
**Giải pháp:**
- Train model mới từ đầu
- Hoặc convert checkpoint cũ (advanced)

## Tips nâng cao

### 1. Fine-tune stem weights
Điều chỉnh trọng số loss cho từng stem trong `config.py`:
```python
STEM_WEIGHTS = {
    'vocals': 2.0,  # Ưu tiên vocals
    'drums': 1.0,
    'bass': 1.0,
    'other': 0.5
}
```

### 2. Data augmentation
Thêm augmentation cho training (trong `src/dataset.py`):
- Time stretching
- Pitch shifting
- Random gain

### 3. Monitor training
Visualize loss curves trong TensorBoard:
```python
# Thêm vào train.py
from torch.utils.tensorboard import SummaryWriter
```

### 4. Ensemble models
Train nhiều models với random seeds khác nhau, sau đó ensemble kết quả.

## Kết quả mong đợi

Sau 50 epochs:
- **Validation Loss:** ~0.01-0.03 (tùy dữ liệu)
- **Vocals separation:** Tốt nhất (90%+ chất lượng)
- **Drums separation:** Khá tốt (80-85%)
- **Bass separation:** Tốt (85-90%)
- **Other separation:** Trung bình (75-80%)

## Tài nguyên

- [MUSDB18-HQ Dataset](https://zenodo.org/record/3338373)
- [Colab Documentation](https://colab.research.google.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Liên hệ & Support

Nếu gặp vấn đề, tạo issue trên GitHub repository!

---

**Happy training! 🎵🚀**
