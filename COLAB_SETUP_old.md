# Hướng dẫn train trên Google Colab

## Chuẩn bị

### 1. Upload dữ liệu lên Google Drive
- Nén thư mục `data/` thành `MSS_data.zip`
- Upload lên Google Drive: `MyDrive/MSS_data.zip`

### 2. Upload code lên Drive hoặc GitHub
**Cách 1 (Đơn giản):** Nén toàn bộ code
```powershell
# Tạo file zip chứa code (không bao gồm data, .venv, __pycache__)
Compress-Archive -Path config.py,train.py,inference.py,src -DestinationPath MSS_code.zip
```
Upload `MSS_code.zip` lên Drive

**Cách 2 (Khuyến nghị):** Push lên GitHub
```powershell
git init
git add .
git commit -m "MSS project"
git remote add origin https://github.com/<username>/MSS.git
git push -u origin main
```

## Sử dụng Colab

### Bước 1: Mở Google Colab
- Truy cập: https://colab.research.google.com
- File → Upload notebook
- Chọn `train_colab.ipynb` từ project

### Bước 2: Kích hoạt GPU
- Menu: **Runtime → Change runtime type**
- Hardware accelerator: **GPU** (T4)
- Save

### Bước 3: Chạy từng cell theo thứ tự
- Cell 1: Kiểm tra GPU (`nvidia-smi`)
- Cell 2: Mount Google Drive
- Cell 3: Giải nén code và data
- Cell 4: Cài thư viện
- Cell 5: Kiểm tra data
- Cell 6: **Train** (50 epoch ~8-10 giờ với GPU)
- Cell 7: Backup checkpoint về Drive
- Cell 8: Test inference (optional)

### Bước 4: Tải checkpoint về máy
Sau khi train xong:
1. Vào Google Drive: `MyDrive/MSS_checkpoints/`
2. Download `best_model.pth`
3. Copy vào `d:\DOAN_CNTT4\MSS\checkpoints\`

## Chạy web app với model đã train

```powershell
# Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# Chạy web server
python app.py
```

Truy cập: http://localhost:5000

## So sánh tốc độ

| Thiết bị | Thời gian/epoch | Tổng 50 epoch |
|----------|----------------|---------------|
| Laptop CPU (16GB RAM) | 60-90 phút | 50-75 giờ |
| Colab GPU (T4) | 10-15 phút | 8-12 giờ |

**Lợi ích GPU:** Nhanh hơn **6-9 lần**, miễn phí, không làm nóng laptop.

## Lưu ý quan trọng

- ⏱️ Colab free: 12h GPU/session, ngắt sau 90 phút idle
- 💾 Nhớ backup checkpoint về Drive thường xuyên
- 🔄 Có thể reconnect và tiếp tục train nếu bị ngắt
- 📊 Theo dõi loss trong output để đảm bảo train đúng

## Xử lý lỗi thường gặp

**Lỗi "Runtime disconnected":**
- Refresh trang, reconnect
- Chạy lại từ cell mount Drive
- Load checkpoint cũ để tiếp tục train (nếu có)

**Lỗi "Out of memory":**
- Giảm batch size: `--batch-size 8` thay vì 16

**Data không load được:**
- Kiểm tra đường dẫn trong cell giải nén
- Verify data structure: mỗi thư mục phải có `mixture.wav` và `vocals.wav`
