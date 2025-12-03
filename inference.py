# Kịch bản để sử dụng mô hình đã huấn luyện tách một bài hát mới

import os
import argparse
import librosa
import numpy as np
import soundfile as sf
import torch

import config
from src.model import UNet


def load_model(checkpoint_path: str, device: str):
	model = UNet(in_channels=1, out_channels=1).to(device)
	state = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state)
	model.eval()
	return model


def separate_file(input_path: str,
				  output_dir: str = "outputs",
				  checkpoint_path: str = os.path.join("checkpoints", "best_model.pth")) -> tuple[str, str]:
	"""
	Tách 1 file audio thành 2 file: vocals và accompaniment.

	Returns
	-------
	(vocals_path, accompaniment_path)
	"""
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

	os.makedirs(output_dir, exist_ok=True)

	device = config.DEVICE
	model = load_model(checkpoint_path, device)

	# Load audio mono để khớp với huấn luyện
	y, sr = librosa.load(input_path, sr=config.SAMPLE_RATE, mono=True)

	# STFT
	stft = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
	mag = np.abs(stft)
	phase = np.angle(stft)

	# Chuẩn bị tensorr
	with torch.no_grad():
		mag_tensor = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(device)
		mask = model(mag_tensor)  # [1,1,F,T], đầu ra sigmoid trong [0,1]
		mask = mask.squeeze(0).squeeze(0).cpu().numpy()

	# Áp dụng mask để tách vocal và nhạc đệm
	vocals_mag = mask * mag
	accomp_mag = (1.0 - mask) * mag

	# Khôi phục miền thời gian với pha của mixture
	vocals_stft = vocals_mag * np.exp(1j * phase)
	accomp_stft = accomp_mag * np.exp(1j * phase)

	vocals = librosa.istft(vocals_stft, hop_length=config.HOP_LENGTH, length=len(y))
	accomp = librosa.istft(accomp_stft, hop_length=config.HOP_LENGTH, length=len(y))

	# Chuẩn hóa để tránh clipping
	def _normalize(wav: np.ndarray):
		m = np.max(np.abs(wav))
		if m > 0:
			wav = wav / (m + 1e-9) * 0.99
		return wav.astype(np.float32)

	vocals = _normalize(vocals)
	accomp = _normalize(accomp)

	base = os.path.splitext(os.path.basename(input_path))[0]
	vocals_path = os.path.join(output_dir, f"{base}_vocals.wav")
	accomp_path = os.path.join(output_dir, f"{base}_accompaniment.wav")

	sf.write(vocals_path, vocals, samplerate=config.SAMPLE_RATE)
	sf.write(accomp_path, accomp, samplerate=config.SAMPLE_RATE)

	return vocals_path, accomp_path


def main():
	parser = argparse.ArgumentParser(description="Tách giọng hát và nhạc đệm từ 1 file âm thanh")
	parser.add_argument("input", help="Đường dẫn file .wav/.mp3 cần tách")
	parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pth"), help="Đường dẫn checkpoint mô hình")
	parser.add_argument("--outdir", default="outputs", help="Thư mục lưu kết quả")
	args = parser.parse_args()

	vocals_path, accomp_path = separate_file(args.input, args.outdir, args.checkpoint)
	print(f"Đã tách xong. Vocals: {vocals_path} | Accompaniment: {accomp_path}")


if __name__ == "__main__":
	main()