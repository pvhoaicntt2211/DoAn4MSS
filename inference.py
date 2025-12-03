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
	model = UNet(in_channels=1, out_channels=4).to(device)  # 4 output channels for 4 stems
	state = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state)
	model.eval()
	return model


def separate_file(input_path: str,
				  output_dir: str = "outputs",
				  checkpoint_path: str = os.path.join("checkpoints", "best_model.pth"),
				  stems_to_separate: list = None) -> dict:
	"""
	Tách 1 file audio thành 4 stems: vocals, drums, bass, other.

	Parameters
	----------
	input_path : str
		Path to input audio file
	output_dir : str
		Directory to save output stems
	checkpoint_path : str
		Path to model checkpoint
	stems_to_separate : list, optional
		List of stems to separate. If None, separates all stems.
		Valid options: ['vocals', 'drums', 'bass', 'other']

	Returns
	-------
	dict
		Dictionary mapping stem names to output file paths
	"""
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

	os.makedirs(output_dir, exist_ok=True)

	# Default to all stems if not specified
	if stems_to_separate is None:
		stems_to_separate = config.STEMS

	device = config.DEVICE
	model = load_model(checkpoint_path, device)

	# Load audio mono để khớp với huấn luyện
	y, sr = librosa.load(input_path, sr=config.SAMPLE_RATE, mono=True)

	# STFT
	stft = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
	mag = np.abs(stft)
	phase = np.angle(stft)

	# Chuẩn bị tensor
	with torch.no_grad():
		mag_tensor = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(device)
		masks = model(mag_tensor)  # [1, 4, F, T], 4 masks for 4 stems
		masks = masks.squeeze(0).cpu().numpy()  # [4, F, T]

	# Chuẩn hóa để tránh clipping
	def _normalize(wav: np.ndarray):
		m = np.max(np.abs(wav))
		if m > 0:
			wav = wav / (m + 1e-9) * 0.99
		return wav.astype(np.float32)

	# Apply masks and reconstruct each stem
	base = os.path.splitext(os.path.basename(input_path))[0]
	output_paths = {}

	for i, stem_name in enumerate(config.STEMS):
		if stem_name not in stems_to_separate:
			continue

		# Apply mask for this stem
		stem_mask = masks[i]
		stem_mag = stem_mask * mag

		# Reconstruct audio with original phase
		stem_stft = stem_mag * np.exp(1j * phase)
		stem_audio = librosa.istft(stem_stft, hop_length=config.HOP_LENGTH, length=len(y))

		# Normalize
		stem_audio = _normalize(stem_audio)

		# Save
		stem_path = os.path.join(output_dir, f"{base}_{stem_name}.wav")
		sf.write(stem_path, stem_audio, samplerate=config.SAMPLE_RATE)
		output_paths[stem_name] = stem_path

	return output_paths


def main():
	parser = argparse.ArgumentParser(description="Tách nhạc thành 4 stems: vocals, drums, bass, other")
	parser.add_argument("input", help="Đường dẫn file .wav/.mp3 cần tách")
	parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pth"), help="Đường dẫn checkpoint mô hình")
	parser.add_argument("--outdir", default="outputs", help="Thư mục lưu kết quả")
	parser.add_argument("--stems", nargs='+', choices=['vocals', 'drums', 'bass', 'other'], 
						help="Chọn stems cần tách (mặc định: tất cả)")
	args = parser.parse_args()

	output_paths = separate_file(args.input, args.outdir, args.checkpoint, args.stems)
	
	print(f"✅ Đã tách xong!")
	for stem_name, stem_path in output_paths.items():
		print(f"  {stem_name}: {stem_path}")


if __name__ == "__main__":
	main()