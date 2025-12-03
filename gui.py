import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from inference import separate_file


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tách Giọng Hát & Nhạc - MSS")
        self.geometry("520x220")

        self.input_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="outputs")
        self.checkpoint = tk.StringVar(value=os.path.join("checkpoints", "best_model.pth"))

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        tk.Label(self, text="File bài hát:").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.input_path, width=48).grid(row=0, column=1, **pad)
        tk.Button(self, text="Chọn...", command=self._pick_input).grid(row=0, column=2, **pad)

        tk.Label(self, text="Checkpoint:").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.checkpoint, width=48).grid(row=1, column=1, **pad)
        tk.Button(self, text="Chọn...", command=self._pick_ckpt).grid(row=1, column=2, **pad)

        tk.Label(self, text="Thư mục xuất:").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.out_dir, width=48).grid(row=2, column=1, **pad)
        tk.Button(self, text="Chọn...", command=self._pick_outdir).grid(row=2, column=2, **pad)

        self.status = tk.StringVar(value="Sẵn sàng")
        tk.Label(self, textvariable=self.status, fg="#555").grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        tk.Button(self, text="Tách", command=self._run).grid(row=4, column=2, sticky="e", **pad)

    def _pick_input(self):
        path = filedialog.askopenfilename(title="Chọn file audio", filetypes=[
            ("Audio", "*.wav *.mp3 *.flac *.m4a"),
            ("Tất cả", "*.*")
        ])
        if path:
            self.input_path.set(path)

    def _pick_ckpt(self):
        path = filedialog.askopenfilename(title="Chọn checkpoint", filetypes=[("PyTorch", "*.pth")])
        if path:
            self.checkpoint.set(path)

    def _pick_outdir(self):
        path = filedialog.askdirectory(title="Chọn thư mục xuất")
        if path:
            self.out_dir.set(path)

    def _run(self):
        inp = self.input_path.get().strip()
        ckpt = self.checkpoint.get().strip()
        outd = self.out_dir.get().strip() or "outputs"

        if not inp:
            messagebox.showwarning("Thiếu file", "Vui lòng chọn file bài hát")
            return
        if not os.path.isfile(inp):
            messagebox.showerror("Lỗi", "File bài hát không tồn tại")
            return
        if not os.path.isfile(ckpt):
            messagebox.showerror("Lỗi", "Checkpoint không tồn tại")
            return

        self.status.set("Đang tách... vui lòng đợi")
        self.update_idletasks()

        def worker():
            try:
                vocals, accomp = separate_file(inp, outd, ckpt)
                self.status.set("Hoàn tất: \nVocals: " + vocals + "\nAccompaniment: " + accomp)
                messagebox.showinfo("Xong", f"Đã tách xong!\n\nVocals:\n{vocals}\n\nAccompaniment:\n{accomp}")
            except Exception as e:
                self.status.set("Lỗi")
                messagebox.showerror("Lỗi", str(e))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
