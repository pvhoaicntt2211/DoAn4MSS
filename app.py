import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

from inference import separate_file
import config

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
WEB_OUTPUT_BASE = BASE_DIR / "outputs" / "web"
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "best_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WEB_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("MSS_SECRET", "dev-secret")
# Giới hạn 100MB mỗi upload (điều chỉnh nếu cần)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_EXT = {"wav", "mp3", "m4a", "flac", "ogg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"]) 
def upload():
    if "audio" not in request.files:
        flash("Không thấy file audio trong form")
        return redirect(url_for("index"))

    f = request.files["audio"]
    if not f or f.filename == "":
        flash("Bạn chưa chọn file")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("Định dạng không hỗ trợ. Hãy chọn wav/mp3/m4a/flac/ogg")
        return redirect(url_for("index"))

    # Tạo phiên làm việc riêng
    session_id = uuid.uuid4().hex
    session_up_dir = UPLOAD_DIR / session_id
    session_out_dir = WEB_OUTPUT_BASE / session_id
    session_up_dir.mkdir(parents=True, exist_ok=True)
    session_out_dir.mkdir(parents=True, exist_ok=True)

    # Lưu file upload
    original_name = f.filename
    save_path = session_up_dir / original_name
    f.save(str(save_path))

    # Chọn checkpoint (nếu có trường form ckpt)
    ckpt = request.form.get("checkpoint", str(DEFAULT_CKPT))
    if not os.path.isfile(ckpt):
        if DEFAULT_CKPT.is_file():
            ckpt = str(DEFAULT_CKPT)
        else:
            flash("Không tìm thấy checkpoint. Hãy train trước hoặc đặt checkpoints/best_model.pth")
            return redirect(url_for("index"))

    # Get selected stems from form (default to all)
    selected_stems = request.form.getlist('stems')
    if not selected_stems:
        selected_stems = None  # Will default to all stems in separate_file
    
    try:
        output_paths = separate_file(str(save_path), str(session_out_dir), ckpt, selected_stems)
    except Exception as e:
        flash(f"Lỗi khi tách: {e}")
        return redirect(url_for("index"))

    # Create a dict of stem files
    stem_files = {stem: os.path.basename(path) for stem, path in output_paths.items()}

    return render_template(
        "result.html",
        session_id=session_id,
        stem_files=stem_files,
    )


@app.route("/download/<session_id>/<path:filename>")
def download(session_id: str, filename: str):
    target_dir = WEB_OUTPUT_BASE / session_id
    if not target_dir.is_dir():
        return "Session không tồn tại", 404
    return send_from_directory(target_dir, filename, as_attachment=True)


if __name__ == "__main__":
    # Chạy dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
