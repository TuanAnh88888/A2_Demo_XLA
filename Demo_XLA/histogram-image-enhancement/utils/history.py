"""
Module: utils/history.py
Mục đích:
1. Quản lý toàn bộ vòng đời trạng thái chỉnh sửa ảnh (Undo / Redo / Reset / History Log).
2. Đảm bảo quy tắc kiến trúc tối thượng:
   - original_image: Ảnh gốc không đổi
   - current_image: Luôn nhận kết quả tích lũy sau mỗi thao tác [ÁP DỤNG]
   - undo_stack & redo_stack: Hỗ trợ hoàn tác và làm lại mượt mà
   - Quản lý bộ nhớ RAM thông minh (giới hạn tối đa 20 bước).
"""

import datetime
import numpy as np


MAX_HISTORY_STEPS = 20


def init_image_state(session_state, original_img, filename="image"):
    """
    Khởi tạo toàn bộ trạng thái trong session_state khi upload hoặc chọn ảnh mới.
    """
    img_copy = original_img.copy()
    session_state["original_image"] = img_copy
    session_state["current_image"] = img_copy.copy()
    session_state["preview_image"] = None
    session_state["image_name"] = filename
    session_state["undo_stack"] = []
    session_state["redo_stack"] = []
    session_state["history"] = [
        {
            "step": 0,
            "name": "Ảnh gốc (Original)",
            "params": {},
            "image": img_copy.copy(),
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        }
    ]


def apply_step(session_state, new_img, operation_name, params=None):
    """
    Áp dụng một thao tác chỉnh sửa thành công:
    - Lưu trạng thái hiện tại vào undo_stack
    - Xóa redo_stack
    - Cập nhật current_image thành new_img
    - Ghi vào history log
    - Xóa preview_image tạm thời
    """
    if params is None:
        params = {}

    current_img = session_state.get("current_image")
    if current_img is None:
        return

    # 1. Đẩy ảnh hiện tại vào undo_stack (sao chép độc lập)
    if "undo_stack" not in session_state:
        session_state["undo_stack"] = []
    session_state["undo_stack"].append({
        "image": current_img.copy(),
        "name": session_state["history"][-1]["name"] if session_state.get("history") else "Previous"
    })

    # Giới hạn kích thước undo_stack để tránh tràn RAM
    if len(session_state["undo_stack"]) > MAX_HISTORY_STEPS:
        session_state["undo_stack"].pop(0)

    # 2. Xóa sạch redo_stack khi có thao tác mới
    session_state["redo_stack"] = []

    # 3. Cập nhật current_image mới
    session_state["current_image"] = new_img.copy()
    session_state["preview_image"] = None

    # 4. Lưu vào lịch sử các bước
    if "history" not in session_state:
        session_state["history"] = []

    next_step = len(session_state["history"])
    session_state["history"].append({
        "step": next_step,
        "name": operation_name,
        "params": params,
        "image": new_img.copy(),
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })

    # Giới hạn lịch sử
    if len(session_state["history"]) > MAX_HISTORY_STEPS + 1:
        session_state["history"].pop(0)


def undo_step(session_state):
    """
    Hoàn tác (Undo): Quay lại bước trước đó.
    """
    if not session_state.get("undo_stack"):
        return False

    current_img = session_state.get("current_image")
    last_hist_name = session_state["history"][-1]["name"] if session_state.get("history") else "Current"

    # Đưa trạng thái hiện tại vào redo_stack
    if "redo_stack" not in session_state:
        session_state["redo_stack"] = []
    session_state["redo_stack"].append({
        "image": current_img.copy(),
        "name": last_hist_name
    })

    # Lấy trạng thái trước từ undo_stack
    prev_state = session_state["undo_stack"].pop()
    session_state["current_image"] = prev_state["image"].copy()
    session_state["preview_image"] = None

    # Loại bỏ bước cuối khỏi history
    if len(session_state.get("history", [])) > 1:
        session_state["history"].pop()

    return True


def redo_step(session_state):
    """
    Làm lại (Redo): Khôi phục thao tác vừa Undo.
    """
    if not session_state.get("redo_stack"):
        return False

    current_img = session_state.get("current_image")

    # Đưa trạng thái hiện tại vào undo_stack
    if "undo_stack" not in session_state:
        session_state["undo_stack"] = []
    session_state["undo_stack"].append({
        "image": current_img.copy(),
        "name": session_state["history"][-1]["name"] if session_state.get("history") else "Current"
    })

    # Lấy trạng thái từ redo_stack
    next_state = session_state["redo_stack"].pop()
    session_state["current_image"] = next_state["image"].copy()
    session_state["preview_image"] = None

    # Ghi nhận lại vào history
    next_step = len(session_state["history"])
    session_state["history"].append({
        "step": next_step,
        "name": next_state["name"],
        "params": {},
        "image": next_state["image"].copy(),
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })

    return True


def reset_to_original(session_state):
    """
    Khôi phục hoàn toàn về ảnh gốc (Reset), xóa sạch toàn bộ ngăn xếp và lịch sử.
    """
    orig = session_state.get("original_image")
    if orig is None:
        return

    session_state["current_image"] = orig.copy()
    session_state["preview_image"] = None
    session_state["undo_stack"] = []
    session_state["redo_stack"] = []
    session_state["history"] = [
        {
            "step": 0,
            "name": "Ảnh gốc (Original)",
            "params": {},
            "image": orig.copy(),
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        }
    ]
