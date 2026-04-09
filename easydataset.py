import os
import sys
import ctypes
import torch
import setuptools
from transformers import AutoModelForCausalLM, AutoTokenizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import json
import shutil
import pickle
import subprocess
from PIL import Image, ImageTk, ImageChops, ImageStat
import io
import requests
import threading
import customtkinter as ctk
from tkinter import messagebox, filedialog
from datetime import datetime
import uuid
import hashlib
from tkinter import Menu
import base64
from queue import Queue, Empty
import random
import numpy as np
import traceback
import types
import tkinter as tk
import tempfile

try:
    from json_repair import repair_json as _json_repair_string
except ImportError:
    _json_repair_string = None


def _trace(step, detail=""):
    """启动/恢复调试用：控制台看执行到哪一步（排查 CTk/Tk 参数错误等）。"""
    try:
        print(f"[TRACE] {step}" + (f" | {detail}" if detail else ""), flush=True)
    except Exception:
        pass


_resume_flow_dbg_last = None


def _resume_flow_dbg_reset():
    """新一次「继续进程」恢复流开始时调用，清零间隔计时。"""
    global _resume_flow_dbg_last
    _resume_flow_dbg_last = None


def _resume_flow_dbg(stage, detail=""):
    """继续进程主路径：打印距上一行的毫秒间隔 + 线程名，用于定位卡住的具体函数。"""
    global _resume_flow_dbg_last
    try:
        now = time.perf_counter()
        last = _resume_flow_dbg_last
        gap_ms = (now - last) * 1000.0 if last is not None else 0.0
        _resume_flow_dbg_last = now
        thr = threading.current_thread().name
        print(
            f"[继续进程-DBG] +{gap_ms:8.1f}ms | {thr:14} | {stage}"
            + (f" | {detail}" if detail else ""),
            flush=True,
        )
    except Exception:
        pass


def _tk_dbg(stage, task_id=None, video_id=None, extra=""):
    """Tk「参数错误」定位：打印当前执行到 UI 的哪一步（配合 _init_step / TclError 日志）。"""
    try:
        print(
            "[TK-DBG] "
            + stage
            + (f" | task={task_id}" if task_id else "")
            + (f" | vid={video_id}" if video_id else "")
            + (f" | {extra}" if extra else ""),
            flush=True,
        )
    except Exception:
        pass


def _log_tcl_error(where, err, video_id=None, task_id=None):
    """打印 TclError / 底层 configure 失败，便于对照行号与控件。"""
    try:
        print(
            f"[TclError] {where}"
            + (f" | task={task_id}" if task_id else "")
            + (f" | vid={video_id!r}" if video_id is not None else "")
            + f" | err={err!r}",
            flush=True,
        )
        traceback.print_exc()
    except Exception:
        pass


def _debug_excepthook(exc_type, exc_value, exc_tb):
    """保证未捕获异常在控制台打出完整栈（含 TclError 中文「参数错误」）。"""
    try:
        print(f"[EXC] {exc_type.__name__}: {exc_value!r}", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
    except Exception:
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass


def _install_thread_excepthook():
    """后台线程未捕获异常也打印完整栈。"""
    if not hasattr(threading, "excepthook"):
        return

    def _hook(args):
        try:
            print("[THREAD-EXC] " + str(args.thread), flush=True)
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            pass

    threading.excepthook = _hook


def _patch_tk_report_callback_exception(root):
    """Tk 事件回调里的异常默认可能只显示在控制台一行；这里强制打印完整 traceback。"""

    def _patched(self, exc, val, tb):
        try:
            print("[TK-CALLBACK-EXC]", repr(val), flush=True)
            traceback.print_exception(exc, val, tb)
        except Exception:
            pass

    try:
        root.report_callback_exception = types.MethodType(_patched, root)
    except Exception:
        pass


# ==================== 相邻帧去重（像素差分，跳过与上一张几乎相同的画面） ====================
FRAME_DEDUP_THUMB = 96
# 差异图 RGB 三通道平均像素差（0～255），低于此视为重复（约 1.8% 全幅变化，可按需调小更严）
FRAME_DEDUP_MEAN_DIFF_MAX = 4.5
# 首轮视频列表：用户若 5 分钟内未对勾选做任何操作，则自动全选并确认（与手动点「点击确认」一致）
VIDEO_LIST_AUTO_CONFIRM_MS = 5 * 60 * 1000
# 视频列表出现后延迟再启动计时，避免 CheckBox 初始化等事件干扰
VIDEO_LIST_AUTO_CONFIRM_ARM_DELAY_MS = 800


def _frames_mean_diff_rgb(img_a, img_b, max_size=FRAME_DEDUP_THUMB):
    """两帧缩略后逐像素平均差异（越小越像）。"""
    a = img_a.convert("RGB")
    b = img_b.convert("RGB")
    a.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    b.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(a, b)
    stat = ImageStat.Stat(diff)
    return sum(stat.mean) / 3.0


def frames_are_near_duplicate(img_a, img_b, mean_diff_max=FRAME_DEDUP_MEAN_DIFF_MAX):
    """与上一张几乎相同则 True，不保存。"""
    try:
        return _frames_mean_diff_rgb(img_a, img_b) <= mean_diff_max
    except Exception:
        return False


# ==================== 安全关闭浏览器（避免 quit 卡死） ====================
def _kill_pid_tree_windows(pid):
    try:
        if not pid:
            return
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
    except Exception:
        pass


def safe_quit_driver(driver, timeout_sec=3):
    """
    尝试在后台线程 quit driver，避免在 UI 线程卡死。
    超时后会尝试 taskkill driver 的 service 进程（Windows）。
    """
    if not driver:
        return

    try:
        pid = None
        try:
            svc = getattr(driver, "service", None)
            proc = getattr(svc, "process", None) if svc else None
            pid = getattr(proc, "pid", None) if proc else None
        except Exception:
            pid = None

        done = threading.Event()

        def _quit():
            try:
                driver.quit()
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=_quit, daemon=True)
        t.start()
        if not done.wait(timeout_sec):
            _kill_pid_tree_windows(pid)
    except Exception:
        pass


def remove_dir_with_retries(path, attempts=8, wait_sec=0.2):
    """尽力删除目录（Windows 文件占用时重试），成功返回 True。"""
    if not path:
        return True
    try:
        attempts = max(1, int(attempts))
    except Exception:
        attempts = 8
    try:
        wait_sec = max(0.05, float(wait_sec))
    except Exception:
        wait_sec = 0.2
    for _ in range(attempts):
        try:
            if not os.path.isdir(path):
                return True
            shutil.rmtree(path, ignore_errors=False)
        except Exception:
            pass
        if not os.path.isdir(path):
            return True
        time.sleep(wait_sec)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    return not os.path.isdir(path)


# ==================== 浏览器系统临时目录清理（非任务缓存） ====================
# Selenium 启动的 Chrome 会在 %TEMP% 等处留下 scoped_dir* 临时用户数据目录（常达百 MB/个）。
# 仅按「目录最后修改时间」删除足够旧的目录，避免删掉正在使用的浏览器配置。


def _iter_system_temp_roots_for_browser_cleanup():
    """收集本机临时目录（Windows 上多为 ...\AppData\Local\Temp；不同用户路径由环境变量决定）。"""
    roots = set()
    for key in ("TEMP", "TMP", "TMPDIR"):
        v = os.environ.get(key)
        if v:
            try:
                v = os.path.normpath(v)
                if os.path.isdir(v):
                    roots.add(v)
            except Exception:
                pass
    try:
        roots.add(os.path.normpath(tempfile.gettempdir()))
    except Exception:
        pass
    return roots


def cleanup_browser_scoped_temp_dirs(min_age_seconds=2700, log_prefix="[浏览器系统缓存清理]"):
    """
    删除临时目录下 Chrome 遗留的 scoped_dir* 文件夹（不触碰任务目录、不删 frame_extract_cache）。
    min_age_seconds：仅删除「整目录 mtime 早于此刻 min_age_seconds 秒」的项，降低误删正在使用配置的风险。
    """
    try:
        min_age_seconds = max(120, int(min_age_seconds))
    except Exception:
        min_age_seconds = 2700
    now = time.time()
    removed = 0
    for root in _iter_system_temp_roots_for_browser_cleanup():
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if not name.startswith("scoped_dir"):
                continue
            path = os.path.join(root, name)
            try:
                if not os.path.isdir(path):
                    continue
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now - mtime < min_age_seconds:
                continue
            try:
                shutil.rmtree(path, ignore_errors=True)
                if not os.path.isdir(path):
                    removed += 1
            except Exception:
                pass
    if removed:
        print(f"{log_prefix} 已移除 {removed} 个过期的 scoped_dir 临时目录（Chrome/Selenium 遗留）", flush=True)
    return removed


# ==================== 线程取消 / 退出 ====================
class TaskThreadManager:
    """全局退出与按任务取消后台线程（不强制杀线程，通过标志位协作退出）。"""
    _shutdown = threading.Event()
    _task_cancel = {}

    @classmethod
    def reset_for_new_run(cls):
        cls._shutdown.clear()
        cls._task_cancel.clear()

    @classmethod
    def shutdown_all(cls):
        cls._shutdown.set()

    @classmethod
    def cancel_task(cls, task_id):
        ev = cls._task_cancel.setdefault(task_id, threading.Event())
        ev.set()

    @classmethod
    def clear_task_cancel(cls, task_id):
        if task_id in cls._task_cancel:
            del cls._task_cancel[task_id]

    @classmethod
    def is_task_cancelled(cls, task_id):
        if cls._shutdown.is_set():
            return True
        ev = cls._task_cancel.get(task_id)
        return ev.is_set() if ev else False

    # 二次/多轮推荐：Event set=运行中，clear=暂停
    _rec_running = {}

    @classmethod
    def ensure_recommendation_event(cls, task_id):
        if task_id not in cls._rec_running:
            ev = threading.Event()
            ev.set()
            cls._rec_running[task_id] = ev
        return cls._rec_running[task_id]

    @classmethod
    def pause_recommendation(cls, task_id):
        ev = cls._rec_running.get(task_id)
        if ev:
            ev.clear()

    @classmethod
    def resume_recommendation(cls, task_id):
        ev = cls._rec_running.get(task_id)
        if ev:
            ev.set()

    @classmethod
    def wait_recommendation_running(cls, task_id):
        ev = cls._rec_running.get(task_id)
        if ev is None:
            return
        while not cls.is_task_cancelled(task_id):
            if ev.wait(0.25):
                return


SESSION_RESUME_FILENAME = "session_resume.json"


def get_session_resume_path():
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath(".")
    return os.path.join(base, "system", SESSION_RESUME_FILENAME)


def read_session_resume_pending():
    p = get_session_resume_path()
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return bool(d.get("pending_resume"))
    except Exception:
        return False


def write_session_resume_pending(flag):
    p = get_session_resume_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"pending_resume": bool(flag), "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def task_has_incomplete_extraction(task):
    """右侧帧提取列表中是否存在非已完成、非已删除的条目。"""
    if not task:
        return False
    fec = task.get("frame_extract_cache") or {}
    for ent in fec.values():
        if not ent:
            continue
        if ent.get("status_text") == "已删除":
            continue
        if ent.get("status") != "已完成":
            return True
    return False


def task_has_user_first_request_message(task_manager, task_id):
    """messages.json 中是否存在任意一条用户消息（即用户已回复过，含第一次说明需要什么画面）。"""
    if not task_manager or not task_id:
        return False
    try:
        msgs = task_manager.load_task_messages(task_id)
    except Exception:
        return False
    for m in msgs or []:
        if m.get("is_user"):
            return True
    return False


def _download_total_from_task_record(task):
    """仅依据 info.json 的 frame_extract_cache 统计条数（冷启动尚无右侧列表 UI 时）。"""
    if not task:
        return 0
    fec = task.get("frame_extract_cache") or {}
    n = 0
    for ent in fec.values():
        if not ent:
            continue
        if ent.get("status_text") == "已删除":
            continue
        if ent.get("video"):
            n += 1
    return n


def task_is_fully_idle_at_goal(task_manager, task):
    """
    已确认首轮且当前列表达到目标数量，且无待续推荐、无未完成帧提取：无需「继续进程」或暂停条。
    未确认首轮前一律不视为「已达目标」。
    """
    if not task_manager or not task:
        return False
    tid = task.get("task_id")
    if not tid:
        return False
    if task.get("recommendation_resume_pending"):
        return False
    if task_has_incomplete_extraction(task):
        return False
    if not task.get("ever_confirmed_to_download"):
        return False
    try:
        tc = int(task.get("target_video_count")) if task.get("target_video_count") is not None else None
    except (TypeError, ValueError):
        tc = None
    if tc is None:
        tc = 20
    cur = _download_total_from_task_record(task)
    return cur >= tc


def task_should_show_session_resume_banner(task, global_pending_file, task_manager=None):
    """仅在「上次通过右上角正常关闭程序后再次启动」时显示「继续进程」：
    依赖 system/session_resume.json 的 pending_resume（关闭时写入），避免「暂停下载」等仅内存/磁盘暂停误显示该按钮。
    另需：用户已发过至少一条需求、任务仍为暂停、且未处于已达目标的空闲态。
    """
    if not task_manager or not task:
        return False
    tid = task.get("task_id")
    if not tid:
        return False
    if not global_pending_file:
        return False
    if not task.get("task_paused"):
        return False
    if not task_has_user_first_request_message(task_manager, tid):
        return False
    if task_is_fully_idle_at_goal(task_manager, task):
        return False
    return True


def compute_any_task_needs_session_resume(task_manager):
    """关闭时写入 session_resume：存在仍需「继续进程」的任务时标记待继续（已达目标空闲的任务不计）。"""
    for t in task_manager.tasks:
        tid = t.get("task_id")
        if not tid:
            continue
        if not task_has_user_first_request_message(task_manager, tid):
            continue
        if task_is_fully_idle_at_goal(task_manager, t):
            continue
        return True
    return False


# ==================== 主题：白色 微信风格 ====================
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ==================== 语言（默认中文，可切英文） ====================
LANGUAGE_PREF_FILENAME = "language.json"

I18N = {
    "zh": {
        "create_task": "+ 创建新任务",
        "task_list": "📋 任务列表",
        "app_title": "EasyDataset",
        "toggle_download": "📥",
        "toggle_chat": "💬",
        "resume_session": "继续进程",
        "search_placeholder_idle": "",
        "search_placeholder_active": "暂停搜索/继续搜索",
        "welcome_text": "你好！我是EasyDataset，请告诉我你想搜集什么视频画面～",
        "ask_video_count": "你需要获取的视频数量是多少？",
        "language_label": "语言 / Language",
        "language_option_zh": "中文",
        "language_option_en": "English",
        "task_default_name": "任务 {time}",
        "task_unnamed": "未命名任务",
        "menu_pin": "📌     置顶",
        "menu_rename": "✏️ 重命名",
        "menu_delete": "🗑️ 删除",
        "rename_dialog_text": "请输入新的任务名称：",
        "rename_dialog_title": "重命名任务",
        "delete_confirm_title": "确认删除",
        "delete_task_confirm": "确定要删除任务「{name}」吗？\n\n此操作将永久删除该任务的所有数据，但会保留已下载文件！",
        "delete_video_confirm": "确定要删除「{name}」吗？\n\n已提取的帧也会被删除。",
        "download_list_title": "📥 帧提取列表",
        "btn_pause_download": "⏸️ 暂停下载",
        "btn_resume_download": "▶️ 继续下载",
        "count_recorded": "已记录，目标视频数量: {count} 个，开始搜索...",
        "count_recorded_no_n": "已记录，开始搜索...",
        "video_keyword_prefix": "关键词：",
        "video_confirm_click": "点击确认",
        "video_confirmed": "已确认",
        "page_prev": "◀ 上一页",
        "page_next": "下一页 ▶",
        "page_prefix": "第",
        "page_suffix": "/{total} 页",
        "page_total_items": " · 共 {count} 条",
        "msg_recommend_hint": "给你推荐一组视频，你可以取消勾选不符合意向的",
        "msg_no_result": "未找到相关视频，请尝试其他关键词。",
        "msg_confirm_first": "请先确认首轮推荐视频（点击「点击确认」或等待自动确认），再输入新的调整需求。",
        "msg_pref_analyzed": "📊 已分析你的偏好：\n\n{analysis}\n\n正在根据偏好推荐更多视频...",
        "msg_auto_search_hint": "根据你的偏好，将自动搜索添加视频到下载列表。如果你有其他需求也可以继续告诉我，比如下载总量调整 等。",
        "msg_search_error": "搜索出错: {err}",
    },
    "en": {
        "create_task": "+ Create New Task",
        "task_list": "📋 Task List",
        "app_title": "EasyDataset",
        "toggle_download": "📥",
        "toggle_chat": "💬",
        "resume_session": "Resume Session",
        "search_placeholder_idle": "",
        "search_placeholder_active": "pause search / resume search",
        "welcome_text": "Hi! I'm EasyDataset. Tell me what video shots you want to collect.",
        "ask_video_count": "How many videos do you want to collect?",
        "language_label": "Language / 语言",
        "language_option_zh": "中文",
        "language_option_en": "English",
        "task_default_name": "Task {time}",
        "task_unnamed": "Untitled Task",
        "menu_pin": "📌     Pin",
        "menu_rename": "✏️ Rename",
        "menu_delete": "🗑️ Delete",
        "rename_dialog_text": "Enter a new task name:",
        "rename_dialog_title": "Rename Task",
        "delete_confirm_title": "Confirm Delete",
        "delete_task_confirm": "Delete task \"{name}\"?\n\nThis will permanently delete all task data, but keep downloaded files.",
        "delete_video_confirm": "Delete \"{name}\"?\n\nExtracted frames will also be deleted.",
        "download_list_title": "📥 Frame Extraction List",
        "btn_pause_download": "⏸️ Pause Download",
        "btn_resume_download": "▶️ Resume Download",
        "count_recorded": "Saved. Target video count: {count}. Starting search...",
        "count_recorded_no_n": "Saved. Starting search...",
        "video_keyword_prefix": "Keyword: ",
        "video_confirm_click": "Confirm",
        "video_confirmed": "Confirmed",
        "page_prev": "◀ Prev",
        "page_next": "Next ▶",
        "page_prefix": "Page",
        "page_suffix": "/{total}",
        "page_total_items": " · Total {count}",
        "msg_recommend_hint": "Here is a recommended list. Uncheck any videos that do not match your intent.",
        "msg_no_result": "No relevant videos found. Please try other keywords.",
        "msg_confirm_first": "Please confirm the first recommendation list (click Confirm or wait for auto-confirm) before sending new adjustment requests.",
        "msg_pref_analyzed": "📊 Preference analysis completed:\n\n{analysis}\n\nRecommending more videos based on your preference...",
        "msg_auto_search_hint": "Based on your preference, videos will be auto-searched and added to the download list. You can keep sending updates, e.g. adjust total count.",
        "msg_search_error": "Search error: {err}",
    },
}


# ==================== 路径配置 ====================
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(".")


def get_system_path():
    return os.path.join(get_base_path(), "system")


def get_download_path():
    return os.path.join(get_base_path(), "download")


def get_language_pref_path():
    return os.path.join(get_system_path(), LANGUAGE_PREF_FILENAME)


def get_saved_language():
    p = get_language_pref_path()
    try:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            lang = str(d.get("lang") or "zh").strip().lower()
            if lang in I18N:
                return lang
    except Exception:
        pass
    return "zh"


def setup_paths():
    system_path = get_system_path()
    os.makedirs(system_path, exist_ok=True)
    cache_path = os.path.join(system_path, "cache")
    os.makedirs(cache_path, exist_ok=True)
    tasks_path = os.path.join(system_path, "tasks")
    os.makedirs(tasks_path, exist_ok=True)
    # Selenium/Chrome 用户数据与磁盘缓存统一落在此目录，避免散落在系统 %TEMP% 的 scoped_dir*
    browser_temp_root = os.path.join(system_path, "temp")
    try:
        if os.path.isdir(browser_temp_root):
            remove_dir_with_retries(browser_temp_root, attempts=12, wait_sec=0.2)
    except Exception:
        pass
    os.makedirs(browser_temp_root, exist_ok=True)

    download_path = get_download_path()
    os.makedirs(download_path, exist_ok=True)

    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["TOKENIZER_PARALLELISM"] = "false"
    return system_path, tasks_path, download_path, browser_temp_root


system_path, tasks_path, download_path, browser_temp_root = setup_paths()

# ==================== 模型加载 ====================
print("正在加载模型...")
model_path = system_path

_load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
_load_common = dict(
    device_map="auto" if torch.cuda.is_available() else None,
    local_files_only=True,
    trust_remote_code=True,
)
try:
    qwen_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=_load_dtype, **_load_common)
except TypeError:
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=_load_dtype, **_load_common
    )

qwen_tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)
print("✅ 模型加载完成")
_trace("model_load_done", model_path)

# 帧提取并行度：每个任务最多同时占用这么多台「帧提取浏览器」（独立进程，各开自己的标签页；单 driver 不能多线程共用）
MAX_PARALLEL_EXTRACT_BROWSERS = 3


# ==================== 任务级双浏览器管理器 ====================
class TaskBrowserManager:
    """为每个任务管理独立的浏览器驱动（帧提取可多台并行，一台用于搜索）"""
    _instances = {}  # {task_id: TaskBrowserManager}
    _lock = threading.Lock()

    def __new__(cls, task_id):
        with cls._lock:
            if task_id not in cls._instances:
                instance = super().__new__(cls)
                instance.task_id = task_id
                # 帧提取：多槽位池（每槽一个独立 WebDriver，实现真并行）
                instance._extract_slots = []  # [{"driver": d, "busy": bool}, ...]
                instance._extract_cond = None  # 在 _lock 创建后绑定
                instance._search_driver = None  # 搜索推荐专用浏览器
                # 必须用 RLock：get_*_driver 在已持锁时会调用 close_*_driver / close_task_drivers 再次加锁，普通 Lock 会死锁
                instance._lock = threading.RLock()
                instance._extract_cond = threading.Condition(instance._lock)
                cls._instances[task_id] = instance
            return cls._instances[task_id]

    def is_driver_alive(self, driver):
        """检查驱动是否存活"""
        if not driver:
            return False
        try:
            driver.current_url
            return True
        except:
            return False

    def borrow_extract_driver(self):
        """占用一台帧提取浏览器（无空闲且未达上限则等待）；用毕必须 release_extract_driver。
        注意：create_browser_driver() 可能阻塞数秒，禁止在持有 _extract_cond 时调用，否则 release_extract_driver
        无法获锁，预热线程与主线程会互相卡死（表现为卡在 2/3、3/3 等）。"""
        while True:
            need_create = False
            with self._extract_cond:
                cleaned = []
                for slot in self._extract_slots:
                    if slot.get("busy"):
                        cleaned.append(slot)
                        continue
                    d = slot.get("driver")
                    if d is None:
                        continue
                    if self.is_driver_alive(d):
                        cleaned.append(slot)
                    else:
                        try:
                            safe_quit_driver(d, timeout_sec=3)
                        except Exception:
                            pass
                self._extract_slots[:] = cleaned

                for slot in self._extract_slots:
                    if slot.get("busy"):
                        continue
                    d = slot.get("driver")
                    if d and self.is_driver_alive(d):
                        slot["busy"] = True
                        return d

                if len(self._extract_slots) < MAX_PARALLEL_EXTRACT_BROWSERS:
                    need_create = True
                else:
                    self._extract_cond.wait()
                    continue

            if not need_create:
                continue
            new_d = create_browser_driver(self.task_id, "extract")
            if not new_d:
                time.sleep(0.2)
                continue
            with self._extract_cond:
                if len(self._extract_slots) < MAX_PARALLEL_EXTRACT_BROWSERS:
                    self._extract_slots.append({"driver": new_d, "busy": True})
                    print(
                        f"任务 {self.task_id} 创建帧提取浏览器 {len(self._extract_slots)}/{MAX_PARALLEL_EXTRACT_BROWSERS}"
                    )
                    return new_d
            try:
                safe_quit_driver(new_d, timeout_sec=3)
            except Exception:
                pass
            continue

    def release_extract_driver(self, driver):
        """归还 borrow 的帧提取浏览器，供其他线程复用。"""
        if not driver:
            return
        with self._extract_cond:
            for slot in self._extract_slots:
                if slot.get("driver") is driver:
                    slot["busy"] = False
                    self._extract_cond.notify_all()
                    return

    def invalidate_extract_driver(self, driver):
        """该 driver 会话已损坏，从池中移除并唤醒等待者。"""
        if not driver:
            return
        with self._extract_cond:
            kept = []
            for slot in self._extract_slots:
                if slot.get("driver") is driver:
                    try:
                        safe_quit_driver(driver, timeout_sec=3)
                    except Exception:
                        pass
                    continue
                kept.append(slot)
            self._extract_slots[:] = kept
            self._extract_cond.notify_all()

    def get_search_driver(self):
        """获取搜索推荐专用的浏览器驱动（create_browser_driver 在锁外执行，避免长时间阻塞其他线程）。"""
        with self._lock:
            if self._search_driver and not self.is_driver_alive(self._search_driver):
                print(f"任务 {self.task_id} 的搜索浏览器已失效，正在重新创建...")
                self.close_search_driver()
            if self._search_driver is not None:
                return self._search_driver
        new_d = create_browser_driver(self.task_id, "search")
        if not new_d:
            return None
        with self._lock:
            if self._search_driver is None:
                self._search_driver = new_d
                print(f"任务 {self.task_id} 创建搜索浏览器驱动")
                return self._search_driver
            try:
                safe_quit_driver(new_d, timeout_sec=3)
            except Exception:
                pass
            return self._search_driver

    def close_extract_driver(self):
        """关闭本任务所有帧提取浏览器"""
        with self._extract_cond:
            for slot in self._extract_slots:
                d = slot.get("driver")
                if d:
                    try:
                        safe_quit_driver(d, timeout_sec=3)
                    except Exception:
                        pass
            self._extract_slots.clear()
            self._extract_cond.notify_all()

    def close_search_driver(self):
        """关闭搜索推荐专用浏览器"""
        with self._lock:
            if self._search_driver:
                try:
                    safe_quit_driver(self._search_driver, timeout_sec=3)
                except:
                    pass
                self._search_driver = None

    def close_task_drivers(self):
        """关闭任务的所有浏览器驱动"""
        with self._lock:
            self.close_extract_driver()
            self.close_search_driver()

    @classmethod
    def close_all_drivers(cls):
        """关闭所有任务的浏览器驱动"""
        with cls._lock:
            for task_id, manager in cls._instances.items():
                manager.close_task_drivers()
            cls._instances.clear()

    @classmethod
    def detach_task_without_closing_browser(cls, task_id):
        """从管理表移除任务但不关闭浏览器进程（故意泄漏 driver，满足「不关浏览器」）。"""
        with cls._lock:
            if task_id in cls._instances:
                del cls._instances[task_id]


# ==================== 任务管理类 ====================
class TaskManager:
    def __init__(self, tasks_path, download_path):
        self.tasks_path = tasks_path
        self.download_path = download_path
        self.current_task = None
        self.tasks = []
        self.load_all_tasks()
        # 关键修复：避免多线程同时更新 info.json 互相覆盖（二次推荐/提取状态/重试计数等）
        self._task_info_lock = threading.RLock()

    @staticmethod
    def _write_info_json_atomic(info_path, info_obj):
        """先写临时文件再 replace，避免写入中途崩溃导致 info.json 半截损坏。
        Windows 上 os.replace 偶发 WinError 5（杀毒/占用），重试后仍失败则直接写主文件。"""
        tmp_path = info_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(info_obj, f, ensure_ascii=False, indent=2)
        last_err = None
        for attempt in range(15):
            try:
                os.replace(tmp_path, info_path)
                return
            except OSError as e:
                last_err = e
                w = getattr(e, "winerror", None)
                en = getattr(e, "errno", None)
                if w == 5 or en in (13, 16, 11):
                    time.sleep(0.04 * (attempt + 1))
                    continue
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info_obj, f, ensure_ascii=False, indent=2)
        except OSError:
            if last_err:
                raise last_err
            raise
        finally:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _repair_json_text(text):
        """修复常见「几乎合法」的 JSON：BOM、对象/数组里多余的尾部逗号（多轮直到稳定）。"""
        t = text.lstrip()
        if t.startswith("\ufeff"):
            t = t.lstrip("\ufeff")
        t = t.strip()
        for _ in range(10000):
            t2 = re.sub(r",(\s*[}\]])", r"\1", t)
            if t2 == t:
                break
            t = t2
        return t

    @classmethod
    def _load_task_info_raw(cls, info_path, try_bak=True):
        """
        解析任务目录下的 info。顺序：主文件原文 → 主文件修复 → .bak 原文 → .bak 修复。
        返回 (dict|None, source)，source 为 'ok' | 'repaired' | 'from_bak' | 'from_bak_repaired' | None。
        """
        paths = []
        if os.path.isfile(info_path):
            paths.append((info_path, "info.json"))
        if try_bak and os.path.isfile(info_path + ".bak"):
            paths.append((info_path + ".bak", "info.json.bak"))
        for path, label in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError as e:
                print(f"无法读取 {label}：{e}")
                continue
            for want_repair in (False, True):
                blob = cls._repair_json_text(raw) if want_repair else raw
                if want_repair and blob == raw:
                    continue
                try:
                    info = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                if not isinstance(info, dict):
                    continue
                is_bak = label == "info.json.bak"
                if is_bak:
                    return info, "from_bak_repaired" if want_repair else "from_bak"
                return info, "repaired" if want_repair else "ok"
            if _json_repair_string is not None:
                try:
                    fixed = _json_repair_string(raw)
                    info = json.loads(fixed)
                except Exception:
                    pass
                else:
                    if isinstance(info, dict):
                        is_bak = label == "info.json.bak"
                        return info, "from_bak_repaired" if is_bak else "repaired"
        return None, None

    @staticmethod
    def _maybe_refresh_info_bak(info_path):
        """主文件存在且比 .bak 新时更新备份，便于下次主文件写坏时回滚。"""
        bak = info_path + ".bak"
        try:
            if not os.path.isfile(info_path):
                return
            if not os.path.isfile(bak) or os.path.getmtime(info_path) > os.path.getmtime(bak):
                shutil.copy2(info_path, bak)
        except OSError:
            pass

    def refresh_all_info_baks(self):
        """正常退出时调用：把各任务当前主文件同步到 .bak（运行中不写 .bak，避免高频整文件拷贝）。"""
        for task in self.tasks:
            tid = task.get("task_id")
            if not tid:
                continue
            self._maybe_refresh_info_bak(os.path.join(self.tasks_path, tid, "info.json"))

    def load_all_tasks(self):
        self.tasks = []
        if os.path.exists(self.tasks_path):
            for task_folder in os.listdir(self.tasks_path):
                task_path = os.path.join(self.tasks_path, task_folder)
                if os.path.isdir(task_path):
                    info_path = os.path.join(task_path, "info.json")
                    if not os.path.isfile(info_path) and not os.path.isfile(info_path + ".bak"):
                        continue
                    info, src = self._load_task_info_raw(info_path, try_bak=True)
                    if info is None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if os.path.isfile(info_path):
                            try:
                                shutil.copy2(info_path, info_path + f".corrupt.{ts}.main")
                            except OSError:
                                pass
                        if os.path.isfile(info_path + ".bak"):
                            try:
                                shutil.copy2(info_path + ".bak", info_path + f".corrupt.{ts}.bak")
                            except OSError:
                                pass
                        print(
                            f"警告：任务「{task_folder}」的 info.json 与 info.json.bak 均无法解析（可能写入中途被中断）。"
                            f"已在同目录留下 .corrupt.* 副本，可尝试用文本编辑器或在线 JSON 修复工具抢救。"
                        )
                        continue
                    info['task_id'] = task_folder
                    if 'extraction_cache' not in info:
                        info['extraction_cache'] = {}
                    # 新增：帧提取列表专用缓存（包含视频信息+状态，用于重启恢复右侧列表）
                    if 'frame_extract_cache' not in info:
                        info['frame_extract_cache'] = {}
                    if 'seen_video_ids' not in info:
                        info['seen_video_ids'] = []
                    if 'task_paused' not in info:
                        info['task_paused'] = False
                    if 'failed_videos' not in info:
                        info['failed_videos'] = []
                    if 'retry_count' not in info:
                        info['retry_count'] = {}
                    if 'extracting_videos' not in info:
                        info['extracting_videos'] = []
                    if 'ever_confirmed_to_download' not in info:
                        info['ever_confirmed_to_download'] = bool(
                            info.get('frame_extract_cache') and len(info.get('frame_extract_cache') or {}) > 0
                        )
                    if 'recommendation_resume_pending' not in info:
                        info['recommendation_resume_pending'] = False
                    if 'recommendation_resume_payload' not in info:
                        info['recommendation_resume_payload'] = None
                    if 'video_duration_min_sec' not in info:
                        info['video_duration_min_sec'] = None
                    if 'video_duration_max_sec' not in info:
                        info['video_duration_max_sec'] = None
                    if 'pinned_at' not in info:
                        info['pinned_at'] = None
                    if src != "ok":
                        hint = {
                            "repaired": "主文件存在语法问题（如多余尾逗号），已自动修复",
                            "from_bak": "主文件无法解析，已用上次启动时留下的 info.json.bak 恢复",
                            "from_bak_repaired": "已用备份并结合自动修复恢复",
                        }.get(src, "已恢复")
                        print(f"提示：任务「{task_folder}」{hint}，正在写回 info.json …")
                        try:
                            self._write_info_json_atomic(info_path, info)
                        except Exception as e:
                            print(f"警告：写回修复后的 info.json 失败：{e}")
                        else:
                            self._maybe_refresh_info_bak(info_path)
                    else:
                        self._maybe_refresh_info_bak(info_path)
                    self.tasks.append(info)
        # 置顶任务优先（pinned_at 非空），其内再按置顶时间倒序；未置顶任务按创建时间倒序
        self.tasks.sort(
            key=lambda x: (
                1 if x.get("pinned_at") else 0,
                x.get("pinned_at") or "",
                x.get("created_at", ""),
            ),
            reverse=True,
        )

    def get_task_thumbnails_path(self, task_id):
        thumbnails_path = os.path.join(self.tasks_path, task_id, "thumbnails")
        os.makedirs(thumbnails_path, exist_ok=True)
        return thumbnails_path

    def get_task_download_path(self, task_id):
        task_download_path = os.path.join(self.download_path, task_id)
        os.makedirs(task_download_path, exist_ok=True)
        return task_download_path

    def get_video_frames_path(self, task_id, video_title, create=False):
        """
        返回该视频帧目录路径。默认不创建目录，避免仅浏览列表就产生大量空文件夹；
        真正开始提取前再传 create=True（或调用 ensure_video_frames_path）。
        """
        safe_title = self.sanitize_filename(video_title)
        video_folder = os.path.join(self.download_path, task_id, safe_title)
        if create:
            os.makedirs(video_folder, exist_ok=True)
        return video_folder

    def ensure_video_frames_path(self, task_id, video_title):
        """开始写入帧/metadata 前调用，确保目录存在。"""
        return self.get_video_frames_path(task_id, video_title, create=True)

    def sanitize_filename(self, filename):
        """清理文件名中的非法字符，用于文件夹命名"""
        if not filename:
            return "untitled"

        # 1. Windows 文件名非法字符: \ / : * ? " < > |
        filename = re.sub(r'[\\/*?:"<>|]', '', filename)

        # 2. 移除各种引号（中英文）
        filename = filename.replace("'", "").replace('"', "")
        filename = filename.replace("'", "").replace('"', '')
        filename = filename.replace("‘", "").replace("’", "")
        filename = filename.replace("“", "").replace("”", "")

        # 3. 处理中文括号（转换为英文括号更安全，或者直接移除）
        filename = filename.replace("（", "(").replace("）", ")")
        filename = filename.replace("【", "[").replace("】", "]")
        filename = filename.replace("《", "<").replace("》", ">")

        # 4. 移除控制字符（ASCII 0-31）
        filename = ''.join(ch for ch in filename if ord(ch) >= 32 or ch in '.-_ ')

        # 5. 移除可能导致问题的特殊 Unicode 字符（保留常用中文、英文、数字）
        # 保留：字母、数字、中文、空格、点、横线、下划线、括号
        filename = re.sub(r'[^\w\s\u4e00-\u9fff\.\-_\(\)\[\]]', '', filename)

        # 6. 清理空格（首尾去空格，中间多个空格合并为一个）
        filename = filename.strip()
        filename = re.sub(r'\s+', ' ', filename)

        # 7. 限制长度（Windows 路径限制 255，这里取 100 更安全）
        if len(filename) > 100:
            # 尝试在空格处截断
            truncated = filename[:97].rsplit(' ', 1)[0]
            filename = truncated if truncated else filename[:100]

        # 8. 避免以点或空格结尾
        filename = filename.strip('. ')

        # 9. 如果清理后为空，使用默认名称
        if not filename:
            filename = "untitled"

        return filename

    def save_thumbnail(self, task_id, video_id, image):
        """
        原子写入：先写 .tmp 再 replace，避免保存中途崩溃/杀进程留下半截 JPEG。
        下次启动 load_thumbnail → CTkImage 读到坏文件时，Windows 上常报「参数错误」。
        """
        thumbnails_path = self.get_task_thumbnails_path(task_id)
        file_path = os.path.join(thumbnails_path, f"{video_id}.jpg")
        tmp_path = file_path + ".tmp"
        image.save(tmp_path, "JPEG", quality=85)
        last_err = None
        for attempt in range(15):
            try:
                os.replace(tmp_path, file_path)
                return file_path
            except OSError as e:
                last_err = e
                w = getattr(e, "winerror", None)
                en = getattr(e, "errno", None)
                if w == 5 or en in (13, 16, 11):
                    time.sleep(0.04 * (attempt + 1))
                    continue
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise
        try:
            image.save(file_path, "JPEG", quality=85)
        except OSError:
            if last_err:
                raise last_err
            raise
        finally:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        return file_path

    def load_thumbnail(self, task_id, video_id):
        thumbnails_path = self.get_task_thumbnails_path(task_id)
        file_path = os.path.join(thumbnails_path, f"{video_id}.jpg")
        if os.path.exists(file_path):
            try:
                im = Image.open(file_path)
                im.load()
                return im
            except Exception:
                return None
        return None

    def delete_thumbnail(self, task_id, video_id):
        thumbnails_path = self.get_task_thumbnails_path(task_id)
        file_path = os.path.join(thumbnails_path, f"{video_id}.jpg")
        if os.path.exists(file_path):
            os.remove(file_path)

    def delete_video_frames(self, task_id, video_title):
        safe_title = self.sanitize_filename(video_title)
        folder_path = os.path.join(self.download_path, task_id, safe_title)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            return True
        return False

    def get_video_extraction_status(self, task_id, video_id):
        task = self.get_task(task_id)
        if not task:
            return {'status': '等待中', 'progress': 0, 'frame_count': 0, 'status_text': '等待中', 'last_frame_time': -1}

        # 新缓存优先（包含成员+状态）
        frame_cache = task.get('frame_extract_cache', {}) or {}
        if video_id in frame_cache:
            entry = frame_cache.get(video_id) or {}
            return {
                'status': entry.get('status', '等待中'),
                'progress': entry.get('progress', 0),
                'frame_count': entry.get('frame_count', 0),
                'status_text': entry.get('status_text', '等待中'),
                'last_frame_time': entry.get('last_frame_time', -1),
                'updated_at': entry.get('updated_at')
            }

        # 兼容旧 extraction_cache
        extraction_cache = task.get('extraction_cache', {}) or {}
        return extraction_cache.get(video_id, {
            'status': '等待中',
            'progress': 0,
            'frame_count': 0,
            'status_text': '等待中',
            'last_frame_time': -1
        })

    def update_video_extraction_status(self, task_id, video_id, status_text, progress=0, frame_count=0,
                                       last_frame_time=-1):
        task = self.get_task(task_id)
        if not task:
            return False

        if 'extraction_cache' not in task:
            task['extraction_cache'] = {}
        if 'frame_extract_cache' not in task:
            task['frame_extract_cache'] = {}

        if "已完成" in status_text:
            status = "已完成"
        elif "失败" in status_text:
            status = "失败"
        elif "等待中" in status_text:
            status = "等待中"
        elif "提取中" in status_text or "继续提取" in status_text:
            status = "提取中"
        elif "已暂停" in status_text:
            status = "已暂停"
        elif "排队中" in status_text:
            status = "排队中"
        else:
            status = "未知"

        # 旧缓存（兼容）
        task['extraction_cache'][video_id] = {
            'status': status,
            'status_text': status_text,
            'progress': progress,
            'frame_count': frame_count,
            'last_frame_time': last_frame_time,
            'updated_at': datetime.now().isoformat()
        }

        # 新缓存：确保存在 entry（video 会在 add_video_to_queue 写入；这里兜底不影响）
        fec = task.get('frame_extract_cache', {})
        if video_id not in fec or fec.get(video_id) is None:
            fec[video_id] = {'video': None}
        fec[video_id].update({
            'status': status,
            'status_text': status_text,
            'progress': progress,
            'frame_count': frame_count,
            'last_frame_time': last_frame_time,
            'updated_at': datetime.now().isoformat()
        })

        if status == "已完成" and video_id not in task.get('extracted_videos', []):
            if 'extracted_videos' not in task:
                task['extracted_videos'] = []
            task['extracted_videos'].append(video_id)
            if video_id in task.get('failed_videos', []):
                task['failed_videos'].remove(video_id)
            if video_id in task.get('extracting_videos', []):
                task['extracting_videos'].remove(video_id)
        elif status == "提取中" and video_id not in task.get('extracting_videos', []):
            if 'extracting_videos' not in task:
                task['extracting_videos'] = []
            task['extracting_videos'].append(video_id)
        elif status in ["失败", "已暂停", "等待中"] and video_id in task.get('extracting_videos', []):
            task['extracting_videos'].remove(video_id)

        self.update_task_info(task_id, {
            'extraction_cache': task['extraction_cache'],
            'frame_extract_cache': fec,
            'extracted_videos': task.get('extracted_videos', []),
            'failed_videos': task.get('failed_videos', []),
            'extracting_videos': task.get('extracting_videos', [])
        })
        return True

    def mark_video_failed(self, task_id, video_id):
        """标记视频提取失败"""
        task = self.get_task(task_id)
        if task:
            if 'failed_videos' not in task:
                task['failed_videos'] = []
            if video_id not in task['failed_videos']:
                task['failed_videos'].append(video_id)
            if video_id in task.get('extracting_videos', []):
                task['extracting_videos'].remove(video_id)
            self.update_task_info(task_id, {
                'failed_videos': task['failed_videos'],
                'extracting_videos': task.get('extracting_videos', [])
            })
            return True
        return False

    def get_retry_count(self, task_id, video_id):
        """获取视频的重试次数"""
        task = self.get_task(task_id)
        if task:
            retry_count = task.get('retry_count', {})
            return retry_count.get(video_id, 0)
        return 0

    def increment_retry_count(self, task_id, video_id):
        """增加重试次数"""
        task = self.get_task(task_id)
        if task:
            if 'retry_count' not in task:
                task['retry_count'] = {}
            task['retry_count'][video_id] = task['retry_count'].get(video_id, 0) + 1
            self.update_task_info(task_id, {'retry_count': task['retry_count']})
            return task['retry_count'][video_id]
        return 0

    def reset_retry_count(self, task_id, video_id):
        """重置重试次数"""
        task = self.get_task(task_id)
        if task and 'retry_count' in task:
            if video_id in task['retry_count']:
                del task['retry_count'][video_id]
                self.update_task_info(task_id, {'retry_count': task['retry_count']})

    def get_failed_videos(self, task_id):
        """获取失败视频列表"""
        task = self.get_task(task_id)
        if task:
            return task.get('failed_videos', [])
        return []

    def set_task_paused(self, task_id, paused):
        task = self.get_task(task_id)
        if task:
            task['task_paused'] = paused
            self.update_task_info(task_id, {'task_paused': paused})
            return True
        return False

    def is_task_paused(self, task_id):
        task = self.get_task(task_id)
        if task:
            return task.get('task_paused', False)
        return False

    def get_video_frames_info(self, task_id, video_title, video_id):
        video_folder = self.get_video_frames_path(task_id, video_title, create=False)
        metadata_path = os.path.join(video_folder, "metadata.json")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    return {
                        'total_frames': metadata.get('total_frames', 0),
                        'frames': metadata.get('frames', []),
                        'video_duration': metadata.get('video_duration', 0),
                        'last_extracted_time': metadata.get('last_extracted_time', -1),
                        'frame_interval': metadata.get('frame_interval', 3)
                    }
            except:
                pass
        return {'total_frames': 0, 'frames': [], 'video_duration': 0, 'last_extracted_time': -1, 'frame_interval': 3}

    def create_task(self, name=None):
        task_id = str(uuid.uuid4())[:8]
        task_folder = os.path.join(self.tasks_path, task_id)
        os.makedirs(task_folder, exist_ok=True)

        thumbnails_path = os.path.join(task_folder, "thumbnails")
        os.makedirs(thumbnails_path, exist_ok=True)

        task_download_path = self.get_task_download_path(task_id)
        os.makedirs(task_download_path, exist_ok=True)

        if not name:
            lang = get_saved_language()
            tpl = I18N["en"]["task_default_name"] if lang == "en" else I18N["zh"]["task_default_name"]
            name = tpl.format(time=datetime.now().strftime('%m-%d %H:%M'))

        task_info = {
            'task_id': task_id,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'messages': [],
            'selected_videos': [],
            'deselected_videos': [],
            'preferences': None,
            'search_history': [],
            'download_queue': [],
            'extracted_videos': [],
            'extraction_cache': {},
            # 新增：帧提取列表专用缓存（包含视频信息+状态）
            'frame_extract_cache': {},
            'target_video_count': None,
            'seen_video_ids': [],
            'task_paused': False,
            'failed_videos': [],
            'retry_count': {},
            'extracting_videos': [],
            # UI/推荐状态缓存
            'ui_state': {},
            'cached_preferences': {},
            'auto_named': False,
            'ever_confirmed_to_download': False,
            'recommendation_resume_pending': False,
            'recommendation_resume_payload': None,
            'video_duration_min_sec': None,
            'video_duration_max_sec': None,
            'pinned_at': None,
        }

        self._write_info_json_atomic(os.path.join(task_folder, "info.json"), task_info)

        with open(os.path.join(task_folder, "messages.json"), 'w', encoding='utf-8') as f:
            json.dump([], f)

        self.tasks.insert(0, task_info)
        # 与 load_all_tasks 一致：置顶任务始终在最上，新建任务排在置顶任务之后
        self.tasks.sort(
            key=lambda x: (
                1 if x.get("pinned_at") else 0,
                x.get("pinned_at") or "",
                x.get("created_at", ""),
            ),
            reverse=True,
        )
        self.current_task = task_info
        return task_info

    def get_task(self, task_id):
        for task in self.tasks:
            if task['task_id'] == task_id:
                return task
        return None

    def load_task_messages(self, task_id):
        task_path = os.path.join(self.tasks_path, task_id)
        messages_path = os.path.join(task_path, "messages.json")
        if os.path.exists(messages_path):
            with open(messages_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_task_messages(self, task_id, messages):
        task_path = os.path.join(self.tasks_path, task_id)
        messages_path = os.path.join(task_path, "messages.json")
        tmp_path = messages_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, messages_path)

    def update_task_info(self, task_id, updates):
        with self._task_info_lock:
            task_path = os.path.join(self.tasks_path, task_id)
            info_path = os.path.join(task_path, "info.json")
            if not os.path.isfile(info_path) and not os.path.isfile(info_path + ".bak"):
                return
            info, _src = self._load_task_info_raw(info_path, try_bak=True)
            if info is None:
                print(f"警告：任务 {task_id} 的 info.json / .bak 均无法解析，本次更新已跳过。")
                return
            info.update(updates)
            try:
                self._write_info_json_atomic(info_path, info)
            except Exception as e:
                print(f"写入 info.json 失败：{e}")
                raise
            # 不在此处更新 .bak：帧提取等会高频写 info，整文件拷贝备份代价过大
            for task in self.tasks:
                if task['task_id'] == task_id:
                    task.update(updates)
                    break

    def delete_task(self, task_id):
        TaskBrowserManager.detach_task_without_closing_browser(task_id)
        task_path = os.path.join(self.tasks_path, task_id)
        if os.path.exists(task_path):
            shutil.rmtree(task_path)
        self.tasks = [t for t in self.tasks if t['task_id'] != task_id]
        if self.current_task and self.current_task['task_id'] == task_id:
            self.current_task = None

    def rename_task(self, task_id, new_name):
        task = self.get_task(task_id)
        if task:
            self.update_task_info(task_id, {'name': new_name})
            return True
        return False

    def generate_unique_task_name(self, base_name, exclude_task_id=None):
        base_name = (base_name or "").strip()
        if not base_name:
            lang = get_saved_language()
            base_name = I18N["en"]["task_unnamed"] if lang == "en" else I18N["zh"]["task_unnamed"]
        existing = set()
        for t in self.tasks:
            if exclude_task_id and t.get("task_id") == exclude_task_id:
                continue
            name = (t.get("name") or "").strip()
            if name:
                existing.add(name)
        if base_name not in existing:
            return base_name
        i = 2
        while True:
            candidate = f"{base_name} {i}"
            if candidate not in existing:
                return candidate
            i += 1

    def pin_task(self, task_id):
        task = self.get_task(task_id)
        if task:
            ts = datetime.now().isoformat()
            task["pinned_at"] = ts
            self.update_task_info(task_id, {"pinned_at": ts})
            # 立即按同一排序规则刷新内存顺序，确保 UI 立刻生效
            self.tasks.sort(
                key=lambda x: (
                    1 if x.get("pinned_at") else 0,
                    x.get("pinned_at") or "",
                    x.get("created_at", ""),
                ),
                reverse=True,
            )
            return True
        return False

    def add_task_seen_video(self, task_id, video_id):
        task = self.get_task(task_id)
        if task:
            if 'seen_video_ids' not in task:
                task['seen_video_ids'] = []
            if video_id not in task['seen_video_ids']:
                task['seen_video_ids'].append(video_id)
                self.update_task_info(task_id, {'seen_video_ids': task['seen_video_ids']})
                return True
        return False

    def is_task_video_seen(self, task_id, video_id):
        task = self.get_task(task_id)
        if task:
            seen_ids = task.get('seen_video_ids', [])
            return video_id in seen_ids
        return False

    def merge_seen_video_ids_batch(self, task_id, video_ids):
        """批量合并 video_id 到任务级 seen_video_ids（去重），单次写盘。"""
        task = self.get_task(task_id)
        if not task:
            return
        cur = set(task.get("seen_video_ids") or [])
        for vid in video_ids or []:
            if vid:
                cur.add(vid)
        if cur == set(task.get("seen_video_ids") or []):
            return
        task["seen_video_ids"] = list(cur)
        self.update_task_info(task_id, {"seen_video_ids": task["seen_video_ids"]})


# ==================== 浏览器驱动 ====================
# 将窗口放在所有显示器并集之外，避免遮挡桌面；不用 headless，保留真实渲染（截帧/视频仍可用）
BROWSER_WINDOW_WIDTH = 1200
BROWSER_WINDOW_HEIGHT = 800
# 非 Windows 或取虚拟屏失败时的兜底（尽量远离常见单屏区域）
BROWSER_OFFSCREEN_FALLBACK_X = -32000
BROWSER_OFFSCREEN_FALLBACK_Y = -32000


def _windows_virtual_screen_rect():
    """
    返回虚拟桌面矩形 (left, top, width, height)，单位像素，与 SetWindowPos / 多屏几何一致。
    失败或非 Windows 返回 None。
    """
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        SM_XVIRTUALSCREEN = 76
        SM_YVIRTUALSCREEN = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
        left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        w = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        h = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        if w <= 0 or h <= 0:
            return None
        return left, top, w, h
    except Exception:
        return None


def compute_browser_offscreen_top_left(width=BROWSER_WINDOW_WIDTH, height=BROWSER_WINDOW_HEIGHT, margin=24):
    """
    计算浏览器窗口左上角坐标，使 width×height 的窗口与所有显示器区域不相交。
    Windows：在虚拟屏幕并集（所有监视器覆盖的矩形）的左侧外放置，右缘落在 left - margin。
    其它平台：使用固定大负坐标兜底。
    """
    try:
        w = max(1, int(width))
        h = max(1, int(height))
    except (TypeError, ValueError):
        w, h = BROWSER_WINDOW_WIDTH, BROWSER_WINDOW_HEIGHT
    m = max(8, int(margin))
    rect = _windows_virtual_screen_rect()
    if rect is not None:
        vx, vy, _, _ = rect
        x = vx - w - m
        y = vy
        return x, y
    return BROWSER_OFFSCREEN_FALLBACK_X, BROWSER_OFFSCREEN_FALLBACK_Y


def _sanitize_task_id_for_path(task_id):
    if not task_id:
        return "default"
    s = str(task_id)
    for c in '<>:"/\\|?*':
        s = s.replace(c, "_")
    return s[:120] or "default"


def new_browser_profile_directories(task_id, role_prefix="br"):
    """
    为本次启动的浏览器分配独立用户数据目录，位于「项目/system/temp/browser/…」下，
    通过 --user-data-dir / --disk-cache-dir 指定，避免 Chrome 在系统 %TEMP% 生成 scoped_dir*。
    每次新建 driver 使用唯一子目录，避免多进程争用同一 profile。
    """
    safe_tid = _sanitize_task_id_for_path(task_id)
    uid = uuid.uuid4().hex[:12]
    root = os.path.join(browser_temp_root, "browser", safe_tid, f"{role_prefix}_{uid}")
    os.makedirs(root, exist_ok=True)
    disk_cache = os.path.join(root, "disk_cache")
    os.makedirs(disk_cache, exist_ok=True)
    return root, disk_cache


def position_browser_off_screen(driver, width=BROWSER_WINDOW_WIDTH, height=BROWSER_WINDOW_HEIGHT):
    """把浏览器移到所有屏幕之外；各引擎在创建后统一再调一次以纠正启动参数与最终装饰窗尺寸差。"""
    if not driver:
        return
    try:
        driver.set_window_size(width, height)
        try:
            sz = driver.get_window_size()
            aw = int(sz.get("width", width))
            ah = int(sz.get("height", height))
        except Exception:
            aw, ah = width, height
        ox, oy = compute_browser_offscreen_top_left(aw, ah)
        driver.set_window_position(ox, oy)
    except Exception:
        pass


def create_chrome_driver(user_data_dir=None, disk_cache_dir=None):
    try:
        _ox, _oy = compute_browser_offscreen_top_left(BROWSER_WINDOW_WIDTH, BROWSER_WINDOW_HEIGHT)
        options = ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--page-load-strategy=normal')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument(f'--window-position={_ox},{_oy}')
        if user_data_dir:
            ud = os.path.abspath(user_data_dir)
            os.makedirs(ud, exist_ok=True)
            options.add_argument(f'--user-data-dir={ud}')
        if disk_cache_dir:
            dc = os.path.abspath(disk_cache_dir)
            os.makedirs(dc, exist_ok=True)
            options.add_argument(f'--disk-cache-dir={dc}')
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(30)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })
        return driver
    except:
        return None


def create_edge_driver(user_data_dir=None, disk_cache_dir=None):
    try:
        _ox, _oy = compute_browser_offscreen_top_left(BROWSER_WINDOW_WIDTH, BROWSER_WINDOW_HEIGHT)
        options = EdgeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
        options.add_argument(f'--window-position={_ox},{_oy}')
        if user_data_dir:
            ud = os.path.abspath(user_data_dir)
            os.makedirs(ud, exist_ok=True)
            options.add_argument(f'--user-data-dir={ud}')
        if disk_cache_dir:
            dc = os.path.abspath(disk_cache_dir)
            os.makedirs(dc, exist_ok=True)
            options.add_argument(f'--disk-cache-dir={dc}')
        driver = webdriver.Edge(options=options)
        return driver
    except:
        return None


def create_firefox_driver(profile_dir=None):
    try:
        options = FirefoxOptions()
        if profile_dir:
            pd = os.path.abspath(profile_dir)
            os.makedirs(pd, exist_ok=True)
            options.add_argument("-profile")
            options.add_argument(pd)
        driver = webdriver.Firefox(options=options)
        return driver
    except:
        return None


def create_browser_driver(task_id=None, role_prefix="br"):
    tid = task_id if task_id else "_anon"
    user_data_dir, disk_cache_dir = new_browser_profile_directories(tid, role_prefix)
    driver = create_chrome_driver(user_data_dir, disk_cache_dir)
    if driver:
        position_browser_off_screen(driver)
        return driver
    driver = create_edge_driver(user_data_dir, disk_cache_dir)
    if driver:
        position_browser_off_screen(driver)
        return driver
    driver = create_firefox_driver(user_data_dir)
    if driver:
        position_browser_off_screen(driver)
        return driver
    raise Exception("未找到可用的浏览器")


def format_completed_progress_text(completed, list_total, target_total):
    """统一展示：已完成/Completed 已完成数/(列表总数/目标总数)。"""
    try:
        c = max(0, int(completed))
    except Exception:
        c = 0
    try:
        lt = max(0, int(list_total))
    except Exception:
        lt = 0
    try:
        tg = int(target_total) if target_total is not None else None
    except Exception:
        tg = None
    if tg is None or tg <= 0:
        tg = lt
    lang = get_saved_language()
    if lang == "en":
        return f"Completed {c}/({lt}/{tg})"
    return f"已完成 {c}/({lt}/{tg})"


def close_tab_if_present(driver, tab_handle):
    """切换到指定 handle 并关闭该标签（存在且会话有效时）。"""
    if not driver or not tab_handle:
        return
    try:
        if tab_handle in driver.window_handles:
            driver.switch_to.window(tab_handle)
            try:
                driver.close()
            except Exception:
                try:
                    driver.execute_script("window.close();")
                except Exception:
                    pass
    except Exception:
        pass


def close_extra_windows_keep_one(driver, keep_handle):
    """
    关闭除保留页外的所有标签。keep_handle 若已失效则保留列表中的第一个。
    每关一页后重新枚举 handle，避免快照过期。
    """
    if not driver:
        return
    for _ in range(32):
        try:
            handles = list(driver.window_handles)
        except Exception:
            return
        if len(handles) <= 1:
            break
        keep = keep_handle if (keep_handle and keep_handle in handles) else handles[0]
        to_close = None
        for h in handles:
            if h == keep:
                continue
            to_close = h
            break
        if to_close is None:
            break
        try:
            driver.switch_to.window(to_close)
            driver.close()
        except Exception:
            break
    try:
        rest = list(driver.window_handles)
        if not rest:
            return
        keep = keep_handle if (keep_handle and keep_handle in rest) else rest[0]
        driver.switch_to.window(keep)
    except Exception:
        pass


def cleanup_extract_browser_tabs(driver, keep_handle, work_handle):
    """帧提取结束：先关本次工作标签，再清掉其余多余标签。"""
    if not driver:
        return
    try:
        if work_handle:
            close_tab_if_present(driver, work_handle)
        try:
            handles = list(driver.window_handles)
        except Exception:
            return
        if not handles:
            return
        anchor = keep_handle if (keep_handle and keep_handle in handles) else handles[0]
        close_extra_windows_keep_one(driver, anchor)
    except Exception:
        pass


# ==================== YouTube 广告检测（避免截取广告画面） ====================
_YOUTUBE_AD_PLAYING_JS = """
(function(){
  try {
    var w = document.querySelector('ytd-watch-flexy');
    if (w && w.classList && w.classList.contains('ad-showing')) return true;
    var p = document.querySelector('#movie_player, .html5-video-player');
    if (p && p.classList && p.classList.contains('ad-showing')) return true;
    var ov = document.querySelector('.ytp-ad-player-overlay');
    if (ov) {
      var r = ov.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) return true;
    }
    var mod = document.querySelector('.ytp-ad-module');
    if (mod) {
      var r2 = mod.getBoundingClientRect();
      if (r2.width > 0 && r2.height > 0) return true;
    }
    return false;
  } catch (e) { return false; }
})();
"""

_YOUTUBE_TRY_SKIP_AD_JS = """
(function(){
  try {
    var sels = [
      '.ytp-skip-ad-button',
      'button.ytp-ad-skip-button-modern',
      '.ytp-ad-skip-button-container button',
      'ytd-button-renderer#skip-button button',
      '.ytp-ad-skip-button-modern'
    ];
    for (var s = 0; s < sels.length; s++) {
      document.querySelectorAll(sels[s]).forEach(function(b){
        try { b.click(); } catch (e) {}
      });
    }
    return true;
  } catch (e) { return false; }
})();
"""


def _is_youtube_watch_url(url):
    if not url:
        return False
    u = url.lower()
    return "youtube.com/watch" in u or "youtu.be/" in u


# ==================== 视频帧提取功能 ====================
class FrameExtractor:
    def __init__(self, video_url, video_title, video_id, task_id, task_manager, frame_interval=3):
        self.video_url = video_url
        self.video_title = video_title
        self.video_id = video_id
        self.task_id = task_id
        self.task_manager = task_manager
        self.frame_interval = frame_interval
        self.driver = None
        self.should_stop = False
        self.pause_event = threading.Event()
        self.pause_event.set()

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.should_stop = True
        self.pause_event.set()

    def check_paused(self):
        if not self.pause_event.is_set():
            self.pause_event.wait()
        return False

    def extract(self, progress_callback=None):
        driver = None
        original_window = None
        work_window = None
        try:
            try:
                print(
                    f"[EXTRACT-DBG] borrow_extract_driver begin task={self.task_id} vid={self.video_id} thr={threading.current_thread().name}",
                    flush=True,
                )
                driver = TaskBrowserManager(self.task_id).borrow_extract_driver()
                print(
                    f"[EXTRACT-DBG] borrow_extract_driver ok task={self.task_id} vid={self.video_id}",
                    flush=True,
                )
            except Exception as ex:
                if progress_callback:
                    progress_callback(f"❌ 无法获取浏览器驱动: {ex}", -1)
                return None, 0

            self.driver = driver
            video_folder = self.task_manager.get_video_frames_path(self.task_id, self.video_title, create=False)
            metadata_path = os.path.join(video_folder, "metadata.json")

            existing_frames = []
            last_extracted_time = -self.frame_interval
            total_extracted = 0

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        existing_frames = metadata.get('frames', [])
                        if existing_frames:
                            last_extracted_time = existing_frames[-1]['time']
                            total_extracted = len(existing_frames)
                            if progress_callback:
                                progress_callback(
                                    f"检测到已有 {len(existing_frames)} 帧，将从 {last_extracted_time:.1f}s 继续...",
                                    min(30, int(len(existing_frames) * 100 / 1000)))
                except:
                    pass

            if progress_callback:
                progress_callback("正在打开视频页面...", 0)

            original_window = driver.current_window_handle
            before_handles = frozenset(driver.window_handles)
            driver.execute_script("window.open('');")
            time.sleep(1)
            handles_after = driver.window_handles
            new_handles = [h for h in handles_after if h not in before_handles]
            if not new_handles:
                raise Exception("无法打开新标签页")
            work_window = new_handles[-1]
            driver.switch_to.window(work_window)
            driver.get(self.video_url)
            time.sleep(5)

            wait_time = 0
            video_loaded = False
            while wait_time < 45:
                if self.check_paused():
                    if progress_callback:
                        progress_callback(f"⏸️ 已暂停，已提取 {total_extracted} 帧",
                                          int(total_extracted / 1000 * 100) if total_extracted > 0 else 0)
                    driver.close()
                    driver.switch_to.window(original_window)
                    return None, total_extracted
                try:
                    video = driver.find_element(By.CSS_SELECTOR, "video")
                    if video:
                        video_loaded = True
                        break
                except:
                    pass
                time.sleep(2)
                wait_time += 1

            if not video_loaded:
                raise Exception("视频加载超时")

            if _is_youtube_watch_url(self.video_url):
                ad_wait = 0
                max_ad_wait = 180
                while ad_wait < max_ad_wait:
                    if self.should_stop:
                        raise Exception("提取已停止")
                    if self.check_paused():
                        if progress_callback:
                            progress_callback(f"⏸️ 已暂停，已提取 {total_extracted} 帧",
                                              int(total_extracted / 1000 * 100) if total_extracted > 0 else 0)
                        driver.close()
                        driver.switch_to.window(original_window)
                        return None, total_extracted
                    try:
                        if not driver.execute_script("return " + _YOUTUBE_AD_PLAYING_JS.strip()):
                            break
                        driver.execute_script(_YOUTUBE_TRY_SKIP_AD_JS.strip())
                    except Exception:
                        pass
                    if progress_callback:
                        progress_callback("⏳ 正在等待广告结束或跳过…", 10)
                    time.sleep(1)
                    ad_wait += 1

            if progress_callback:
                progress_callback("视频已加载，开始提取帧...", 10)

            duration = None
            for retry in range(3):
                try:
                    duration = driver.execute_script("return document.querySelector('video').duration;")
                    if duration and duration > 0:
                        break
                except:
                    pass
                time.sleep(2)

            if not duration or duration <= 0:
                raise Exception("无法获取视频时长")

            start_time = max(0, last_extracted_time + self.frame_interval)
            if start_time >= duration:
                if progress_callback:
                    progress_callback(f"视频已完成提取，共 {total_extracted} 帧", 100)
                driver.close()
                driver.switch_to.window(original_window)
                return video_folder, total_extracted

            last_saved_pil = None
            if existing_frames:
                try:
                    last_path = existing_frames[-1].get("path")
                    if last_path and os.path.exists(last_path):
                        last_saved_pil = Image.open(last_path).convert("RGB")
                except Exception:
                    last_saved_pil = None

            # 进入主提取循环前再创建目录：避免排队/仅刷新列表产生空文件夹；暂停与断点续提保留已有目录与 metadata
            self.task_manager.ensure_video_frames_path(self.task_id, self.video_title)

            def extract_batch(start_time, batch_size=3):
                # 使用 async + seeked + 双 requestAnimationFrame：避免「时间变了但画面仍是上一关键帧」、
                # 以及 currentTime 已与目标接近时 seeked 不触发的问题。
                # 每帧前 waitIfYoutubeAd：减少中插广告期间截到广告画面（非 YouTube 页检测恒为 false）。
                js_script = f"""
                return (async function() {{
                    var frameInterval = {self.frame_interval};
                    var startTime = {start_time};
                    var batchSize = {batch_size};

                    var frames = [];
                    var canvas = document.createElement('canvas');
                    var ctx = canvas.getContext('2d');

                    async function waitIfYoutubeAd(maxWaitMs) {{
                        var t0 = Date.now();
                        while (Date.now() - t0 < maxWaitMs) {{
                            var inAd = (function() {{
                                try {{
                                    var w = document.querySelector('ytd-watch-flexy');
                                    if (w && w.classList && w.classList.contains('ad-showing')) return true;
                                    var p = document.querySelector('#movie_player');
                                    if (p && p.classList && p.classList.contains('ad-showing')) return true;
                                    var ov = document.querySelector('.ytp-ad-player-overlay');
                                    if (ov) {{
                                        var r = ov.getBoundingClientRect();
                                        if (r.width > 0 && r.height > 0) return true;
                                    }}
                                    var mod = document.querySelector('.ytp-ad-module');
                                    if (mod) {{
                                        var r2 = mod.getBoundingClientRect();
                                        if (r2.width > 0 && r2.height > 0) return true;
                                    }}
                                    return false;
                                }} catch (e) {{ return false; }}
                            }})();
                            if (!inAd) return;
                            try {{
                                document.querySelectorAll('.ytp-skip-ad-button, button.ytp-ad-skip-button-modern, .ytp-ad-skip-button-container button').forEach(function(b) {{
                                    try {{ b.click(); }} catch (e) {{}}
                                }});
                            }} catch (e) {{}}
                            await new Promise(function(r) {{ setTimeout(r, 350); }});
                        }}
                    }}

                    function seekAndWait(video, newTime) {{
                        return new Promise(function(resolve, reject) {{
                            var to = setTimeout(function() {{
                                video.removeEventListener('seeked', onS);
                                reject(new Error('seek 超时'));
                            }}, 15000);
                            function onS() {{
                                clearTimeout(to);
                                video.removeEventListener('seeked', onS);
                                resolve();
                            }}
                            video.addEventListener('seeked', onS, {{ once: true }});
                            video.currentTime = newTime;
                        }});
                    }}

                    async function seekToTime(video, target) {{
                        var dur = video.duration;
                        if (!dur || dur <= 0) return;
                        var t = Math.min(Math.max(0, target), dur - 0.05);
                        if (Math.abs(video.currentTime - t) < 0.03) {{
                            await seekAndWait(video, Math.max(0, t - 0.25));
                        }}
                        await seekAndWait(video, t);
                    }}

                    function drawAfterPaint(video) {{
                        return new Promise(function(resolve, reject) {{
                            requestAnimationFrame(function() {{
                                requestAnimationFrame(function() {{
                                    try {{
                                        canvas.width = video.videoWidth;
                                        canvas.height = video.videoHeight;
                                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                        var imageData = canvas.toDataURL('image/jpeg', 0.85);
                                        resolve(imageData);
                                    }} catch (e) {{
                                        reject(e);
                                    }}
                                }});
                            }});
                        }});
                    }}

                    for (var i = 0; i < batchSize; i++) {{
                        var video = document.querySelector('video');
                        if (!video) {{
                            return {{error: '未找到视频元素'}};
                        }}
                        video.pause();
                        var duration = video.duration;
                        if (!duration || duration <= 0) {{
                            continue;
                        }}
                        var currentTime = startTime + (i * frameInterval);
                        if (currentTime >= duration) break;
                        try {{
                            await waitIfYoutubeAd(120000);
                            await seekToTime(video, currentTime);
                            var imageData = await drawAfterPaint(video);
                            frames.push({{
                                time: video.currentTime,
                                width: canvas.width,
                                height: canvas.height,
                                data: imageData
                            }});
                        }} catch (e) {{
                            console.error('提取帧失败:', e);
                        }}
                    }}

                    return frames;
                }})();
                """
                return driver.execute_script(js_script)

            batch_number = 0
            consecutive_failures = 0

            while start_time < duration:
                if self.check_paused():
                    self.save_progress(video_folder, metadata_path, existing_frames, duration, start_time)
                    if progress_callback:
                        progress_callback(f"⏸️ 已暂停，已提取 {total_extracted} 帧",
                                          int(total_extracted / (duration / self.frame_interval) * 100))
                    driver.close()
                    driver.switch_to.window(original_window)
                    return None, total_extracted

                if self.should_stop:
                    if progress_callback:
                        progress_callback("提取已停止", 0)
                    driver.close()
                    driver.switch_to.window(original_window)
                    return None, total_extracted

                batch_number += 1
                frames_batch = extract_batch(start_time)

                if not frames_batch or len(frames_batch) == 0:
                    consecutive_failures += 1
                    if progress_callback:
                        progress_callback(f"第 {batch_number} 批提取失败 ({consecutive_failures}/3)，尝试重试...",
                                          int(start_time / duration * 100))
                    if consecutive_failures >= 3:
                        raise Exception("连续提取失败次数过多")
                    time.sleep(3)
                    continue
                else:
                    consecutive_failures = 0

                for frame in frames_batch:
                    try:
                        image_data = base64.b64decode(frame['data'].split(',')[1])
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")

                        if last_saved_pil is not None and frames_are_near_duplicate(last_saved_pil, image):
                            if progress_callback:
                                progress_callback(
                                    f"已提取 {total_extracted} 帧，跳过与上一张几乎相同的画面 ({frame['time']:.1f}s)",
                                    int(frame['time'] / duration * 100) if duration else 0,
                                )
                            continue

                        frame_filename = f"frame_{total_extracted + 1:04d}_{frame['time']:.1f}s.jpg"
                        frame_path = os.path.join(video_folder, frame_filename)
                        image.save(frame_path, "JPEG", quality=85)

                        existing_frames.append({
                            'index': total_extracted + 1,
                            'time': frame['time'],
                            'filename': frame_filename,
                            'path': frame_path,
                            'width': frame['width'],
                            'height': frame['height']
                        })
                        total_extracted += 1
                        last_saved_pil = image.copy()
                    except Exception as e:
                        print(f"保存帧失败: {e}")
                        continue

                self.save_progress(video_folder, metadata_path, existing_frames, duration,
                                   frames_batch[-1]['time'] if frames_batch else start_time)

                current_progress = int((frames_batch[-1]['time'] if frames_batch else start_time) / duration * 100)
                if progress_callback:
                    progress_callback(f"已提取 {total_extracted} 帧 ({current_progress}%)", current_progress)

                start_time = frames_batch[-1]['time'] + self.frame_interval if frames_batch else start_time + self.frame_interval
                time.sleep(1.5)

            if progress_callback:
                progress_callback(f"提取完成！共提取 {total_extracted} 帧", 100)

            driver.close()
            driver.switch_to.window(original_window)
            return video_folder, total_extracted

        except Exception as e:
            print(f"提取帧失败: {e}")
            if driver and ("no such window" in str(e) or "invalid session id" in str(e)):
                TaskBrowserManager(self.task_id).invalidate_extract_driver(driver)
            if progress_callback:
                progress_callback(f"提取失败: {str(e)[:50]}", -1)
            return None, 0
        finally:
            try:
                cleanup_extract_browser_tabs(driver, original_window, work_window)
            except Exception:
                pass
            try:
                if driver:
                    TaskBrowserManager(self.task_id).release_extract_driver(driver)
            except Exception:
                pass
            self.driver = None

    def save_progress(self, video_folder, metadata_path, existing_frames, duration, last_time):
        metadata = {
            'video_title': self.video_title,
            'video_id': self.video_id,
            'task_id': self.task_id,
            'extracted_at': datetime.now().isoformat(),
            'total_frames': len(existing_frames),
            'frame_interval': self.frame_interval,
            'video_duration': duration,
            'frames': existing_frames,
            'last_extracted_time': last_time
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


# ==================== 功能函数 ====================
def parse_chinese_duration(duration_str):
    if not duration_str:
        return "未知时长"
    hours = 0
    minutes = 0
    seconds = 0
    hour_match = re.search(r'(\d+)小时', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    minute_match = re.search(r'(\d+)分钟', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    second_match = re.search(r'(\d+)秒钟', duration_str)
    if second_match:
        seconds = int(second_match.group(1))
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    elif minutes > 0 or seconds > 0:
        return f"{minutes}:{seconds:02d}"
    else:
        return "未知时长"


def analyze_with_qwen(prompt, max_tokens=500):
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                            pad_token_id=qwen_tokenizer.eos_token_id)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    response = qwen_tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return response.strip()


def _normalize_anchor_tokens(anchors):
    """核心英文锚词：小写、去重、去空。"""
    out = []
    seen = set()
    for a in anchors or []:
        s = str(a).strip().lower()
        s = re.sub(r"\s+", " ", s)
        if not s or len(s) > 48:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[:12]


def _keyword_contains_any_anchor(keyword, anchors):
    low = (keyword or "").lower()
    for a in anchors or []:
        if a and a in low:
            return True
    return False


def _tokenize_en(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]


def _infer_primary_anchor_from_request(user_request):
    """
    从用户原始输入里推断一个英文主锚词（优先英文单词），
    避免模型锚词漂移导致关键词固定成无关主题。
    """
    s = (user_request or "").strip().lower()
    if not s:
        return None
    toks = _tokenize_en(s)
    if not toks:
        return None
    stop = {
        "a", "an", "the", "to", "for", "of", "in", "on", "with", "and",
        "video", "videos", "footage", "shot", "shots", "please", "want",
    }
    for t in toks:
        if len(t) >= 3 and t not in stop:
            return t
    return toks[0] if toks else None


def _keywords_too_similar(a, b):
    ta = set(_tokenize_en(a))
    tb = set(_tokenize_en(b))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    if union <= 0:
        return False
    return (inter / union) >= 0.78


def _dedup_keywords_by_similarity(keywords, max_n=10):
    out = []
    for k in keywords or []:
        ks = str(k).strip()
        if not ks:
            continue
        if any(_keywords_too_similar(ks, e) for e in out):
            continue
        out.append(ks)
        if len(out) >= max_n:
            break
    return out


def enforce_keywords_contain_core_anchors(keywords, anchors):
    """
    保证每个英文搜索词至少命中一个核心锚词（子串匹配，大小写不敏感）。
    若模型漏掉主体词，则在短语前拼接首要锚词，避免搜成「纯 close up / lighting / texture」这类泛化词。
    """
    anchors = _normalize_anchor_tokens(anchors)
    if not anchors:
        return [str(x).strip() for x in (keywords or []) if str(x).strip()]
    primary = anchors[0]
    out = []
    seen = set()
    for k in keywords or []:
        ks = str(k).strip()
        if not ks:
            continue
        if not _keyword_contains_any_anchor(ks, anchors):
            ks = f"{primary} {ks}".strip()
        low = ks.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(ks)
    return out


def _to_core_or_core_plus_one_word(keywords, anchors, max_n=10):
    """
    严格收敛为：
    - 核心词（如 hand）
    - 核心词 + 1 个英文单词（如 hand closeup）
    不允许超过两词，也不允许发散短语。
    """
    anchors = _normalize_anchor_tokens(anchors)
    primary = anchors[0] if anchors else ""
    out = []
    seen = set()
    for k in keywords or []:
        s = str(k).strip().lower()
        if not s:
            continue
        toks = _tokenize_en(s)
        if not toks:
            continue
        core = primary
        if not core:
            core = toks[0]
        if core not in toks:
            # 不含核心词则强制收敛到核心词
            cand = core
        else:
            if len(toks) == 1:
                cand = core
            else:
                # 仅保留一个修饰词
                mod = ""
                for t in toks:
                    if t != core:
                        mod = t
                        break
                cand = f"{core} {mod}".strip() if mod else core
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
        if len(out) >= max_n:
            break
    if len(out) < max_n and primary:
        # 兜底补词改为更中性，避免固定出现 table/workshop 等强偏词
        mods = [
            "closeup", "detail", "action", "performance", "practice",
            "tutorial", "reference", "cover", "style", "motion",
        ]
        base = [primary] + [f"{primary} {m}" for m in mods]
        for cand in base:
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
            if len(out) >= max_n:
                break
    return out[:max_n]


def _expand_keywords_with_templates(anchors, existing_keywords, target_n=10):
    anchors = _normalize_anchor_tokens(anchors)
    primary = anchors[0] if anchors else ""
    existing = list(existing_keywords or [])
    out = list(existing)
    if not primary:
        return _dedup_keywords_by_similarity(out, max_n=target_n)

    shot = [
        "close up", "extreme close up", "macro", "pov", "overhead", "top down",
        "slow motion", "4k", "cinematic", "studio", "soft light",
    ]
    intent = [
        "reference", "b roll", "footage", "tutorial", "practice", "demonstration",
    ]
    action = [
        "movement", "gesture", "motion", "detail", "texture", "process",
        "craft", "work", "handling",
    ]
    scene = [
        "on table", "on desk", "in workshop", "in studio", "indoors", "natural light",
    ]

    candidates = []
    for s1 in shot:
        candidates.append(f"{primary} {s1} {intent[0]}".strip())
    for a in action:
        candidates.append(f"{primary} {a} {intent[1]}".strip())
        candidates.append(f"{primary} {a} {shot[0]}".strip())
    for sc in scene:
        candidates.append(f"{primary} {shot[2]} {sc}".strip())
        candidates.append(f"{primary} {shot[0]} {sc}".strip())
    candidates.append(f"{primary} detail shot reference")
    candidates.append(f"{primary} close up b roll")
    candidates.append(f"{primary} macro detail b roll")
    candidates.append(f"{primary} pov close up reference")

    # 保证每个候选都包含 anchor（保险）并做相似度去重补全
    candidates = enforce_keywords_contain_core_anchors(candidates, anchors)
    candidates = _dedup_keywords_by_similarity(candidates, max_n=80)
    for cand in candidates:
        if len(out) >= int(target_n):
            break
        if any(_keywords_too_similar(cand, e) for e in out):
            continue
        out.append(cand)
    return _dedup_keywords_by_similarity(out, max_n=target_n)


def extract_core_english_anchors(user_request):
    """
    从用户需求中抽取「必须在 YouTube 英文搜索词里保留」的主体/核心英文词（用于紧扣需求，避免关键词漂移到泛主题）。
    """
    ur = (user_request or "").strip()
    if not ur:
        return []
    prompt = f"""你是视频素材检索助手。用户需求：{ur}

请先理解用户真正要的**画面主体/对象**（例如：中文「手部特写」→ 英文必须围绕 hand/hands/fingers，而不是泛泛的 close-up 教程或灯光/质感类泛词）。

请输出严格 JSON（不要代码块、不要解释），格式如下：
{{"anchors": ["...", "..."]}}

anchors 规则：
- 4～8 个英文单词或极短短语（小写），每个应能作为 YouTube 搜索里的实质限定词。
- 必须包含与用户主体对应的英文（如：手→hand/hands/finger；猫→cat/kitten；海浪→wave/ocean）。
- 可以包含与需求强相关的场景词（如 macro、studio）但不要全是泛词。
- 不要输出无检索价值的词：video、footage、hd、4k、free、background（除非用户明确要）。

只输出 JSON。"""
    raw = analyze_with_qwen(prompt, max_tokens=260)
    anchors = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            anchors = data.get("anchors") or []
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    anchors = data.get("anchors") or []
            except Exception:
                pass
    return _normalize_anchor_tokens(anchors)


def generate_search_keywords(user_request):
    anchors = extract_core_english_anchors(user_request)
    forced = _infer_primary_anchor_from_request(user_request)
    if forced:
        anchors = [forced] + [a for a in anchors if a != forced]
    anchor_lines = "\n".join(f"- {a}" for a in anchors) if anchors else "- （若上表为空，请你先从需求自行确定主体英文词，如 hand/hands/street/city）"
    prompt = f"""用户想要搜索以下类型的视频画面：{user_request}

下面是为该需求提炼的**核心英文锚词**：
{anchor_lines}

请生成 10 个适合直接在 YouTube 搜索框使用的英文关键词。
每个关键词用英文双引号括起来，逗号分隔。只输出关键词列表，不要其它说明。

硬性要求（必须遵守）：
- 关键词必须是英文。
- 只允许两种形式：1) 核心词本身；2) 核心词+一个英文单词。
- 禁止超过 2 个单词，禁止长短语、禁止发散描述。
- 每个关键词必须包含核心锚词（或同根词）。

示例（当需求与「手」相关）： "hand", "hand macro", "hand closeup", "hand detail"

请输出10个英文关键词："""
    result = analyze_with_qwen(prompt, max_tokens=380)
    keywords = re.findall(r'"([^"]+)"', result)
    keywords = _to_core_or_core_plus_one_word(keywords, anchors, max_n=10)
    if len(keywords) < 10:
        keywords = _to_core_or_core_plus_one_word(keywords, anchors, max_n=10)
    return keywords[:10]


def generate_search_keywords_avoid(user_request, avoid_keywords):
    """
    在多轮推荐/侧边栏已搜尽时生成一批与历史搜索词明显不同的英文关键词。
    avoid_keywords：已用过的关键词集合（str）。
    """
    avoid_list = [str(x).strip() for x in (avoid_keywords or []) if str(x).strip()][:60]
    avoid_text = "\n".join(f"- {a}" for a in avoid_list) if avoid_list else "(无历史词)"
    anchors = extract_core_english_anchors(user_request)
    forced = _infer_primary_anchor_from_request(user_request)
    if forced:
        anchors = [forced] + [a for a in anchors if a != forced]
    anchor_lines = "\n".join(f"- {a}" for a in anchors) if anchors else "- （请先从需求确定主体英文词）"
    prompt = f"""用户需求（视频画面素材检索）：{user_request}

**核心英文锚词**（每个新搜索词必须至少包含其中任意一个，大小写不敏感；用于紧扣需求、避免漂移到无关主题）：
{anchor_lines}

以下英文 YouTube 搜索词已经用过或所在路径已搜尽，请不要再输出相同或仅微调同义的词（不要换皮重复）：
{avoid_text}

请另外输出 12 个**全新的**英文关键词（适合 YouTube 搜索）。
每个关键词用英文双引号括起来，逗号分隔。只输出关键词列表，不要其它说明。

硬性要求：
- 只允许两种形式：1) 核心词本身；2) 核心词+一个英文单词。
- 禁止超过 2 个单词，禁止长短语、禁止发散描述。
- **每个关键词都必须包含至少一个核心锚词**（或同根词）。
- 仍然要避开上面「已用过」列表（不要重复或仅改一两个同义词）。
- 禁止输出不含主体的泛化词。

示例： "hand", "hand macro", "hand texture"

请输出 12 个英文关键词："""
    result = analyze_with_qwen(prompt, max_tokens=420)
    keywords = re.findall(r'"([^"]+)"', result)
    keywords = _to_core_or_core_plus_one_word(keywords, anchors, max_n=20)
    seen_lower = {a.lower() for a in avoid_list}
    out = []
    for k in keywords:
        ks = (k or "").strip()
        if not ks:
            continue
        low = ks.lower()
        if low in seen_lower:
            continue
        seen_lower.add(low)
        out.append(ks)
        if len(out) >= 12:
            break
    return out[:12]


def _extract_keywords_and_preferences(user_request):
    """
    把用户长描述拆成：
    - keywords: 用于 YouTube 搜索的短英文短语（10个）
    - preferences: 作为偏好缓存，用于后续判别
    """
    lang = get_saved_language()
    prompt = f"""你是一个YouTube素材检索助手。
用户需求（可能很长）：{user_request}

请输出严格的 JSON（不要代码块，不要额外文字），格式如下：
{{
  "anchors": ["english token or short phrase", "...(共4~8个，小写)"],
  "keywords": ["short english phrase", "...(共10个)"],
  "preferences": {{
    "scene": ["..."],
    "subject": ["..."],
    "camera": ["..."],
    "style": ["..."],
    "avoid": ["..."]
  }},
  "task_name": "不超过12个字的中文任务名"
}}

要求：
- 先输出 anchors：从需求提炼必须出现的主体核心英文词。
- keywords 必须严格是「核心词」或「核心词+一个单词」两种形式，不允许超过2词。
- 每个 keyword 必须包含 anchors 中任意一个词（大小写不敏感）。
- 不要输出长短语，不要发散描述。
- preferences 用中文短语概括即可
- task_name 用中文，尽量像素材集合名（例如“手部特写素材”“街景延时合集”）
"""
    if lang == "en":
        prompt += "\n额外要求：请保持 JSON 结构不变，但 preferences 的值尽量使用英文短语。"
    raw = analyze_with_qwen(prompt, max_tokens=500)
    try:
        data = json.loads(raw)
    except Exception:
        # 兜底：只用旧方法生成关键词
        return {
            "keywords": generate_search_keywords(user_request),
            "preferences": {"scene": [], "subject": [], "camera": [], "style": [], "avoid": []},
            "task_name": None
        }

    anchors = _normalize_anchor_tokens(data.get("anchors") or [])
    if not anchors:
        anchors = extract_core_english_anchors(user_request)

    kws = data.get("keywords") or []
    kws = [str(x).strip() for x in kws if str(x).strip()]
    kws = _to_core_or_core_plus_one_word(kws, anchors, max_n=10)
    # 保底补齐到10个
    if len(kws) < 10:
        extra = generate_search_keywords(user_request)
        for k in extra:
            if k not in kws:
                kws.append(k)
            if len(kws) >= 10:
                break
        kws = _to_core_or_core_plus_one_word(kws, anchors, max_n=10)
    kws = kws[:10]

    prefs = data.get("preferences") or {}
    if not isinstance(prefs, dict):
        prefs = {"scene": [], "subject": [], "camera": [], "style": [], "avoid": []}
    for key in ["scene", "subject", "camera", "style", "avoid"]:
        v = prefs.get(key) or []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        prefs[key] = [str(x).strip() for x in v if str(x).strip()]

    task_name = data.get("task_name")
    if task_name is not None:
        task_name = str(task_name).strip()
        if not task_name:
            task_name = None

    return {"keywords": kws, "preferences": prefs, "task_name": task_name}


def analyze_user_preferences(user_request, selected_videos, deselected_videos):
    lang = get_saved_language()
    selected_lines = chr(10).join([f"- {v['title']}" for v in (selected_videos or [])[:10]]) or "- （无）"
    deselected_lines = chr(10).join([f"- {v['title']}" for v in (deselected_videos or [])[:10]]) or "- （无）"
    has_deselected = bool(deselected_videos)
    cancel_rule = (
        "用户取消样本为空（可能是用户未取消或超时默认全选）。"
        "此时严禁臆造“被取消视频特征”，第2点必须写“无数据（未发生取消）”。"
        if not has_deselected
        else "若有取消样本，再总结取消特征。"
    )
    prompt = f"""用户想要搜索的视频画面类型：{user_request}

用户保留的视频标题：
{selected_lines}

用户取消保留的视频标题：
{deselected_lines}

约束：
{cancel_rule}

请分析用户偏好：
1. 用户保留的视频有什么共同特点？
2. 用户取消的视频有什么共同特点？（若无取消样本，必须写“无数据（未发生取消）”）
3. 根据用户的选择，总结用户最关注的是什么类型的画面？

输出格式简洁明了。"""
    if lang == "en":
        prompt += "\n额外要求：请用英文回答。"
    return analyze_with_qwen(prompt, max_tokens=400)


def check_video_match(user_request, preferences, video_title):
    prompt = f"""用户想要搜索：{user_request}
用户偏好（可能包含缓存偏好/分析结果）：{preferences}
视频标题：{video_title}

请判断这个视频是否符合用户的需求和偏好，只回答"是"或"否"。"""
    result = analyze_with_qwen(prompt, max_tokens=10)
    return "是" in result


def video_passes_duration_filter(task, video):
    """任务级时长过滤（秒）；未知时长不拦截。"""
    if not task:
        return True
    lo = task.get("video_duration_min_sec")
    hi = task.get("video_duration_max_sec")
    if lo is None and hi is None:
        return True
    dur = video.get("duration_seconds")
    if dur is None or dur <= 0:
        return True
    try:
        if lo is not None and dur < int(lo):
            return False
        if hi is not None and dur > int(hi):
            return False
    except (TypeError, ValueError):
        return True
    return True


def analyze_followup_instruction(user_input, task_snapshot):
    """已确认下载后的聊天指令：时长/总数/偏好/离题判断。返回 dict。"""
    lang = get_saved_language()
    payload = json.dumps(task_snapshot, ensure_ascii=False)
    prompt = f"""你是一个 YouTube 素材检索助手的指令解析器。
当前任务概况（JSON）：{payload}
用户在输入框里的新需求：{user_input}

请输出严格 JSON（不要代码块，不要 markdown），格式如下：
{{
  "intent": "duration|count|preference|off_topic|none",
  "duration_min_sec": null,
  "duration_max_sec": null,
  "target_delta": null,
  "target_absolute": null,
  "preference_patch": {{"scene": [], "subject": [], "camera": [], "style": [], "avoid": []}},
  "off_topic_gap": 0.0,
  "reply_zh": "给用户的简明中文回复"
}}

规则：
- intent=duration：用户提到时长、分钟、秒、不超过、至少、范围等；填 duration_min_sec / duration_max_sec（整数秒，可为 null）。
- intent=count：用户调整下载/推荐「总数」「改成N个」「再要5个」「减少3个」等；target_absolute 为最终目标个数或 null；target_delta 为相对当前目标的增减（整数，可为负）。
- intent=preference：用户补充画面、风格、主体、要避免的内容；只把新增短语写入 preference_patch 对应键（列表）。
- intent=off_topic：用户需求与当前任务主题明显不一致（如当前是动作类却要纯风景）；off_topic_gap 取 0~1，越大越偏离。
- intent=none：闲聊、感谢或无法归类。
- reply_zh：简短说明已执行或建议。
"""
    if lang == "en":
        prompt += "\n额外要求：JSON 结构不变；reply_zh 字段请使用英文回复。"
    raw = analyze_with_qwen(prompt, max_tokens=550)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "intent": "none",
        "reply_zh": (
            "I couldn't parse that command. Try: total 10 videos, or under 3 minutes."
            if lang == "en"
            else "我没能理解这条指令，请换一种说法（例如：总数十个、只要 3 分钟以内的）。"
        ),
    }


def download_thumbnail(video_id, max_size=(300, 169)):
    qualities = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg",
    ]
    for url in qualities:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content))
                img.load()
                img = img.convert("RGB")
                if img.size[0] < 1 or img.size[1] < 1:
                    continue
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                return img
        except Exception:
            continue
    return Image.new('RGB', max_size, (200, 200, 200))


def pil_to_ctk_image_safe(pil_img, box_w, box_h, fill=(200, 200, 200)):
    """
    PIL → CTkImage。Windows 上若 size 与像素矩阵不一致、或尺寸非整数/过大，Tk PhotoImage 会报「参数错误」。
    始终输出与 box 完全一致的 RGB 图再交给 CTkImage。
    """
    _trace("pil_to_ctk_image_safe enter", f"box=({box_w},{box_h}) pil_type={type(pil_img).__name__}")
    box_w = max(1, min(int(box_w), 4096))
    box_h = max(1, min(int(box_h), 4096))
    canvas = Image.new("RGB", (box_w, box_h), fill)
    try:
        if pil_img is None:
            raise ValueError("empty image")
        im = pil_img.copy() if hasattr(pil_img, "copy") else pil_img
        im = im.convert("RGB")
        im.load()
        if im.size[0] < 1 or im.size[1] < 1:
            raise ValueError("bad size")
        im.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
        ox = max(0, (box_w - im.size[0]) // 2)
        oy = max(0, (box_h - im.size[1]) // 2)
        canvas.paste(im, (ox, oy))
    except Exception as e:
        _trace("pil_to_ctk_image_safe canvas/paste failed", str(e))
        traceback.print_exc()
    try:
        out = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(box_w, box_h))
        _trace("pil_to_ctk_image_safe CTkImage ok", f"size=({box_w},{box_h})")
        return out
    except Exception as e:
        _trace("pil_to_ctk_image_safe CTkImage failed", str(e))
        traceback.print_exc()
        ph = Image.new("RGB", (box_w, box_h), fill)
        return ctk.CTkImage(light_image=ph, dark_image=ph, size=(box_w, box_h))


def get_videos_from_search(task_id, keyword, num_videos=5, seen_ids=None):
    """搜索视频并返回指定数量的视频"""
    if seen_ids is None:
        seen_ids = set()
    videos = []

    driver = TaskBrowserManager(task_id).get_search_driver()
    if not driver:
        print("无法获取浏览器驱动")
        return videos

    try:
        driver.set_page_load_timeout(30)
        driver.get(f"https://www.youtube.com/results?search_query={keyword}&sp=EgIQAQ%3D%3D")
        time.sleep(3)

        video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        cnt = 0

        for elem in video_elements:
            if cnt >= num_videos:
                break

            try:
                title_elem = elem.find_element(By.CSS_SELECTOR, "a#video-title")
                href = title_elem.get_attribute("href")
                title = title_elem.get_attribute("title") or "无标题"

                if href and ("/shorts/" in href):
                    continue

                if href and "watch" in href:
                    vid = re.search(r'v=([^&]+)', href)
                    if vid:
                        v_id = vid.group(1)
                        if v_id in seen_ids:
                            continue
                        seen_ids.add(v_id)

                        duration = "未知时长"
                        duration_seconds = 0
                        try:
                            duration_elem = elem.find_element(By.CSS_SELECTOR,
                                                              "span.ytd-thumbnail-overlay-time-status-renderer")
                            aria_label = duration_elem.get_attribute("aria-label")
                            if aria_label and aria_label != "Shorts":
                                duration = parse_chinese_duration(aria_label)
                                duration_parts = duration.split(':')
                                if len(duration_parts) == 2:
                                    duration_seconds = int(duration_parts[0]) * 60 + int(duration_parts[1])
                                elif len(duration_parts) == 3:
                                    duration_seconds = int(duration_parts[0]) * 3600 + int(
                                        duration_parts[1]) * 60 + int(duration_parts[2])
                        except:
                            try:
                                alt_elem = elem.find_element(By.CSS_SELECTOR,
                                                             "ytd-thumbnail-overlay-time-status-renderer span")
                                aria_label = alt_elem.get_attribute("aria-label")
                                if aria_label and aria_label != "Shorts":
                                    duration = parse_chinese_duration(aria_label)
                                    duration_parts = duration.split(':')
                                    if len(duration_parts) == 2:
                                        duration_seconds = int(duration_parts[0]) * 60 + int(duration_parts[1])
                                    elif len(duration_parts) == 3:
                                        duration_seconds = int(duration_parts[0]) * 3600 + int(
                                            duration_parts[1]) * 60 + int(duration_parts[2])
                            except:
                                pass

                        if duration == "未知时长":
                            html = elem.get_attribute('outerHTML')
                            time_match = re.search(r'(\d+):(\d+)(?::(\d+))?', html)
                            if time_match:
                                minutes = int(time_match.group(1))
                                seconds = int(time_match.group(2))
                                if time_match.group(3):
                                    hours = minutes
                                    minutes = seconds
                                    seconds = int(time_match.group(3))
                                    duration = f"{hours}:{minutes:02d}:{seconds:02d}"
                                    duration_seconds = hours * 3600 + minutes * 60 + seconds
                                else:
                                    duration = f"{minutes}:{seconds:02d}"
                                    duration_seconds = minutes * 60 + seconds

                        videos.append({
                            "title": title,
                            "url": href,
                            "video_id": v_id,
                            "keyword": keyword,
                            "selected": True,
                            "duration": duration,
                            "duration_seconds": duration_seconds
                        })
                        cnt += 1
            except Exception as e:
                print(f"处理视频出错: {e}")
                continue

    except Exception as e:
        print(f"搜索出错: {e}")
        if "no such window" in str(e) or "invalid session id" in str(e):
            TaskBrowserManager(task_id).close_search_driver()

    return videos


def get_all_videos_from_search(task_id, keyword, seen_ids=None):
    """搜索关键词下的所有视频（滚动加载）"""
    if seen_ids is None:
        seen_ids = set()
    videos = []
    all_video_ids = set()

    driver = TaskBrowserManager(task_id).get_search_driver()
    if not driver:
        print("无法获取浏览器驱动")
        return videos

    try:
        driver.set_page_load_timeout(30)
        driver.get(f"https://www.youtube.com/results?search_query={keyword}&sp=EgIQAQ%3D%3D")
        time.sleep(3)

        # 滚动页面以加载更多视频
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        scroll_attempts = 0
        max_scrolls = 5  # 最多滚动5次

        while scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

            # 检查是否有"加载更多"按钮
            try:
                load_more = driver.find_element(By.CSS_SELECTOR, "ytd-continuation-item-renderer button")
                if load_more and load_more.is_enabled():
                    load_more.click()
                    time.sleep(2)
            except:
                pass

            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1

        # 获取所有视频元素
        video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        print(f"关键词 {keyword} 共找到 {len(video_elements)} 个视频")

        for elem in video_elements:
            try:
                title_elem = elem.find_element(By.CSS_SELECTOR, "a#video-title")
                href = title_elem.get_attribute("href")
                title = title_elem.get_attribute("title") or "无标题"

                if href and ("/shorts/" in href):
                    continue

                if href and "watch" in href:
                    vid = re.search(r'v=([^&]+)', href)
                    if vid:
                        v_id = vid.group(1)

                        if v_id in all_video_ids or v_id in seen_ids:
                            continue

                        all_video_ids.add(v_id)

                        duration = "未知时长"
                        duration_seconds = 0
                        try:
                            duration_elem = elem.find_element(By.CSS_SELECTOR,
                                                              "span.ytd-thumbnail-overlay-time-status-renderer")
                            aria_label = duration_elem.get_attribute("aria-label")
                            if aria_label and aria_label != "Shorts":
                                duration = parse_chinese_duration(aria_label)
                                duration_parts = duration.split(':')
                                if len(duration_parts) == 2:
                                    duration_seconds = int(duration_parts[0]) * 60 + int(duration_parts[1])
                                elif len(duration_parts) == 3:
                                    duration_seconds = int(duration_parts[0]) * 3600 + int(
                                        duration_parts[1]) * 60 + int(duration_parts[2])
                        except:
                            try:
                                alt_elem = elem.find_element(By.CSS_SELECTOR,
                                                             "ytd-thumbnail-overlay-time-status-renderer span")
                                aria_label = alt_elem.get_attribute("aria-label")
                                if aria_label and aria_label != "Shorts":
                                    duration = parse_chinese_duration(aria_label)
                                    duration_parts = duration.split(':')
                                    if len(duration_parts) == 2:
                                        duration_seconds = int(duration_parts[0]) * 60 + int(duration_parts[1])
                                    elif len(duration_parts) == 3:
                                        duration_seconds = int(duration_parts[0]) * 3600 + int(
                                            duration_parts[1]) * 60 + int(duration_parts[2])
                            except:
                                pass

                        if duration == "未知时长":
                            html = elem.get_attribute('outerHTML')
                            time_match = re.search(r'(\d+):(\d+)(?::(\d+))?', html)
                            if time_match:
                                minutes = int(time_match.group(1))
                                seconds = int(time_match.group(2))
                                if time_match.group(3):
                                    hours = minutes
                                    minutes = seconds
                                    seconds = int(time_match.group(3))
                                    duration = f"{hours}:{minutes:02d}:{seconds:02d}"
                                    duration_seconds = hours * 3600 + minutes * 60 + seconds
                                else:
                                    duration = f"{minutes}:{seconds:02d}"
                                    duration_seconds = minutes * 60 + seconds

                        videos.append({
                            "title": title,
                            "url": href,
                            "video_id": v_id,
                            "keyword": keyword,
                            "selected": True,
                            "duration": duration,
                            "duration_seconds": duration_seconds
                        })

            except Exception as e:
                print(f"处理视频出错: {e}")
                continue

    except Exception as e:
        print(f"搜索出错: {e}")
        if "no such window" in str(e) or "invalid session id" in str(e):
            TaskBrowserManager(task_id).close_search_driver()

    return videos


def get_recommended_videos_from_watch(task_id, video_url, current_video_id=None, seen_ids=None, max_videos=20):
    """在视频播放页获取右侧推荐视频列表"""
    if seen_ids is None:
        seen_ids = set()
    videos = []

    driver = TaskBrowserManager(task_id).get_search_driver()
    if not driver:
        print("无法获取浏览器驱动")
        return videos

    try:
        driver.set_page_load_timeout(30)
        driver.get(video_url)
        print(f"🎬 正在打开视频页面: {video_url}")

        # 等待视频页面加载
        time.sleep(5)

        # 滚动以触发推荐视频加载
        try:
            driver.execute_script("window.scrollTo(0, 400);")
        except Exception:
            pass
        time.sleep(3)

        # 等待推荐区域出现
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#related"))
            )
            print("✅ 推荐区域已加载")
        except:
            print("⚠️ 推荐区域加载超时")

        video_elements = []

        # 方法1：通过 ytd-watch-next-secondary-results-renderer
        try:
            secondary_results = driver.find_element(By.CSS_SELECTOR, "ytd-watch-next-secondary-results-renderer")
            video_elements = secondary_results.find_elements(By.CSS_SELECTOR, "ytd-compact-video-renderer")
            if not video_elements:
                # 尝试其他选择器
                video_elements = secondary_results.find_elements(By.CSS_SELECTOR, "a[href*='/watch?v=']")
        except Exception as e:
            print(f"方法1失败: {e}")
            # 方法2：直接通过 #related 查找
            try:
                related_div = driver.find_element(By.ID, "related")
                video_elements = related_div.find_elements(By.CSS_SELECTOR, "ytd-compact-video-renderer")
                if not video_elements:
                    video_elements = related_div.find_elements(By.CSS_SELECTOR, "a[href*='/watch?v=']")
            except Exception as e2:
                print(f"方法2失败: {e2}")
                return videos

        print(f"找到 {len(video_elements)} 个推荐视频元素")

        for element in video_elements:
            if len(videos) >= max_videos:
                break

            try:
                # 获取容器元素
                container = element
                if element.tag_name == "a":
                    try:
                        container = element.find_element(By.XPATH, "./ancestor::ytd-compact-video-renderer[1]")
                    except Exception:
                        container = element

                # 获取视频链接
                video_link = ""
                try:
                    # 尝试从 a#video-title 获取链接
                    title_link = container.find_element(By.CSS_SELECTOR, "a#video-title")
                    video_link = title_link.get_attribute("href") or ""
                except:
                    # 尝试从 a#thumbnail 获取链接
                    try:
                        thumb_link = container.find_element(By.CSS_SELECTOR, "a#thumbnail")
                        video_link = thumb_link.get_attribute("href") or ""
                    except:
                        if element.tag_name == "a":
                            video_link = element.get_attribute("href") or ""

                if not video_link or "watch" not in video_link:
                    continue

                # 获取视频标题 - 这是关键修复部分
                title = ""

                # 方法1: 从 a#video-title 获取 title 属性
                try:
                    title_element = container.find_element(By.CSS_SELECTOR, "a#video-title")
                    title = title_element.get_attribute("title") or ""
                    if not title:
                        title = title_element.text or ""
                except:
                    pass

                # 方法2: 从 span#video-title 获取文本
                if not title:
                    try:
                        title_span = container.find_element(By.CSS_SELECTOR, "span#video-title")
                        title = title_span.text or ""
                    except:
                        pass

                # 方法3: 从 yt-formatted-string 获取
                if not title:
                    try:
                        title_yt = container.find_element(By.CSS_SELECTOR, "yt-formatted-string#video-title")
                        title = title_yt.text or ""
                    except:
                        pass

                # 方法4: 查找任何包含标题的 span 元素
                if not title:
                    try:
                        title_spans = container.find_elements(By.CSS_SELECTOR, "span")
                        for span in title_spans:
                            text = span.text.strip()
                            if text and len(text) > 3:  # 标题通常不会太短
                                title = text
                                break
                    except:
                        pass

                # 清理标题
                title = title.strip()
                if not title:
                    print("⚠️ 无法获取标题，跳过此视频")
                    continue

                # 提取视频ID
                vid_match = re.search(r'v=([^&]+)', video_link)
                if not vid_match:
                    continue
                video_id = vid_match.group(1)

                # 跳过当前视频
                if current_video_id and video_id == current_video_id:
                    continue
                # 跳过已见过的视频
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)

                # 获取频道名称
                channel = "未知频道"
                try:
                    channel_element = container.find_element(By.CSS_SELECTOR, "ytd-channel-name, #channel-name")
                    channel = channel_element.text.strip() or "未知频道"
                except:
                    pass

                # 获取时长
                duration = "未知时长"
                duration_seconds = 0
                try:
                    duration_elem = container.find_element(By.CSS_SELECTOR,
                                                           "span.ytd-thumbnail-overlay-time-status-renderer")
                    aria_label = duration_elem.get_attribute("aria-label")
                    if aria_label and aria_label != "Shorts":
                        # 解析时长
                        duration = parse_chinese_duration(aria_label)
                        parts = duration.split(":")
                        if len(parts) == 2:
                            duration_seconds = int(parts[0]) * 60 + int(parts[1])
                        elif len(parts) == 3:
                            duration_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                except Exception:
                    pass

                # 如果时长还是未知，尝试从 HTML 中提取
                if duration == "未知时长":
                    try:
                        html = container.get_attribute("outerHTML")
                        time_match = re.search(r'(\d+):(\d+)(?::(\d+))?', html)
                        if time_match:
                            minutes = int(time_match.group(1))
                            seconds = int(time_match.group(2))
                            if time_match.group(3):
                                hours = minutes
                                minutes = seconds
                                seconds = int(time_match.group(3))
                                duration = f"{hours}:{minutes:02d}:{seconds:02d}"
                                duration_seconds = hours * 3600 + minutes * 60 + seconds
                            else:
                                duration = f"{minutes}:{seconds:02d}"
                                duration_seconds = minutes * 60 + seconds
                    except Exception:
                        pass

                videos.append({
                    "title": title,
                    "url": video_link,
                    "video_id": video_id,
                    "channel": channel,
                    "keyword": "related",
                    "selected": True,
                    "duration": duration,
                    "duration_seconds": duration_seconds
                })

                print(f"✅ 获取到推荐视频: {title[:50]}...")

            except Exception as e:
                print(f"处理推荐视频出错: {e}")
                continue

    except Exception as e:
        print(f"打开视频页面或获取推荐失败: {e}")
        import traceback
        traceback.print_exc()
        if "no such window" in str(e) or "invalid session id" in str(e):
            TaskBrowserManager(task_id).close_search_driver()

    print(f"共获取到 {len(videos)} 个推荐视频")
    return videos

# ==================== UI 组件 ====================
class StatusMessageFrame(ctk.CTkFrame):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        b = ctk.CTkFrame(self, fg_color="transparent")
        b.grid(row=0, column=0, sticky="w", padx=(20, 10), pady=6)
        self.label = ctk.CTkLabel(b, text=text, font=ctk.CTkFont(family="Microsoft YaHei", size=13),
                                  text_color="#999999", justify="left")
        self.label.pack(padx=8, pady=6)

    def update_text(self, text, text_color=None):
        self.label.configure(text=text)
        if text_color:
            self.label.configure(text_color=text_color)


class VideoCardFrame(ctk.CTkFrame):
    # 聊天区多张卡片错开贴图间隔（毫秒），避免主线程同一时刻创建过多位图
    _CHAT_THUMB_STAGGER_MS = 160

    def __init__(self, parent, video, task_manager, task_id, video_index=None, total_videos=None,
                 on_selection_change=None, defer_image_load=False, **kwargs):
        vid = (video or {}).get("video_id", "?")
        super().__init__(parent, corner_radius=8, fg_color="#ffffff", border_width=1, border_color="#e6e6e6", **kwargs)
        self.video = video
        self.task_manager = task_manager
        self.task_id = task_id
        self.on_selection_change = on_selection_change
        self.video_index = video_index
        self.total_videos = total_videos
        self._thumb_label = None
        self._thumb_placeholder = None
        self._thumb_container = None
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        thumb_container = ctk.CTkFrame(self, fg_color="transparent")
        thumb_container.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        self._thumb_container = thumb_container

        if defer_image_load:
            # 不要用 CTkImage 做灰色占位：每张卡一张位图，多卡 + 右侧列表易触发 Windows GDI 上限（Tk_GetPixmap / CreateDIBSection）
            self._thumb_placeholder = ctk.CTkFrame(
                thumb_container, width=160, height=90, corner_radius=6, fg_color="#e6e6e6")
            self._thumb_placeholder.grid(row=0, column=0)
            # 冷启动也允许加载聊天区封面，不再强依赖点击「继续进程」。
            self._thumb_load_started = True
            threading.Thread(target=self._load_video_card_thumb_background, daemon=True).start()
        else:
            img = self.load_or_download_thumbnail()
            self.ctk_img = pil_to_ctk_image_safe(img, 160, 90)
            self._thumb_label = ctk.CTkLabel(thumb_container, image=self.ctk_img, text="", corner_radius=6)
            self._thumb_label.grid(row=0, column=0)

        duration_text = video.get('duration', '未知时长')
        duration_label = ctk.CTkLabel(thumb_container, text=duration_text,
                                      font=ctk.CTkFont(family="Microsoft YaHei", size=11, weight="bold"),
                                      text_color="#ffffff", fg_color="#000000", corner_radius=3, padx=6, pady=2)
        duration_label.place(relx=0.95, rely=0.92, anchor="se")

        txt = ctk.CTkFrame(self, fg_color="transparent")
        txt.grid(row=0, column=1, padx=10, pady=12, sticky="nsew")
        txt.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(txt, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(title_frame, text=video['title'],
                     font=ctk.CTkFont(family="Microsoft YaHei", size=14, weight="bold"),
                     wraplength=350, anchor="w").grid(row=0, column=0, sticky="ew")

        if video_index is not None and total_videos is not None:
            index_text = f"{video_index}/{total_videos}"
            ctk.CTkLabel(title_frame, text=index_text,
                         font=ctk.CTkFont(family="Microsoft YaHei", size=12, weight="bold"),
                         text_color="#5c9eff", anchor="e").grid(row=0, column=1, padx=(5, 0), sticky="e")

        info_frame = ctk.CTkFrame(txt, fg_color="transparent")
        info_frame.grid(row=1, column=0, sticky="ew", pady=4)
        info_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(info_frame, text=f"关键词：{video['keyword']}", font=ctk.CTkFont(family="Microsoft YaHei", size=12),
                     text_color="#666", anchor="w").grid(row=0, column=0, sticky="w")
        try:
            lang = get_saved_language()
            if lang == "en":
                for w in info_frame.winfo_children():
                    if isinstance(w, ctk.CTkLabel) and str(w.cget("text")).startswith("关键词："):
                        w.configure(text=f"Keyword: {video['keyword']}")
                        break
        except Exception:
            pass
        duration_display = f"⏱️ {duration_text}"
        ctk.CTkLabel(info_frame, text=duration_display, font=ctk.CTkFont(family="Microsoft YaHei", size=12),
                     text_color="#999", anchor="e").grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.check_var = ctk.BooleanVar(value=video.get('selected', True))
        self.checkbox = ctk.CTkCheckBox(self, text="", variable=self.check_var, width=24, height=24,
                                        corner_radius=4, border_width=2, checkbox_width=20, checkbox_height=20,
                                        command=self.on_checkbox_change)
        self.checkbox.grid(row=0, column=2, padx=15, pady=10, sticky="ne")

    def schedule_thumbnail_after_resume(self):
        """用户点击「继续进程」后：再启动后台读盘贴聊天卡片缩略图（避免启动阶段主线程/GDI 爆）。"""
        if getattr(self, "_thumb_load_started", False):
            return
        if not self._thumb_placeholder:
            return
        self._thumb_load_started = True
        threading.Thread(target=self._load_video_card_thumb_background, daemon=True).start()

    def _schedule_chat_thumb_apply_main(self, pil_img):
        """主线程错开创建 CTkImage：若多个 after(0) 同时贴图，仍会短时间耗尽 GDI（Tk_GetPixmap / CreateDIBSection）。"""
        try:
            if not self.winfo_exists():
                return
            ix = (self.video_index or 1) - 1
            gap = int(getattr(type(self), "_CHAT_THUMB_STAGGER_MS", 160))
            delay_ms = max(0, ix * gap)
            self.after(delay_ms, lambda p=pil_img: self._apply_video_card_thumb_from_pil(p))
        except Exception:
            pass

    def _load_video_card_thumb_background(self):
        """线程里读盘/拉网；用顶层 after(0) 切回主线程再错峰贴图，勿在子线程直接 after(多级)。"""
        try:
            pil = self._resolve_thumbnail_pil_for_card()
        except Exception:
            pil = None
        if pil is None:
            return

        def _to_main():
            try:
                self._schedule_chat_thumb_apply_main(pil)
            except Exception:
                pass

        try:
            root = self.winfo_toplevel()
            root.after(0, _to_main)
        except Exception:
            try:
                self.after(0, _to_main)
            except Exception:
                pass

    def _resolve_thumbnail_pil_for_card(self):
        # 聊天卡片：优先本地缓存；无缓存时允许补拉网络封面（即使任务暂停也加载封面）
        img = self.task_manager.load_thumbnail(self.task_id, self.video['video_id'])
        if img is not None:
            return img
        img = download_thumbnail(self.video['video_id'])
        self.task_manager.save_thumbnail(self.task_id, self.video['video_id'], img)
        return img

    def _apply_video_card_thumb_from_pil(self, pil_img):
        try:
            if not self.winfo_exists() or not self._thumb_container:
                return
            self.ctk_img = pil_to_ctk_image_safe(pil_img, 160, 90)
            ph = getattr(self, "_thumb_placeholder", None)
            if ph is not None:
                try:
                    ph.destroy()
                except Exception:
                    pass
                self._thumb_placeholder = None
            if self._thumb_label is None:
                self._thumb_label = ctk.CTkLabel(
                    self._thumb_container, image=self.ctk_img, text="", corner_radius=6)
                self._thumb_label.grid(row=0, column=0)
            else:
                self._thumb_label.configure(image=self.ctk_img)
        except Exception:
            pass

    def load_or_download_thumbnail(self):
        # 聊天卡片：优先本地缓存；无缓存时补拉网络封面（即使任务暂停也加载封面）
        img = self.task_manager.load_thumbnail(self.task_id, self.video['video_id'])
        if img is not None:
            return img
        img = download_thumbnail(self.video['video_id'])
        self.task_manager.save_thumbnail(self.task_id, self.video['video_id'], img)
        return img

    def on_checkbox_change(self):
        selected = self.check_var.get()
        self.video['selected'] = selected
        if self.on_selection_change:
            self.on_selection_change(self.video, selected)

    def get_selected(self):
        return self.check_var.get()

    def set_selected(self, selected):
        self.check_var.set(selected)
        self.on_checkbox_change()


class DownloadItemFrame(ctk.CTkFrame):
    def __init__(self, parent, video, task_manager, task_id, on_status_change=None, on_delete=None, list_frame=None,
                 **kwargs):
        _vid_early = (video or {}).get("video_id")
        self._init_step = "enter"
        _trace("DownloadItemFrame.__init__", f"task={task_id} vid={_vid_early}")
        try:
            self._init_step = "super"
            super().__init__(parent, corner_radius=8, fg_color="#ffffff", border_width=1, border_color="#e6e6e6",
                             height=80, **kwargs)
            self._init_step = "attrs"
            self.video = video
            self.task_manager = task_manager
            self.task_id = task_id
            self.list_frame = list_frame
            self.on_status_change = on_status_change
            self.on_delete = on_delete
            self.video_id = video['video_id']
            self.extraction_thread = None
            self.extractor = None
            self.is_extracting = False
            # Configure 可能在 restore_from_cache 之前触发 on_resize；进度条宽 0 在 Windows 上易触发「参数错误」
            self.progress = 0

            self.grid_columnconfigure(0, weight=0)
            self.grid_columnconfigure(1, weight=1)

            self.bind("<Button-3>", self.show_context_menu)

            # 封面：主线程不读盘；占位 + 后台队列（见 add_video_to_queue / _ensure_thumb_worker）
            self._init_step = "thumb_slot"
            self.thumb_slot = ctk.CTkFrame(self, fg_color="transparent")
            self.thumb_slot.grid(row=0, column=0, padx=8, pady=8, sticky="nw")
            self.thumb_slot.bind("<Button-3>", self.show_context_menu)
            # 不在主线程同步读盘/解码 JPEG：占位，封面由 DownloadListFrame 后台队列逐个补全
            self._init_step = "thumb_placeholder"
            self.refresh_thumbnail_widget(placeholder_only=True)

            self._init_step = "info_frame"
            info_frame = ctk.CTkFrame(self, fg_color="transparent")
            info_frame.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
            info_frame.grid_columnconfigure(0, weight=1)
            info_frame.bind("<Button-3>", self.show_context_menu)

            _t = video.get('title') or ''
            title_text = (_t[:40] + '...') if len(_t) > 40 else str(_t)
            self._init_step = "title_label"
            self.title_label = ctk.CTkLabel(info_frame, text=title_text,
                                            font=ctk.CTkFont(family="Microsoft YaHei", size=12, weight="bold"),
                                            anchor="w")
            self.title_label.grid(row=0, column=0, sticky="ew")
            self.title_label.bind("<Button-3>", self.show_context_menu)

            self._init_step = "duration_label"
            ctk.CTkLabel(info_frame, text=f"时长: {video.get('duration', '未知')}",
                         font=ctk.CTkFont(family="Microsoft YaHei", size=10), text_color="#666", anchor="w"
                         ).grid(row=1, column=0, sticky="w", pady=(2, 0))

            self._init_step = "progress_container"
            self.progress_container = ctk.CTkFrame(info_frame, height=4, fg_color="#e0e0e0", corner_radius=2)
            self.progress_container.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            self.progress_container.grid_columnconfigure(0, weight=1)

            self._init_step = "progress_mask"
            self.progress_mask = ctk.CTkFrame(self.progress_container, height=4, fg_color="#5c9eff", corner_radius=2)
            self.progress_mask.grid(row=0, column=0, sticky="w")

            self._init_step = "status_text"
            self.status_text = ctk.CTkLabel(info_frame, text="", font=ctk.CTkFont(family="Microsoft YaHei", size=9),
                                            text_color="#999", anchor="w")
            self.status_text.grid(row=3, column=0, sticky="ew", pady=(2, 0))
            self.status_text.bind("<Button-3>", self.show_context_menu)

            self._init_step = "bind_configure"
            self.bind("<Configure>", self.on_resize)
            self._init_step = "restore_from_cache"
            _tk_dbg("DownloadItemFrame before restore_from_cache", task_id, self.video_id)
            self.restore_from_cache()
            self._init_step = "done"
        except Exception as e:
            try:
                print(
                    f"[DL_ITEM] FAIL step={getattr(self, '_init_step', '?')} task={task_id!r} vid={_vid_early!r} "
                    f"err={e!r}",
                    flush=True,
                )
                traceback.print_exc()
            except Exception:
                pass
            raise

    def restore_from_cache(self):
        cached_status = self.task_manager.get_video_extraction_status(self.task_id, self.video_id)
        status_text = cached_status.get('status_text', '等待中')
        progress = cached_status.get('progress', 0)
        self.update_status(status_text, progress, save_to_cache=False)

    def load_thumbnail(self):
        return self.task_manager.load_thumbnail(self.task_id, self.video['video_id'])

    def refresh_thumbnail_widget(self, pil_image=None, placeholder_only=False):
        """左侧封面：pil_image 由后台线程读完再传入主线程；placeholder_only 仅灰色占位（启动批量恢复时不卡 UI）。"""
        try:
            for w in self.thumb_slot.winfo_children():
                w.destroy()
        except tk.TclError as e:
            _log_tcl_error("refresh_thumbnail_widget.destroy_children", e, self.video_id, self.task_id)
            raise
        if placeholder_only:
            try:
                placeholder = ctk.CTkLabel(
                    self.thumb_slot, text="···", width=60, height=34, fg_color="#e8e8e8",
                    text_color="#bbbbbb", font=ctk.CTkFont(size=11), corner_radius=4)
                placeholder.pack(fill="both", expand=True)
                placeholder.bind("<Button-3>", self.show_context_menu)
            except tk.TclError as e:
                _log_tcl_error("refresh_thumbnail_widget.placeholder", e, self.video_id, self.task_id)
                raise
            return
        img = pil_image if pil_image is not None else self.load_thumbnail()
        if img:
            self.ctk_img = pil_to_ctk_image_safe(img, 60, 34)
            try:
                img_label = ctk.CTkLabel(self.thumb_slot, image=self.ctk_img, text="")
                img_label.pack(fill="both", expand=True)
                img_label.bind("<Button-3>", self.show_context_menu)
            except tk.TclError as e:
                _log_tcl_error("refresh_thumbnail_widget.ctk_image_label", e, self.video_id, self.task_id)
                raise
        else:
            try:
                placeholder = ctk.CTkLabel(self.thumb_slot, text="无封面", width=60, height=34, fg_color="#e0e0e0",
                                           corner_radius=4)
                placeholder.pack(fill="both", expand=True)
                placeholder.bind("<Button-3>", self.show_context_menu)
            except tk.TclError as e:
                _log_tcl_error("refresh_thumbnail_widget.no_thumb_placeholder", e, self.video_id, self.task_id)
                raise

    def on_resize(self, event):
        """Configure 回调里不要向上抛 TclError，否则可能打断 Tk 事件循环；只打日志。"""
        try:
            self.update_progress(self.progress)
        except tk.TclError as e:
            _log_tcl_error("on_resize.update_progress", e, self.video_id, self.task_id)
        except Exception:
            pass

    def update_progress(self, percent):
        try:
            p = float(percent)
        except (TypeError, ValueError):
            p = 0.0
        p = max(0.0, min(100.0, p))
        self.progress = p
        try:
            self.progress_container.update_idletasks()
            cw = int(self.progress_container.winfo_width())
        except Exception:
            cw = 0
        if cw <= 1:
            return
        mask_w = int(round(cw * p / 100.0))
        # Windows 上 CTkFrame/Tk 对 width=0 常报「参数错误」，0% 时隐藏进度条前景
        try:
            if mask_w < 1:
                self.progress_mask.grid_remove()
            else:
                self.progress_mask.grid(row=0, column=0, sticky="w")
                self.progress_mask.configure(width=mask_w)
        except tk.TclError as e:
            _log_tcl_error(
                f"update_progress.configure mask_w={mask_w} cw={cw} p={p}",
                e,
                getattr(self, "video_id", None),
                getattr(self, "task_id", None),
            )
            raise
        except Exception as e:
            _tk_dbg(f"update_progress non-tcl err={e!r}", getattr(self, "task_id", None), getattr(self, "video_id", None))
            traceback.print_exc()

    def update_status(self, status_text, progress=0, save_to_cache=True):
        try:
            self.status_text.configure(text=status_text)
        except tk.TclError as e:
            _log_tcl_error("update_status.status_text.configure", e, self.video_id, self.task_id)
            raise

        if "(" in status_text and "帧" in status_text:
            match = re.search(r'\((\d+)帧\)', status_text)
            if match:
                frame_count = int(match.group(1))
                progress = min(100, int(frame_count * 100 / 1000))

        if "已完成" in status_text:
            self.update_progress(100)
        elif "失败" in status_text:
            self.update_progress(0)
        elif "等待中" in status_text or "排队中" in status_text:
            self.update_progress(0)
        else:
            self.update_progress(progress)

        if save_to_cache:
            frame_count = 0
            if "(" in status_text and "帧" in status_text:
                match = re.search(r'\((\d+)帧\)', status_text)
                if match:
                    frame_count = int(match.group(1))
            elif "已完成" in status_text:
                frames_info = self.task_manager.get_video_frames_info(self.task_id, self.video['title'], self.video_id)
                frame_count = frames_info.get('total_frames', 0)

            frames_info = self.task_manager.get_video_frames_info(self.task_id, self.video['title'], self.video_id)
            last_frame_time = frames_info.get('last_extracted_time', -1)

            self.task_manager.update_video_extraction_status(
                self.task_id, self.video_id, status_text, progress, frame_count, last_frame_time
            )

        if self.on_status_change:
            self.on_status_change(self.video_id, status_text, progress)

        if self.list_frame:
            try:
                self.list_frame.update_count_display()
            except Exception:
                pass

    def show_context_menu(self, event):
        menu = Menu(self, tearoff=False)
        label = self.list_frame._app.tr("menu_delete") if self.list_frame and self.list_frame._app else "🗑️ 删除"
        menu.add_command(label=label, font=("Microsoft YaHei", 12), foreground="#ff4444", command=self.delete_item)
        menu.post(event.x_root, event.y_root)

    def delete_item(self):
        title = self.list_frame._app.tr("delete_confirm_title") if self.list_frame and self.list_frame._app else "确认删除"
        text = (
            self.list_frame._app.tr("delete_video_confirm", name=self.video['title'][:50])
            if self.list_frame and self.list_frame._app
            else f"确定要删除「{self.video['title'][:50]}」吗？\n\n已提取的帧也会被删除。"
        )
        result = messagebox.askyesno(title, text,
                                     icon='warning')
        if result:
            if self.extractor:
                self.extractor.stop()
            self.task_manager.merge_seen_video_ids_batch(self.task_id, [self.video_id])
            self.task_manager.delete_thumbnail(self.task_id, self.video['video_id'])
            self.task_manager.delete_video_frames(self.task_id, self.video['title'])
            self.task_manager.update_video_extraction_status(self.task_id, self.video_id, "已删除", 0, 0)
            if self.on_delete:
                self.on_delete(self.video_id)

    def pause_extraction(self):
        if "已完成" in self.status_text.cget("text"):
            return False
        if self.extractor:
            self.extractor.pause()
            self.update_status(f"⏸️ 已暂停", self.progress)
            return True
        return False

    def resume_extraction(self):
        if "已完成" in self.status_text.cget("text"):
            return False
        if self.extractor:
            self.extractor.resume()
            self.update_status(f"▶️ 继续提取中... ({self.progress}%)", self.progress)
            return True
        return False

    def start_extraction(self):
        if "已完成" in self.status_text.cget("text"):
            return
        if self.is_extracting:
            return
        cached_status = self.task_manager.get_video_extraction_status(self.task_id, self.video_id)
        if cached_status.get('status') == "已完成":
            return

        self.is_extracting = True

        def progress_callback(message, progress):
            def update_progress():
                try:
                    if self.winfo_exists():
                        self.update_status(message, progress)
                except:
                    pass

            self.after(0, update_progress)

        def extraction_thread():
            self.extractor = FrameExtractor(
                self.video['url'], self.video['title'], self.video['video_id'],
                self.task_id, self.task_manager, frame_interval=1
            )
            if self.task_manager.is_task_paused(self.task_id):
                self.extractor.pause()
                self.update_status(f"⏸️ 已暂停 (等待恢复)", 0)

            frames_path, frame_count = self.extractor.extract(progress_callback)

            def finish():
                self.is_extracting = False
                try:
                    if self.winfo_exists():
                        if frames_path:
                            self.update_status(f"✅ 已完成 ({frame_count}帧)", 100)
                            self.task_manager.reset_retry_count(self.task_id, self.video_id)
                        else:
                            self.update_status("❌ 失败", 0)
                            self.task_manager.mark_video_failed(self.task_id, self.video_id)
                except:
                    pass

            self.after(0, finish)

        thread = threading.Thread(target=extraction_thread, daemon=True)
        thread.start()
        self.extraction_thread = thread


class MinimalExtractionItem:
    """
    帧提取逻辑体（无列表行 UI），供 8×5 宫格按需挂载 GridCell。
    接口与 DownloadItemFrame 中 extraction/status 部分对齐，供队列与重试调用。
    """

    def __init__(self, list_frame, video, task_manager, task_id, on_status_change=None, on_delete=None):
        self.list_frame = list_frame
        self.video = video
        self.video_id = video["video_id"]
        self.task_manager = task_manager
        self.task_id = task_id
        self.on_status_change = on_status_change
        self.on_delete = on_delete
        self.progress = 0
        self._status_text = "等待中"
        self.extractor = None
        self.is_extracting = False
        self.extraction_thread = None

    def get_status_text(self):
        return self._status_text

    def restore_from_cache(self):
        cached = self.task_manager.get_video_extraction_status(self.task_id, self.video_id)
        self._status_text = cached.get("status_text", "等待中")
        self.progress = cached.get("progress", 0)
        self.update_status(self._status_text, self.progress, save_to_cache=False)

    def load_thumbnail(self):
        return self.task_manager.load_thumbnail(self.task_id, self.video_id)

    def update_progress(self, percent):
        try:
            p = float(percent)
        except (TypeError, ValueError):
            p = 0.0
        self.progress = max(0.0, min(100.0, p))
        self.list_frame.after(0, lambda: self.list_frame.refresh_item_ui(self.video_id))

    def update_status(self, status_text, progress=0, save_to_cache=True):
        self._status_text = status_text
        if "(" in status_text and "帧" in status_text:
            match = re.search(r"\((\d+)帧\)", status_text)
            if match:
                frame_count = int(match.group(1))
                progress = min(100, int(frame_count * 100 / 1000))
        if "已完成" in status_text:
            self.update_progress(100)
        elif "失败" in status_text:
            self.update_progress(0)
        elif "等待中" in status_text or "排队中" in status_text:
            self.update_progress(0)
        else:
            self.update_progress(progress)

        if save_to_cache:
            frame_count = 0
            if "(" in status_text and "帧" in status_text:
                match = re.search(r"\((\d+)帧\)", status_text)
                if match:
                    frame_count = int(match.group(1))
            elif "已完成" in status_text:
                fi = self.task_manager.get_video_frames_info(
                    self.task_id, self.video["title"], self.video_id
                )
                frame_count = fi.get("total_frames", 0)
            fi = self.task_manager.get_video_frames_info(
                self.task_id, self.video["title"], self.video_id
            )
            last_t = fi.get("last_extracted_time", -1)
            self.task_manager.update_video_extraction_status(
                self.task_id, self.video_id, status_text, progress, frame_count, last_t
            )

        if self.on_status_change:
            self.on_status_change(self.video_id, status_text, progress)
        if self.list_frame:
            try:
                self.list_frame.update_count_display()
            except Exception:
                pass
        self.list_frame.after(0, lambda: self.list_frame.refresh_item_ui(self.video_id))

    def show_context_menu(self, event):
        menu = Menu(self.list_frame, tearoff=False)
        tr = self.list_frame._app.tr if (self.list_frame and self.list_frame._app and hasattr(self.list_frame._app, "tr")) else None
        menu.add_command(
            label=("🔄 Redownload" if tr and self.list_frame._app.lang == "en" else "🔄 重新下载"),
            font=("Microsoft YaHei", 12),
            foreground="#1565c0",
            command=self.redownload_item,
        )
        menu.add_command(
            label=(tr("menu_delete") if tr else "🗑️ 删除"),
            font=("Microsoft YaHei", 12),
            foreground="#ff4444",
            command=self.delete_item,
        )
        menu.post(event.x_root, event.y_root)

    def redownload_item(self):
        try:
            self.list_frame.priority_redownload_video(self.video_id)
        except Exception:
            pass

    def delete_item(self):
        tr = self.list_frame._app.tr if (self.list_frame and self.list_frame._app and hasattr(self.list_frame._app, "tr")) else None
        result = messagebox.askyesno(
            tr("delete_confirm_title") if tr else "确认删除",
            tr("delete_video_confirm", name=self.video['title'][:50]) if tr else f"确定要删除「{self.video['title'][:50]}」吗？\n\n已提取的帧也会被删除。",
            icon="warning",
        )
        if result:
            if self.extractor:
                self.extractor.stop()
            self.task_manager.merge_seen_video_ids_batch(self.task_id, [self.video_id])
            self.task_manager.delete_thumbnail(self.task_id, self.video["video_id"])
            self.task_manager.delete_video_frames(self.task_id, self.video["title"])
            self.task_manager.update_video_extraction_status(self.task_id, self.video_id, "已删除", 0, 0)
            if self.on_delete:
                self.on_delete(self.video_id)

    def pause_extraction(self):
        if "已完成" in self._status_text:
            return False
        if self.extractor:
            self.extractor.pause()
            self.update_status("⏸️ 已暂停", self.progress)
            return True
        return False

    def resume_extraction(self):
        if "已完成" in self._status_text:
            return False
        if self.extractor:
            self.extractor.resume()
            self.update_status(f"▶️ 继续提取中... ({self.progress}%)", self.progress)
            return True
        return False

    def start_extraction(self):
        if "已完成" in self._status_text:
            return
        if self.is_extracting:
            return
        cached_status = self.task_manager.get_video_extraction_status(self.task_id, self.video_id)
        if cached_status.get("status") == "已完成":
            return

        self.is_extracting = True
        lf = self.list_frame

        def progress_callback(message, progress):
            def up():
                try:
                    self.update_status(message, progress)
                except Exception:
                    pass

            lf.after(0, up)

        def extraction_thread():
            self.extractor = FrameExtractor(
                self.video["url"],
                self.video["title"],
                self.video["video_id"],
                self.task_id,
                self.task_manager,
                frame_interval=1,
            )
            if self.task_manager.is_task_paused(self.task_id):
                self.extractor.pause()
                self.update_status("⏸️ 已暂停 (等待恢复)", 0)

            frames_path, frame_count = self.extractor.extract(progress_callback)

            def finish():
                self.is_extracting = False
                try:
                    if frames_path:
                        self.update_status(f"✅ 已完成 ({frame_count}帧)", 100)
                        self.task_manager.reset_retry_count(self.task_id, self.video_id)
                    else:
                        self.update_status("❌ 失败", 0)
                        self.task_manager.mark_video_failed(self.task_id, self.video_id)
                except Exception:
                    pass

            lf.after(0, finish)

        threading.Thread(target=extraction_thread, daemon=True).start()


class _HoverTitleTooltip:
    """鼠标悬停在标题上时显示完整标题（Tk 浮层）。"""

    def __init__(self, widget, get_full_text):
        self.widget = widget
        self.get_full_text = get_full_text
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _hide(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _show(self):
        text = (self.get_full_text() or "").strip()
        if not text:
            return
        self._hide()
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget.winfo_toplevel())
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self._tip.configure(bg="#2b2b2b")
        lbl = tk.Label(
            self._tip,
            text=text,
            font=("Microsoft YaHei", 10),
            bg="#2b2b2b",
            fg="#ffffff",
            justify="left",
            wraplength=420,
            padx=10,
            pady=6,
        )
        lbl.pack()
        self._tip.update_idletasks()
        self._tip.geometry(f"+{x}+{y}")

    def _on_enter(self, _e):
        self._hide()
        full = (self.get_full_text() or "").strip()
        if not full:
            return
        if len(full) <= 22:
            return
        self._after_id = self.widget.after(350, self._show)

    def _on_leave(self, _e):
        self._hide()


class FrameGridCell(ctk.CTkFrame):
    """单格：封面在上，标题与状态在下；时长叠在封面上。"""

    def __init__(self, parent, list_frame, video_id, title="", duration="", **kwargs):
        super().__init__(parent, corner_radius=6, fg_color="#ffffff", border_width=1, border_color="#e6e6e6", **kwargs)
        self.list_frame = list_frame
        self.video_id = video_id
        self._thumb_label = None
        tw = 120
        th = 68
        self._full_title = (title or "未命名").strip() or "未命名"
        title_s = self._full_title[:22] + ("…" if len(self._full_title) > 22 else "")
        dur_s = str(duration or "未知时长")[:20]
        # 上：封面 + 时长角标
        self.thumb_slot = ctk.CTkFrame(self, fg_color="#e8e8e8", width=tw, height=th)
        self.thumb_slot.pack(fill="x", padx=5, pady=(4, 2))
        self._thumb_inner = ctk.CTkFrame(self.thumb_slot, fg_color="#e8e8e8")
        self._thumb_inner.place(x=0, y=0, relwidth=1, relheight=1)
        ph = ctk.CTkLabel(
            self._thumb_inner,
            text="···",
            width=tw,
            height=th,
            fg_color="#e8e8e8",
            text_color="#bbbbbb",
            font=ctk.CTkFont(size=10),
        )
        ph.pack(fill="both", expand=True)
        self._thumb_placeholder = ph
        self._dur_overlay = ctk.CTkLabel(
            self.thumb_slot,
            text=dur_s,
            font=ctk.CTkFont(family="Microsoft YaHei", size=9, weight="bold"),
            fg_color="#333333",
            text_color="#ffffff",
            corner_radius=4,
            padx=5,
            pady=2,
        )
        self._dur_overlay.place(relx=0.98, rely=0.98, anchor="se")
        self._dur_overlay.lift()
        # 下：标题（悬停可看全文）
        self.title_lbl = ctk.CTkLabel(
            self,
            text=title_s,
            font=ctk.CTkFont(family="Microsoft YaHei", size=10, weight="bold"),
            text_color="#222",
            wraplength=130,
            anchor="w",
            justify="left",
        )
        self.title_lbl._last_display_title = title_s
        self.title_lbl.pack(fill="x", padx=5, pady=(2, 0))
        _HoverTitleTooltip(self.title_lbl, lambda: self._full_title)
        # 状态：多行完整展示（宽度内自动换行；宫格外层可纵向滚动，不裁切）
        self.status_lbl = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(family="Microsoft YaHei", size=9),
            text_color="#444444",
            wraplength=118,
            anchor="nw",
            justify="left",
        )
        self.status_lbl.pack(fill="x", padx=5, pady=(0, 4))
        self.bind("<Button-3>", self._menu)
        self.thumb_slot.bind("<Button-3>", self._menu)
        self._thumb_inner.bind("<Button-3>", self._menu)
        self.status_lbl.bind("<Button-3>", self._menu)
        self.title_lbl.bind("<Button-3>", self._menu)
        self._dur_overlay.bind("<Button-3>", self._menu)

    def _menu(self, event):
        item = self.list_frame.download_items.get(self.video_id)
        if item:
            item.show_context_menu(event)

    def set_status_text(self, text):
        raw = (text or "").strip()
        self.status_lbl.configure(text=raw)
        col = "#444444"
        if "已完成" in raw or "✅" in raw:
            col = "#1b5e20"
        elif "失败" in raw or "❌" in raw:
            col = "#b71c1c"
        elif "提取" in raw or "排队" in raw or "等待" in raw or "暂停" in raw or "▶" in raw or "⏸" in raw:
            col = "#0d47a1"
        self.status_lbl.configure(text_color=col)

    def apply_thumbnail_pil(self, pil_image):
        try:
            tw, th = 120, 68
            if self._thumb_placeholder:
                self._thumb_placeholder.destroy()
                self._thumb_placeholder = None
            for w in self._thumb_inner.winfo_children():
                w.destroy()
            if pil_image is None:
                ctk.CTkLabel(self._thumb_inner, text="无", width=tw, height=th, fg_color="#e0e0e0").pack(fill="both", expand=True)
            else:
                img = pil_to_ctk_image_safe(pil_image, tw, th)
                self._thumb_label = ctk.CTkLabel(self._thumb_inner, image=img, text="")
                self._thumb_label.pack(fill="both", expand=True)
                self._thumb_label.bind("<Button-3>", self._menu)
            try:
                self._dur_overlay.lift()
            except Exception:
                pass
        except Exception:
            pass


class DownloadListFrame(ctk.CTkFrame):
    MAX_CONCURRENT_EXTRACTIONS = MAX_PARALLEL_EXTRACT_BROWSERS
    MAX_RETRY_COUNT = 2
    GRID_COLS = 8
    GRID_ROWS = 5
    PAGE_SIZE = 40

    def __init__(self, parent, task_manager, task_id, defer_covers_until_resume=False, app=None, **kwargs):
        super().__init__(parent, fg_color="#f8f9fa", corner_radius=0, **kwargs)
        self.task_manager = task_manager
        self.task_id = task_id
        self._app = app
        def _tr_local(key, **kwargs):
            if self._app and hasattr(self._app, "tr"):
                return self._app.tr(key, **kwargs)
            text = I18N["zh"].get(key, key)
            if kwargs:
                try:
                    return text.format(**kwargs)
                except Exception:
                    return text
            return text
        self._tr_local = _tr_local
        self._suppress_covers_until_resume = bool(defer_covers_until_resume)
        self._page_nav_programmatic = False
        self.width_expanded = 1
        self.download_items = {}
        self._video_payload = {}
        self._frame_extract_visible = False
        self._video_order = []
        self._current_page = 0
        self._page_cells = {}
        self.extraction_queue = []
        self.active_extractions = set()
        # 须用 RLock：_process_queue 在持锁时可能经 restore_from_cache→update_status→update_count_display
        # 触发 _refresh_download_badge，其内部会再次 with queue_lock；普通 Lock 会自死锁（界面卡死）。
        self.queue_lock = threading.RLock()
        self.is_processing_queue = False

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.header = ctk.CTkFrame(self.content_frame, fg_color="#e6e6e6", corner_radius=10, height=50)
        self.header.pack(fill="x", padx=0, pady=(0, 6))

        self.title_label = ctk.CTkLabel(
            self.header,
            text=self._tr_local("download_list_title"),
            font=ctk.CTkFont(family="Microsoft YaHei", size=14, weight="bold"),
            text_color="#333",
        )
        self.title_label.pack(side="left", padx=15, pady=10)

        self.count_label = ctk.CTkLabel(
            self.header,
            text="(0)",
            font=ctk.CTkFont(family="Microsoft YaHei", size=12, weight="bold"),
            text_color="#5c9eff",
        )
        self.count_label.pack(side="left", padx=(0, 10), pady=10)

        control_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        control_frame.pack(side="right", padx=10, pady=8)

        self.pause_btn = ctk.CTkButton(
            control_frame,
            text=self._tr_local("btn_pause_download"),
            width=(140 if (self._app and getattr(self._app, "lang", "zh") == "en") else 90),
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ff9800",
            hover_color="#f57c00",
            text_color="#ffffff",
            corner_radius=6,
            command=self.pause_all_downloads,
            state="normal",
        )
        self.pause_btn.pack(side="left", padx=2)

        self.resume_btn = ctk.CTkButton(
            control_frame,
            text=self._tr_local("btn_resume_download"),
            width=(140 if (self._app and getattr(self._app, "lang", "zh") == "en") else 90),
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#4caf50",
            hover_color="#45a049",
            text_color="#ffffff",
            corner_radius=6,
            command=self.resume_all_downloads,
            state="disabled",
        )
        self.resume_btn.pack(side="left", padx=2)

        nav = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        nav.pack(fill="x", pady=(0, 6))
        self._prev_page_btn = ctk.CTkButton(
            nav,
            text=self._tr_local("page_prev"),
            width=88,
            height=28,
            font=ctk.CTkFont(family="Microsoft YaHei", size=12),
            command=self._go_prev_page,
        )
        self._prev_page_btn.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            nav, text=self._tr_local("page_prefix"), font=ctk.CTkFont(family="Microsoft YaHei", size=12), text_color="#555"
        ).pack(side="left", padx=(4, 2))
        self._page_num_box = ctk.CTkFrame(
            nav, fg_color="#ffffff", border_width=1, border_color="#c8c8c8", corner_radius=4, width=48, height=28
        )
        self._page_num_box.pack(side="left", padx=2)
        self._page_num_box.pack_propagate(False)
        self._page_num_entry = ctk.CTkEntry(
            self._page_num_box,
            width=40,
            height=22,
            font=ctk.CTkFont(family="Microsoft YaHei", size=12),
            border_width=0,
            fg_color="#ffffff",
            justify="center",
        )
        self._page_num_entry.pack(expand=True, padx=4, pady=2)
        self._page_num_entry.bind("<Return>", lambda _e: self._commit_page_entry())
        self._page_num_entry.bind("<FocusOut>", lambda _e: self._commit_page_entry())
        self._page_total_suffix = ctk.CTkLabel(
            nav, text=self._tr_local("page_suffix", total=1), font=ctk.CTkFont(family="Microsoft YaHei", size=12), text_color="#555"
        )
        self._page_total_suffix.pack(side="left", padx=(2, 0))
        self._page_count_label = ctk.CTkLabel(
            nav, text=self._tr_local("page_total_items", count=0), font=ctk.CTkFont(family="Microsoft YaHei", size=12), text_color="#555"
        )
        self._page_count_label.pack(side="left", padx=(0, 8))
        self._next_page_btn = ctk.CTkButton(
            nav,
            text=self._tr_local("page_next"),
            width=88,
            height=28,
            font=ctk.CTkFont(family="Microsoft YaHei", size=12),
            command=self._go_next_page,
        )
        self._next_page_btn.pack(side="left", padx=8)

        self._grid_scroll = ctk.CTkScrollableFrame(
            self.content_frame, fg_color="#eeeeee", corner_radius=8, orientation="vertical"
        )
        self._grid_scroll.pack(fill="both", expand=True)
        self._grid_container = ctk.CTkFrame(self._grid_scroll, fg_color="transparent")
        self._grid_container.pack(fill="both", expand=True)
        for c in range(self.GRID_COLS):
            self._grid_container.grid_columnconfigure(c, weight=1, uniform="gc")
        for r in range(self.GRID_ROWS):
            self._grid_container.grid_rowconfigure(r, weight=0)

        self._thumb_queue = Queue()
        self._thumb_enqueued = set()
        self._thumb_worker_started = False
        self._RIGHT_LIST_THUMB_STAGGER_MS = 180

        self.update_button_states()
        self._update_page_nav_ui()

    def _total_pages(self):
        n = len(self._video_order)
        return max(1, (n + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

    def _go_prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self.render_current_page()

    def _go_next_page(self):
        tp = self._total_pages()
        if self._current_page < tp - 1:
            self._current_page += 1
            self.render_current_page()

    def _commit_page_entry(self):
        if getattr(self, "_page_nav_programmatic", False):
            return
        try:
            s = self._page_num_entry.get().strip()
            if not s:
                return
            p = int(s)
        except (ValueError, TypeError):
            return
        tp = self._total_pages()
        # 只做越界判断：超出范围则保持当前页，不自动改写用户输入值
        if p < 1 or p > tp:
            return
        self._current_page = p - 1
        self.render_current_page()

    def _page_index_for_first_processing_video(self):
        """当前队列中最靠前「正在处理」的视频所在页（0-based 页索引）。"""
        for i, vid in enumerate(self._video_order):
            if vid in self.active_extractions:
                return i // self.PAGE_SIZE
            st = self.task_manager.get_video_extraction_status(self.task_id, vid)
            if st.get("status") == "提取中":
                return i // self.PAGE_SIZE
            item = self.download_items.get(vid)
            if item:
                t = item.get_status_text() or ""
                if "已完成" in t:
                    continue
                if (
                    "提取中" in t
                    or "继续提取" in t
                    or ("提取" in t and "帧" in t and "失败" not in t)
                    or ("⏳" in t and "提取" in t)
                    or ("▶" in t and "提取" in t)
                ):
                    return i // self.PAGE_SIZE
        return 0

    def jump_to_active_processing_page(self):
        self._current_page = self._page_index_for_first_processing_video()

    def priority_redownload_video(self, video_id):
        video = self._video_payload.get(video_id)
        if not video:
            return
        self._register_fec_for_video(video)
        item = self.download_items.get(video_id)
        if item is None:
            item = self.instantiate_item_for_worker_if_missing(video_id)
        if not item:
            return
        try:
            if getattr(item, "extractor", None):
                item.extractor.stop()
        except Exception:
            pass
        item.is_extracting = False
        self.task_manager.reset_retry_count(self.task_id, video_id)
        with self.queue_lock:
            self.extraction_queue = [(vid, v) for vid, v in self.extraction_queue if vid != video_id]
            self.active_extractions.discard(video_id)
        self.task_manager.update_video_extraction_status(
            self.task_id, video_id, "⏳ 排队中... (优先重下)", 0, 0, -1
        )
        item.update_status("⏳ 排队中... (优先重下)", 0)
        with self.queue_lock:
            self.extraction_queue.insert(0, (video_id, video))
        self._process_queue()
        self.update_button_states()
        try:
            self.render_current_page()
        except Exception:
            pass

    def _update_page_nav_ui(self):
        tp = self._total_pages()
        if self._current_page >= tp:
            self._current_page = max(0, tp - 1)
        n = len(self._video_order)
        self._page_nav_programmatic = True
        try:
            try:
                self._page_num_entry.delete(0, "end")
                self._page_num_entry.insert(0, str(self._current_page + 1))
            except Exception:
                pass
            self._page_total_suffix.configure(text=self._tr_local("page_suffix", total=tp))
            self._page_count_label.configure(text=self._tr_local("page_total_items", count=n))
        finally:
            self._page_nav_programmatic = False
        self._prev_page_btn.configure(state="normal" if self._current_page > 0 else "disabled")
        self._next_page_btn.configure(state="normal" if self._current_page < tp - 1 else "disabled")

    def set_frame_extract_visible(self, visible):
        self._frame_extract_visible = bool(visible)

    def _register_fec_for_video(self, video):
        video_id = video["video_id"]
        task = self.task_manager.get_task(self.task_id)
        if task is None:
            return
        fec = dict(task.get("frame_extract_cache", {}) or {})
        if video_id not in fec or fec.get(video_id) is None:
            fec[video_id] = {
                "video": video,
                "status": "等待中",
                "status_text": "等待中",
                "progress": 0,
                "frame_count": 0,
                "last_frame_time": -1,
                "updated_at": datetime.now().isoformat(),
            }
            self.task_manager.update_task_info(self.task_id, {"frame_extract_cache": fec})
        else:
            if fec[video_id].get("video") is None:
                fec[video_id]["video"] = video
                self.task_manager.update_task_info(self.task_id, {"frame_extract_cache": fec})

    def _create_minimal_item(self, video):
        return MinimalExtractionItem(
            self,
            video,
            self.task_manager,
            self.task_id,
            on_status_change=self.on_status_change,
            on_delete=self.on_delete_item,
        )

    def instantiate_item_for_worker_if_missing(self, video_id):
        """仅当前页需要展示时创建逻辑体（含状态/右键菜单）；未翻到的页不实例化。"""
        if video_id in self.download_items:
            return self.download_items[video_id]
        video = self._video_payload.get(video_id)
        if not video:
            return None
        self._register_fec_for_video(video)
        item = self._create_minimal_item(video)
        self.download_items[video_id] = item
        item.restore_from_cache()
        return item

    def _refill_extraction_queue_after_hydrate(self):
        """hydrate 顺序与元数据后，按磁盘状态填满提取队列（不创建未翻页 UI）。"""
        _resume_flow_dbg("DownloadList._refill.enter", f"task={self.task_id} n_order={len(self._video_order)}")
        with self.queue_lock:
            self.extraction_queue.clear()
        if self.task_manager.is_task_paused(self.task_id):
            _resume_flow_dbg("DownloadList._refill.return_paused", f"task={self.task_id}")
            return
        for video_id in self._video_order:
            video = self._video_payload.get(video_id)
            if not video:
                continue
            task = self.task_manager.get_task(self.task_id)
            fec = (task or {}).get("frame_extract_cache", {}) or {}
            ent = fec.get(video_id) or {}
            if ent.get("status_text") == "已删除":
                continue
            cached_status = self.task_manager.get_video_extraction_status(self.task_id, video_id)
            if cached_status.get("status") == "已完成":
                continue
            if cached_status.get("status") == "提取中":
                with self.queue_lock:
                    self.extraction_queue.append((video_id, video))
                continue
            if cached_status.get("status") == "失败":
                with self.queue_lock:
                    self.extraction_queue.append((video_id, video))
                continue
            with self.queue_lock:
                self.extraction_queue.append((video_id, video))
        _resume_flow_dbg(
            "DownloadList._refill.before_process_queue",
            f"task={self.task_id} q={len(self.extraction_queue)}",
        )
        self._process_queue()
        _resume_flow_dbg("DownloadList._refill.after_process_queue", f"task={self.task_id}")

    def render_current_page(self):
        for w in self._grid_container.winfo_children():
            w.destroy()
        self._page_cells.clear()
        start = self._current_page * self.PAGE_SIZE
        slice_ids = self._video_order[start : start + self.PAGE_SIZE]
        for i, vid in enumerate(slice_ids):
            r, c = divmod(i, self.GRID_COLS)
            video = self._video_payload.get(vid) or {}
            title = (video.get("title") or "") if isinstance(video, dict) else ""
            dur = (video.get("duration") or "未知时长") if isinstance(video, dict) else "未知时长"
            item = self.instantiate_item_for_worker_if_missing(vid)
            if not item:
                continue
            cell = FrameGridCell(self._grid_container, self, vid, title=title, duration=dur)
            cell.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            self._page_cells[vid] = cell
            cell.set_status_text(self._status_text_for_display(item.get_status_text()))
        self._update_page_nav_ui()
        self._thumb_request_visible_page()

    def refresh_item_ui(self, video_id):
        cell = self._page_cells.get(video_id)
        if not cell:
            return
        item = self.download_items.get(video_id)
        if item:
            cell.set_status_text(self._status_text_for_display(item.get_status_text()))

    def _thumb_request_visible_page(self):
        if getattr(self, "_suppress_covers_until_resume", False):
            return
        for vid in list(self._page_cells.keys()):
            self._request_thumb_async(vid)

    def _idx_on_current_page(self, idx):
        start = self._current_page * self.PAGE_SIZE
        return start <= idx < start + self.PAGE_SIZE

    def _ensure_thumb_worker(self):
        if self._thumb_worker_started:
            return
        self._thumb_worker_started = True
        list_ref = self

        def worker():
            while True:
                try:
                    vid = list_ref._thumb_queue.get(timeout=0.5)
                except Empty:
                    continue
                pil = None
                try:
                    pil = list_ref.task_manager.load_thumbnail(list_ref.task_id, vid)
                    if pil is not None:
                        list_ref.after(0, lambda v=vid, p=pil: list_ref._apply_thumb_ui(v, p))
                        continue
                    pil = download_thumbnail(vid)
                    list_ref.task_manager.save_thumbnail(list_ref.task_id, vid, pil)
                except Exception:
                    pil = None
                list_ref.after(0, lambda v=vid, p=pil: list_ref._apply_thumb_ui(v, p))

        threading.Thread(target=worker, daemon=True).start()

    def _request_thumb_async(self, video_id):
        """缺本地封面时加入队列，按 FIFO 在后台逐个下载。"""
        if getattr(self, "_suppress_covers_until_resume", False):
            return
        if video_id in self._thumb_enqueued:
            return
        self._thumb_enqueued.add(video_id)
        self._thumb_queue.put(video_id)
        self._ensure_thumb_worker()

    def _apply_thumb_ui(self, video_id, pil_image=None):
        self._thumb_enqueued.discard(video_id)
        cell = self._page_cells.get(video_id)
        if cell:
            try:
                cell.apply_thumbnail_pil(pil_image)
            except Exception:
                traceback.print_exc()

    def _status_text_for_display(self, text):
        if not (self._app and hasattr(self._app, "lang") and self._app.lang == "en"):
            return text
        s = str(text or "")
        mapping = [
            ("已完成", "Completed"),
            ("失败", "Failed"),
            ("等待中", "Waiting"),
            ("排队中", "Queued"),
            ("提取中", "Extracting"),
            ("继续提取", "Resuming extraction"),
            ("已暂停", "Paused"),
            ("任务暂停", "Task paused"),
            ("等待恢复", "Waiting resume"),
            ("等待空闲", "Waiting for idle slot"),
            ("正在打开视频页面", "Opening video page"),
            ("视频已加载，开始提取帧", "Video loaded, extracting frames"),
            ("正在等待广告结束或跳过", "Waiting for ad to finish/skip"),
            ("无法获取浏览器驱动", "Cannot get browser driver"),
            ("提取失败", "Extraction failed"),
            ("检测到已有", "Detected existing"),
            ("将从", "resuming from"),
            ("已提取", "Extracted"),
            ("提取完成", "Extraction complete"),
            ("准备重试", "Preparing retry"),
            ("重试", "Retry"),
            ("秒后", "s later"),
            ("帧", "frames"),
        ]
        for zh, en in mapping:
            s = s.replace(zh, en)
        return s

    def schedule_missing_thumbnails_in_order(self, slow=False):
        """按当前页顺序把封面交给后台队列（仅当前页最多一页格数）。"""
        if getattr(self, "_suppress_covers_until_resume", False):
            return
        vids = list(self._page_cells.keys())
        if not vids:
            return

        def enqueue_one(vid):
            try:
                self._thumb_enqueued.discard(vid)
                self._request_thumb_async(vid)
            except Exception:
                pass

        if not slow:
            for vid in vids:
                enqueue_one(vid)
            return

        gap = max(40, int(getattr(self, '_RIGHT_LIST_THUMB_STAGGER_MS', 180)))

        def step(i):
            if i >= len(vids):
                return
            enqueue_one(vids[i])
            self.after(gap, lambda: step(i + 1))

        step(0)

    def _is_video_completed(self, item):
        status_text = item.get_status_text() if hasattr(item, "get_status_text") else ""
        return "已完成" in status_text

    def _is_video_failed(self, item):
        status_text = item.get_status_text() if hasattr(item, "get_status_text") else ""
        return "失败" in status_text and "已完成" not in status_text

    def _has_incomplete_videos(self):
        """检查是否还有未完成的视频（含未实例化条目的任务级状态）。"""
        for video_id in self._video_order:
            task = self.task_manager.get_task(self.task_id)
            fec = (task or {}).get("frame_extract_cache", {}) or {}
            ent = fec.get(video_id) or {}
            if ent.get("status_text") == "已删除":
                continue
            item = self.download_items.get(video_id)
            if item:
                status_text = item.get_status_text() if hasattr(item, "get_status_text") else ""
            else:
                cached = self.task_manager.get_video_extraction_status(self.task_id, video_id)
                status_text = cached.get("status_text") or ent.get("status_text") or ""
            if "已完成" not in status_text and "失败" not in status_text and "已删除" not in status_text:
                return True
        return False

    def update_button_states(self):
        """更新按钮状态：如果只剩下失败和已完成，则禁用暂停按钮，启用继续按钮（用于重试失败视频）"""
        is_paused = self.task_manager.is_task_paused(self.task_id) if self.task_id else False
        has_incomplete = self._has_incomplete_videos()
        has_failed = False
        for video_id in self._video_order:
            item = self.download_items.get(video_id)
            if item:
                if self._is_video_failed(item):
                    has_failed = True
                    break
            else:
                cached = self.task_manager.get_video_extraction_status(self.task_id, video_id)
                st = cached.get("status_text") or ""
                if "失败" in st and "已完成" not in st:
                    has_failed = True
                    break

        if not has_incomplete:
            self.pause_btn.configure(state="disabled", fg_color="#cccccc", hover_color="#cccccc")
            if has_failed:
                self.resume_btn.configure(state="normal", fg_color="#4caf50", hover_color="#45a049")
            else:
                self.resume_btn.configure(state="disabled", fg_color="#cccccc", hover_color="#cccccc")
        elif is_paused:
            self.pause_btn.configure(state="disabled", fg_color="#cccccc", hover_color="#cccccc")
            self.resume_btn.configure(state="normal", fg_color="#4caf50", hover_color="#45a049")
        else:
            self.pause_btn.configure(state="normal", fg_color="#ff9800", hover_color="#f57c00")
            self.resume_btn.configure(state="disabled", fg_color="#cccccc", hover_color="#cccccc")

    def update_count_display(self):
        count = len(self._video_order)
        completed_count = 0
        for video_id in self._video_order:
            cached = self.task_manager.get_video_extraction_status(self.task_id, video_id)
            if cached.get("status") == "已完成":
                completed_count += 1
                continue
            item = self.download_items.get(video_id)
            if item:
                status_text = item.get_status_text() if hasattr(item, "get_status_text") else ""
                if "已完成" in status_text:
                    completed_count += 1

        target_total = None
        try:
            t = self.task_manager.get_task(self.task_id) if self.task_id else None
            target_total = (t or {}).get("target_video_count")
        except Exception:
            target_total = None
        self.count_label.configure(text=format_completed_progress_text(completed_count, count, target_total))
        self._update_page_nav_ui()
        self.update_button_states()
        if self._app and hasattr(self._app, "_refresh_download_badge"):
            try:
                self._app._refresh_download_badge()
            except Exception:
                pass

    def _on_extraction_complete(self, video_id):
        with self.queue_lock:
            if video_id in self.active_extractions:
                self.active_extractions.remove(video_id)
        self._process_queue()
        self.update_button_states()

    def _on_extraction_failed(self, video_id):
        """重试耗尽或不可重试的失败：删除该视频帧目录（清空残留），暂停/中途退出不走此路径。"""
        video = (self._video_payload.get(video_id) or {})
        title = video.get("title")
        if title:
            try:
                self.task_manager.delete_video_frames(self.task_id, title)
            except Exception:
                pass
        self._on_extraction_complete(video_id)

    def _process_queue(self):
        _resume_flow_dbg("_process_queue.enter", f"task={self.task_id}")
        with self.queue_lock:
            if self.is_processing_queue:
                _resume_flow_dbg("_process_queue.reentrant_skip", f"task={self.task_id}")
                return
            self.is_processing_queue = True

            try:
                if self.task_manager.is_task_paused(self.task_id):
                    _resume_flow_dbg("_process_queue.return_paused", f"task={self.task_id}")
                    return

                _n_started = 0
                while len(self.active_extractions) < self.MAX_CONCURRENT_EXTRACTIONS and self.extraction_queue:
                    video_id, video = self.extraction_queue.pop(0)

                    if video_id not in self.download_items:
                        if not video:
                            video = self._video_payload.get(video_id)
                        if not video:
                            continue
                        self._register_fec_for_video(video)
                        item = self._create_minimal_item(video)
                        self.download_items[video_id] = item
                        item.restore_from_cache()

                    item = self.download_items.get(video_id)
                    if not item:
                        continue
                    cached_status = self.task_manager.get_video_extraction_status(self.task_id, video_id)

                    if cached_status.get('status') in ["已完成"]:
                        continue
                    if self._is_video_completed(item):
                        continue

                    retry_count = self.task_manager.get_retry_count(self.task_id, video_id)
                    if retry_count >= self.MAX_RETRY_COUNT and cached_status.get('status') == "失败":
                        item.update_status(f"❌ 失败 (已重试{self.MAX_RETRY_COUNT}次)", 0)
                        continue

                    self.active_extractions.add(video_id)
                    if _n_started < 4:
                        _resume_flow_dbg(
                            "_process_queue.start_extraction",
                            f"task={self.task_id} vid={video_id} slot={_n_started + 1}",
                        )
                    _n_started += 1
                    self._start_extraction_with_callback(item, video_id)
                if _n_started:
                    _resume_flow_dbg("_process_queue.started_n", f"task={self.task_id} n={_n_started}")
            finally:
                self.is_processing_queue = False
        _resume_flow_dbg("_process_queue.exit", f"task={self.task_id}")

    def _start_extraction_with_callback(self, item, video_id):
        original_update_status = item.update_status

        def should_retry_fast(error_msg):
            no_retry_keywords = ["视频不存在", "无法获取视频时长", "403", "404"]
            return not any(kw in error_msg for kw in no_retry_keywords)

        def get_retry_delay(retry_count):
            return min(30, 2 ** retry_count)

        def wrapped_update_status(status_text, progress=0, save_to_cache=True):
            original_update_status(status_text, progress, save_to_cache)

            if "已完成" in status_text:
                self.after(0, lambda: self._on_extraction_complete(video_id))
            elif "失败" in status_text:
                retry_count = self.task_manager.increment_retry_count(self.task_id, video_id)

                if retry_count < self.MAX_RETRY_COUNT and should_retry_fast(status_text):
                    delay = get_retry_delay(retry_count)
                    with self.queue_lock:
                        self.extraction_queue.insert(0, (video_id, item.video))
                        item.update_status(f"🔄 重试 {retry_count + 1}/{self.MAX_RETRY_COUNT} ({delay}秒后)...", 0)
                        self.after(int(delay * 1000), lambda: self._process_queue())
                else:
                    self.after(0, lambda: self._on_extraction_failed(video_id))

        item.update_status = wrapped_update_status
        item.start_extraction()

    def add_video_to_queue(self, video, load_thumbnails=True):
        video_id = video['video_id']
        _trace("add_video_to_queue", f"task={self.task_id} video_id={video_id} thumbs={load_thumbnails}")
        if video_id in self.download_items:
            return

        self.task_manager.merge_seen_video_ids_batch(self.task_id, [video_id])
        try:
            if self._app is not None:
                self._app._register_seen_video(self.task_id, video_id)
        except Exception:
            pass

        self._register_fec_for_video(video)
        self._video_payload[video_id] = video
        if video_id not in self._video_order:
            self._video_order.append(video_id)
        idx = self._video_order.index(video_id)

        item = self._create_minimal_item(video)
        self.download_items[video_id] = item
        item.restore_from_cache()
        if self._frame_extract_visible and self._idx_on_current_page(idx):
            self.render_current_page()
        else:
            self.update_count_display()

        cached_status = self.task_manager.get_video_extraction_status(self.task_id, video_id)

        if cached_status.get('status') == "已完成":
            item.update_status(f"✅ 已完成 ({cached_status.get('frame_count', 0)}帧)", 100)
            self.update_button_states()
            return

        if self.task_manager.is_task_paused(self.task_id):
            item.update_status("⏸️ 已暂停 (任务暂停)", 0)
            self.update_button_states()
            return

        if cached_status.get('status') == "提取中":
            with self.queue_lock:
                self.active_extractions.add(video_id)
            self._start_extraction_with_callback(item, video_id)
            self.update_button_states()
            return

        with self.queue_lock:
            self.extraction_queue.append((video_id, video))
        item.update_status("⏳ 排队中... (等待空闲)", 0)
        self._process_queue()
        self.update_button_states()

    def add_video(self, video, load_thumbnails=True):
        self.add_video_to_queue(video, load_thumbnails=load_thumbnails)

    def on_delete_item(self, video_id):
        try:
            self.task_manager.merge_seen_video_ids_batch(self.task_id, [video_id])
        except Exception:
            pass
        try:
            if self._app is not None:
                self._app._register_seen_video(self.task_id, video_id)
        except Exception:
            pass
        if video_id in self.download_items:
            with self.queue_lock:
                self.extraction_queue = [(vid, v) for vid, v in self.extraction_queue if vid != video_id]
                if video_id in self.active_extractions:
                    self.active_extractions.remove(video_id)

            item = self.download_items[video_id]
            try:
                if getattr(item, "extractor", None):
                    item.extractor.stop()
            except Exception:
                pass
            del self.download_items[video_id]
            if video_id in self._video_order:
                self._video_order.remove(video_id)
        self._video_payload.pop(video_id, None)

        # 删除帧提取专用缓存条目（重启不会再恢复该视频）
        task = self.task_manager.get_task(self.task_id)
        if task is not None:
            fec = dict(task.get('frame_extract_cache', {}) or {})
            if video_id in fec:
                del fec[video_id]
                self.task_manager.update_task_info(self.task_id, {'frame_extract_cache': fec})

        self.update_count_display()
        self._process_queue()
        self.update_button_states()
        try:
            self.render_current_page()
        except Exception:
            pass

    def on_status_change(self, video_id, status_text, progress=0):
        self.update_button_states()

    def clear_all(self):
        _resume_flow_dbg("DownloadList.clear_all.enter", f"task={self.task_id} actives={len(self.active_extractions)}")
        with self.queue_lock:
            for video_id in list(self.active_extractions):
                if video_id in self.download_items:
                    item = self.download_items[video_id]
                    if hasattr(item, 'extractor') and item.extractor:
                        item.extractor.stop()
            self.active_extractions.clear()
            self.extraction_queue.clear()

        self.download_items.clear()
        self._video_payload.clear()
        self._video_order.clear()
        self._current_page = 0
        try:
            for w in self._grid_container.winfo_children():
                w.destroy()
        except Exception:
            pass
        self._page_cells.clear()
        self.update_count_display()
        self.update_button_states()
        _resume_flow_dbg("DownloadList.clear_all.done", f"task={self.task_id}")

    def pause_all_downloads(self):
        if not self.task_id:
            return
        self.task_manager.set_task_paused(self.task_id, True)
        for video_id, item in self.download_items.items():
            if self._is_video_completed(item):
                continue
            if hasattr(item, 'pause_extraction'):
                item.pause_extraction()
        self.update_button_states()

    def resume_all_downloads(self):
        if not self.task_id:
            return
        self.task_manager.set_task_paused(self.task_id, False)

        videos_to_retry = []
        for video_id, item in self.download_items.items():
            if self._is_video_completed(item):
                continue
            status_text = item.get_status_text() if hasattr(item, "get_status_text") else ""
            if "失败" in status_text:
                videos_to_retry.append((video_id, item.video))
            elif "已暂停" in status_text:
                if hasattr(item, 'resume_extraction'):
                    item.resume_extraction()

        for video_id in self._video_order:
            if video_id in self.download_items:
                continue
            cached = self.task_manager.get_video_extraction_status(self.task_id, video_id)
            if cached.get("status") == "失败":
                video = self._video_payload.get(video_id)
                if video:
                    videos_to_retry.append((video_id, video))

        if videos_to_retry:
            with self.queue_lock:
                for video_id, video in videos_to_retry:
                    self.task_manager.reset_retry_count(self.task_id, video_id)
                    self.task_manager.update_video_extraction_status(
                        self.task_id, video_id, "⏳ 准备重试...", 0, 0, -1
                    )
                    if not any(vid == video_id for vid, _ in self.extraction_queue):
                        self.extraction_queue.append((video_id, video))
                        if video_id in self.download_items:
                            self.download_items[video_id].update_status("⏳ 准备重试...", 0)

        self._process_queue()
        self.update_button_states()
        delay_ms = 120000

        def _later_thumbs():
            try:
                self._RIGHT_LIST_THUMB_STAGGER_MS = max(8000, int(getattr(self, "_RIGHT_LIST_THUMB_STAGGER_MS", 180)))
                self.schedule_missing_thumbnails_in_order(slow=True)
            except Exception:
                pass

        self.after(delay_ms, _later_thumbs)

    def stop_all_threads_for_delete_task(self):
        """删除任务时停止提取队列与 extractor（不关闭浏览器）。"""
        with self.queue_lock:
            for video_id in list(self.active_extractions):
                if video_id in self.download_items:
                    item = self.download_items[video_id]
                    if hasattr(item, "extractor") and item.extractor:
                        item.extractor.stop()
            self.active_extractions.clear()
            self.extraction_queue.clear()

    def set_startup_resume_pending_ui(self):
        """程序再次启动且有待继续：右侧「继续下载」可点，「暂停下载」灰掉。"""
        self.pause_btn.configure(state="disabled", fg_color="#cccccc", hover_color="#cccccc")
        self.resume_btn.configure(state="normal", fg_color="#4caf50", hover_color="#45a049")

    def resume_from_first_incomplete_after_relaunch(self, schedule_thumbnails=False):
        """点击主界面「继续进程」：从第一个非已完成视频起重新排队（含暂停、失败）。封面恢复由 EasyDatasetApp 延后调度。"""
        if not self.task_id:
            return
        _resume_flow_dbg(
            "DownloadList.resume_incomplete.enter",
            f"task={self.task_id} order={len(self._video_order)}",
        )
        self.task_manager.set_task_paused(self.task_id, False)
        with self.queue_lock:
            self.extraction_queue.clear()
            self.active_extractions.clear()
        _ord = list(self._video_order)
        _nord = len(_ord)
        for i, video_id in enumerate(_ord):
            if i == 0 or i == _nord - 1 or (_nord > 15 and i % max(1, _nord // 5) == 0):
                _resume_flow_dbg(
                    "DownloadList.resume_incomplete.loop",
                    f"i={i + 1}/{_nord} vid={video_id}",
                )
            video = self._video_payload.get(video_id)
            if not video:
                continue
            task = self.task_manager.get_task(self.task_id)
            fec = (task or {}).get("frame_extract_cache", {}) or {}
            ent = fec.get(video_id) or {}
            if ent.get("status_text") == "已删除":
                continue
            item = self.download_items.get(video_id)
            if item:
                if self._is_video_completed(item):
                    continue
                st = item.get_status_text() if hasattr(item, "get_status_text") else ""
            else:
                cached = self.task_manager.get_video_extraction_status(self.task_id, video_id)
                if cached.get("status") == "已完成":
                    continue
                st = cached.get("status_text") or ent.get("status_text") or ""
            if "已删除" in st:
                continue
            self.task_manager.reset_retry_count(self.task_id, video_id)
            self.task_manager.update_video_extraction_status(
                self.task_id, video_id, "⏳ 排队中... (继续)", 0, 0, -1
            )
            if item:
                item.update_status("⏳ 排队中... (继续)", 0)
            with self.queue_lock:
                self.extraction_queue.append((video_id, video))
        _resume_flow_dbg(
            "DownloadList.resume_incomplete.before_process_queue",
            f"task={self.task_id} q={len(self.extraction_queue)}",
        )
        self._process_queue()
        _resume_flow_dbg("DownloadList.resume_incomplete.after_process_queue", f"task={self.task_id}")
        self.update_button_states()
        if schedule_thumbnails:
            self.schedule_missing_thumbnails_in_order(slow=True)

    def flush_items_status_to_disk(self):
        """关闭程序前把右侧每条视频的进度文字写回 frame_extract_cache / info.json。"""
        for _vid, item in list(self.download_items.items()):
            try:
                text = item.get_status_text() if hasattr(item, "get_status_text") else ""
                prog = getattr(item, "progress", 0)
                item.update_status(text, prog, save_to_cache=True)
            except Exception:
                pass


class MessageBubble(ctk.CTkFrame):
    def __init__(self, parent, msg, task_manager, task_id, is_user=False, on_selection_change=None, on_confirm=None,
                 user_request=None, app=None, defer_video_cards=False, on_video_cards_finished=None, **kwargs):
        _trace("MessageBubble.__init__", f"task={task_id} type={msg.get('type')} defer_cards={defer_video_cards}")
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self._external_on_selection_change = on_selection_change
        self.on_confirm = on_confirm
        self.user_request = user_request
        self.app = app
        self.msg = msg
        self.task_id = task_id
        self.confirm_button = None
        self.video_frames = []
        self._on_video_cards_finished = on_video_cards_finished
        self._defer_video_cards = defer_video_cards
        self._auto_confirm_after_id = None
        self._auto_confirm_arm_after_id = None
        self._suppress_video_checkbox_cancel = False
        self._video_list_checkbox_ready = False

        if is_user:
            color = "#e6f2ff"
            bd = None
            side = "e"
            pad = (10, 20)
        else:
            color = "#ffffff"
            bd = "#e0e0e0"
            side = "w"
            pad = (20, 10)

        if is_user:
            b = ctk.CTkFrame(self, fg_color=color, corner_radius=8)
        else:
            b = ctk.CTkFrame(self, fg_color=color, corner_radius=8, border_width=1, border_color=bd)
        b.grid(row=0, column=0, sticky=side, padx=pad, pady=6)

        if msg['type'] == 'text':
            if is_user:
                ctk.CTkLabel(b, text=msg['content'], font=ctk.CTkFont(family="Microsoft YaHei", size=14),
                             text_color="#000000", wraplength=550, justify="left").pack(padx=16, pady=12)
            else:
                ctk.CTkLabel(b, text=msg['content'], font=ctk.CTkFont(family="Microsoft YaHei", size=14),
                             wraplength=550, justify="left").pack(padx=16, pady=12)
        elif msg['type'] == 'video_list':
            video_container = ctk.CTkFrame(b, fg_color="transparent")
            video_container.pack(padx=8, pady=8, fill="x")
            total_videos_in_msg = len(msg.get('videos') or [])
            defer = defer_video_cards and total_videos_in_msg > 0
            if defer:
                self._video_container = video_container
                self._video_defer_outer_b = b
                self._defer_task_manager = task_manager
                self._pending_videos = list(msg['videos'])
                self._pending_video_idx = 0
                self._total_videos_in_msg = total_videos_in_msg
                self.after(1, self._deferred_pack_next_video_card)
            else:
                for idx, v in enumerate(msg.get('videos') or [], 1):
                    video_frame = VideoCardFrame(video_container, v, task_manager, task_id, video_index=idx,
                                                 total_videos=total_videos_in_msg,
                                                 on_selection_change=self._video_selection_cb)
                    video_frame.pack(fill="x", pady=4, padx=4)
                    self.video_frames.append(video_frame)
                self._place_video_list_confirm_button(b)

    def _place_video_list_confirm_button(self, b):
        msg = self.msg
        lang = get_saved_language()
        btn_text = I18N["en"]["video_confirm_click"] if lang == "en" else I18N["zh"]["video_confirm_click"]
        done_text = I18N["en"]["video_confirmed"] if lang == "en" else I18N["zh"]["video_confirmed"]
        self.confirm_button = ctk.CTkButton(b, text=btn_text,
                                            font=ctk.CTkFont(family="Microsoft YaHei", size=12, weight="bold"),
                                            text_color="#5c9eff", fg_color="transparent", hover_color="#e6f2ff",
                                            width=80, height=30, command=self.on_confirm_click)
        self.confirm_button.pack(pady=(8, 12))
        if msg.get("selection_confirmed"):
            self.confirm_button.configure(text=done_text, fg_color="#e0e0e0", text_color="#999999", state="disabled")
        else:
            self._video_list_checkbox_ready = False
            self._arm_video_list_auto_confirm_delayed()

    def _arm_video_list_auto_confirm_delayed(self):
        """稍后再挂 5 分钟计时，避免 CheckBox 初始化/布局触发的回调误取消计时。"""
        if self._auto_confirm_arm_after_id is not None:
            try:
                self.after_cancel(self._auto_confirm_arm_after_id)
            except Exception:
                pass
            self._auto_confirm_arm_after_id = None
        self._auto_confirm_arm_after_id = self.after(
            VIDEO_LIST_AUTO_CONFIRM_ARM_DELAY_MS,
            self._on_arm_video_list_auto_confirm_ready,
        )

    def _on_arm_video_list_auto_confirm_ready(self):
        self._auto_confirm_arm_after_id = None
        if self.msg.get("selection_confirmed"):
            return
        self._video_list_checkbox_ready = True
        self._schedule_video_list_auto_confirm()

    def _video_selection_cb(self, video, selected):
        if not getattr(self, "_suppress_video_checkbox_cancel", False):
            # 取消勾选：立刻取消自动确认；其它操作在「就绪」后才视为用户动作（避免初始化误触）
            if not selected:
                self._cancel_arm_video_list_auto_confirm()
                self._cancel_video_list_auto_confirm()
            elif getattr(self, "_video_list_checkbox_ready", False):
                self._cancel_video_list_auto_confirm()
                self._cancel_arm_video_list_auto_confirm()
        if self._external_on_selection_change:
            self._external_on_selection_change(video, selected)

    def _cancel_video_list_auto_confirm(self):
        aid = getattr(self, "_auto_confirm_after_id", None)
        if aid is not None:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
            self._auto_confirm_after_id = None

    def _cancel_arm_video_list_auto_confirm(self):
        aid = getattr(self, "_auto_confirm_arm_after_id", None)
        if aid is not None:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
            self._auto_confirm_arm_after_id = None

    def _schedule_video_list_auto_confirm(self):
        self._cancel_video_list_auto_confirm()
        if self.msg.get("selection_confirmed"):
            return
        self._auto_confirm_after_id = self.after(
            VIDEO_LIST_AUTO_CONFIRM_MS, self._on_video_list_auto_confirm_timeout
        )

    def _on_video_list_auto_confirm_timeout(self):
        self._auto_confirm_after_id = None
        if self.msg.get("selection_confirmed"):
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        self._suppress_video_checkbox_cancel = True
        try:
            for vf in self.video_frames:
                vf.set_selected(True)
        except Exception:
            pass
        finally:
            self._suppress_video_checkbox_cancel = False
        self._execute_video_list_confirm()

    def _execute_video_list_confirm(self):
        """与点击「点击确认」相同：写回消息、回调、按钮变灰（不依赖按钮文字 cget）。"""
        if self.msg.get("selection_confirmed"):
            return
        if not self.confirm_button:
            return
        self._cancel_arm_video_list_auto_confirm()
        self._cancel_video_list_auto_confirm()
        selected_videos = []
        deselected_videos = []
        for video_frame in self.video_frames:
            if video_frame.get_selected():
                selected_videos.append(video_frame.video)
            else:
                deselected_videos.append(video_frame.video)
        if self.msg.get("type") == "video_list":
            self.msg["selection_confirmed"] = True
            if self.app:
                try:
                    st = self.app.get_task_state(self.task_id)
                    self.app.task_manager.save_task_messages(self.task_id, st.get("current_messages", []))
                except Exception:
                    pass
        if self.on_confirm:
            self.after(0, lambda: self.on_confirm(self.user_request, selected_videos, deselected_videos))
        try:
            lang = get_saved_language()
            done_text = I18N["en"]["video_confirmed"] if lang == "en" else I18N["zh"]["video_confirmed"]
            self.confirm_button.configure(text=done_text, fg_color="#e0e0e0", text_color="#999999", state="disabled")
        except Exception:
            pass

    def _deferred_pack_next_video_card(self):
        try:
            if not self.winfo_exists():
                if self._on_video_cards_finished:
                    self._on_video_cards_finished()
                return
        except Exception:
            if self._on_video_cards_finished:
                self._on_video_cards_finished()
            return
        if self._pending_video_idx >= len(self._pending_videos):
            self._place_video_list_confirm_button(self._video_defer_outer_b)
            cb = self._on_video_cards_finished
            self._on_video_cards_finished = None
            if cb:
                cb()
            return
        v = self._pending_videos[self._pending_video_idx]
        idx = self._pending_video_idx + 1
        total = self._total_videos_in_msg
        _trace("_deferred_pack_next_video_card", f"task={self.task_id} idx={idx}/{total} vid={(v or {}).get('video_id')}")
        video_frame = VideoCardFrame(
            self._video_container, v, self._defer_task_manager, self.task_id, video_index=idx,
            total_videos=total, on_selection_change=self._video_selection_cb, defer_image_load=True)
        video_frame.pack(fill="x", pady=4, padx=4)
        self.video_frames.append(video_frame)
        self._pending_video_idx += 1
        delay = 12
        if self.app and hasattr(self.app, "_STARTUP_VIDEO_CARD_DELAY_MS"):
            delay = getattr(self.app, "_STARTUP_VIDEO_CARD_DELAY_MS", delay)
        self.after(delay, self._deferred_pack_next_video_card)

    def on_confirm_click(self):
        self._cancel_video_list_auto_confirm()
        self._cancel_arm_video_list_auto_confirm()
        if not self.confirm_button:
            return
        if self.msg.get("selection_confirmed"):
            return
        self._execute_video_list_confirm()


class TaskItemFrame(ctk.CTkFrame):
    def __init__(self, parent, task, app, **kwargs):
        super().__init__(parent, fg_color="transparent", height=40, **kwargs)
        self.task = task
        self.app = app
        self.pack(fill="x", pady=2)
        self.grid_columnconfigure(0, weight=1)

        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=0, column=0, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)

        is_selected = (self.app.current_task_id == task['task_id'])
        bg_color = "#e6f2ff" if is_selected else "#f8f9fa"
        text_color = "#5c9eff" if is_selected else "#333333"

        self.task_btn = ctk.CTkButton(self.button_frame, text=task['name'], fg_color=bg_color, text_color=text_color,
                                      anchor="w", height=40, font=ctk.CTkFont(family="Microsoft YaHei", size=13),
                                      command=lambda: self.app.switch_task(task['task_id']))
        self.task_btn.grid(row=0, column=0, sticky="ew")

        self.menu_btn = ctk.CTkButton(self.button_frame, text="⋯", width=30, height=30, fg_color="transparent",
                                      text_color="#666666",
                                      font=ctk.CTkFont(family="Microsoft YaHei", size=16, weight="bold"),
                                      hover_color="#e0e0e0", command=self.show_menu)
        self.menu_btn.grid(row=0, column=1, padx=(0, 10), sticky="e")
        self.menu_btn.grid_remove()

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.task_btn.bind("<Enter>", self.on_enter)
        self.task_btn.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.menu_btn.grid()

    def on_leave(self, event):
        x = self.winfo_pointerx()
        y = self.winfo_pointery()
        menu_btn_x = self.menu_btn.winfo_rootx()
        menu_btn_y = self.menu_btn.winfo_rooty()
        menu_btn_width = self.menu_btn.winfo_width()
        menu_btn_height = self.menu_btn.winfo_height()
        if not (menu_btn_x <= x <= menu_btn_x + menu_btn_width and
                menu_btn_y <= y <= menu_btn_y + menu_btn_height):
            self.menu_btn.grid_remove()

    def show_menu(self):
        import tkinter as tk
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label=self.app.tr("menu_pin"), font=("Microsoft YaHei", 12), command=self.pin_task)
        menu.add_command(label=self.app.tr("menu_rename"), font=("Microsoft YaHei", 12), command=self.rename_task)
        menu.add_command(label=self.app.tr("menu_delete"), font=("Microsoft YaHei", 12), foreground="#ff4444", command=self.delete_task)
        menu.post(self.menu_btn.winfo_rootx(), self.menu_btn.winfo_rooty() + self.menu_btn.winfo_height())

    def pin_task(self):
        if self.app.task_manager.pin_task(self.task['task_id']):
            self.app.load_tasks_to_sidebar()
            if self.app.current_task_id == self.task['task_id']:
                self.app.switch_task(self.task['task_id'])

    def rename_task(self):
        dialog = ctk.CTkInputDialog(text=self.app.tr("rename_dialog_text"), title=self.app.tr("rename_dialog_title"),
                                    font=ctk.CTkFont(family="Microsoft YaHei", size=12))
        new_name = dialog.get_input()
        if new_name and new_name.strip():
            if self.app.task_manager.rename_task(self.task['task_id'], new_name.strip()):
                self.app.load_tasks_to_sidebar()
                if self.app.current_task_id == self.task['task_id']:
                    self.app.switch_task(self.task['task_id'])

    def delete_task(self):
        result = messagebox.askyesno(self.app.tr("delete_confirm_title"),
                                     self.app.tr("delete_task_confirm", name=self.task['name']),
                                     icon='warning')
        if result:
            self.app.cancel_task_workers(self.task['task_id'])
            self.app.task_manager.delete_task(self.task['task_id'])
            self.app.load_tasks_to_sidebar()
            if self.app.current_task_id == self.task['task_id']:
                if self.app.task_manager.tasks:
                    self.app.switch_task(self.app.task_manager.tasks[0]['task_id'])
                else:
                    self.app.create_new_task()
            self.app.update_stats_display()

    def update_selected_style(self):
        is_selected = (self.app.current_task_id == self.task['task_id'])
        bg_color = "#e6f2ff" if is_selected else "#f8f9fa"
        text_color = "#5c9eff" if is_selected else "#333333"
        self.task_btn.configure(fg_color=bg_color, text_color=text_color)


# ==================== 主窗口 ====================
class EasyDatasetApp(ctk.CTk):
    def __init__(self):
        _trace("EasyDatasetApp.__init__ start")
        super().__init__()
        self.lang = self._load_language_pref()
        _patch_tk_report_callback_exception(self)
        self.title("EasyDataset1.0")
        self.geometry("1100x700")
        self.minsize(1000, 600)

        self.configure(fg_color="#f8f9fa")

        self.task_manager = TaskManager(tasks_path, download_path)
        self.current_task_id = None
        self.is_searching = False
        self.task_items = []
        self.download_list_frame = None
        self.frame_extract_view_visible = False
        self.waiting_for_video_count = False
        self.pending_search_request = None

        self.task_download_frames = {}
        self.task_widgets = {}
        self.task_states = {}
        self.recommendation_flow_active = False
        self.recommendation_ui_paused = False
        # 二次/多轮推荐控制条归属的任务（与 current_task_id 解耦，避免切换任务后误操作或串台）
        self._recommendation_bar_task_id = None
        # 启动时分步加载：批越小、间隔越大 → 初始化越「瘦」，越不易触发 Tk/GDI 参数错误
        self._load_job_seq = 0
        self._STARTUP_DL_BATCH = 1
        self._STARTUP_DL_DELAY_MS = 100
        self._STARTUP_MSG_DELAY_MS = 32
        self._STARTUP_VIDEO_CARD_DELAY_MS = 28
        # 聊天区全部画完后再等这么久才开始右侧 fec 行（给主线程/GDI 喘息）
        self._STARTUP_FEC_DELAY_AFTER_CHAT_MS = 2200
        self._startup_resume_hooks_after_load = False
        # 冷启动待「继续进程」：右侧 fec 列表暂不增量添加，点击后再跑 _run_incremental_dl_batch
        self._fec_list_deferred_for_task = None
        self._incr_after_dl_batch_callback = None
        # 点击主界面「继续进程」后：先恢复搜索/帧提取，再延迟这么久开始补右侧封面（与任务并行、错峰加载）
        self._THUMB_RECOVERY_DELAY_MS = 6000

        self.setup_ui()
        self.load_tasks_to_sidebar()
        # 先让窗口进入事件循环并绘制，再在 idle 后恢复任务/聊天记录/右侧列表（避免老任务阻塞 __init__ 导致「像没打开」）
        try:
            self.update_idletasks()
            self.update()
        except Exception:
            pass
        self.after_idle(self._deferred_initial_task_load)
        _trace("EasyDatasetApp.__init__ end", "scheduled _deferred_initial_task_load")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # 定时清理 %TEMP% 下 Chrome/Selenium 遗留的 scoped_dir*（非任务缓存）
        self.after(120000, self._schedule_browser_scoped_temp_cleanup)

    def _load_language_pref(self):
        p = get_language_pref_path()
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                lang = str(d.get("lang") or "zh").strip().lower()
                if lang in I18N:
                    return lang
        except Exception:
            pass
        return "zh"

    def _suggest_english_task_name(self, user_request):
        toks = _tokenize_en(user_request)
        if not toks:
            return None
        keep = []
        stop = {"a", "an", "the", "to", "for", "of", "in", "on", "with", "and"}
        for t in toks:
            if t in stop:
                continue
            keep.append(t)
            if len(keep) >= 4:
                break
        if not keep:
            return None
        return " ".join(w.capitalize() for w in keep)

    def _save_language_pref(self):
        p = get_language_pref_path()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"lang": self.lang, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def tr(self, key, **kwargs):
        table = I18N.get(self.lang, I18N["zh"])
        text = table.get(key, I18N["zh"].get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def on_language_changed(self, selected_label):
        target = "zh"
        if str(selected_label).strip() == I18N["en"]["language_option_en"]:
            target = "en"
        if target == self.lang:
            return
        self.lang = target
        self._save_language_pref()
        self._refresh_language_ui_texts()

    def _refresh_language_ui_texts(self):
        try:
            self.create_task_btn.configure(text=self.tr("create_task"))
            self.task_list_title.configure(text=self.tr("task_list"))
            self.title_bar.configure(text=self.tr("app_title"))
            self.resume_continue_btn.configure(text=self.tr("resume_session"))
            self.lang_title_label.configure(text=self.tr("language_label"))
            for df in self.task_download_frames.values():
                df.title_label.configure(text=self.tr("download_list_title"))
                df.pause_btn.configure(text=self.tr("btn_pause_download"))
                df.resume_btn.configure(text=self.tr("btn_resume_download"))
                w = 140 if self.lang == "en" else 90
                df.pause_btn.configure(width=w)
                df.resume_btn.configure(width=w)
                df._prev_page_btn.configure(text=self.tr("page_prev"))
                df._next_page_btn.configure(text=self.tr("page_next"))
                try:
                    df.render_current_page()
                except Exception:
                    pass
            self._refresh_search_hint_placeholder()
        except Exception:
            pass

    def _schedule_browser_scoped_temp_cleanup(self):
        """后台清理系统临时目录中的 scoped_dir*（Chrome 临时用户数据目录），不触碰任务目录。"""
        def _run():
            try:
                cleanup_browser_scoped_temp_dirs(min_age_seconds=2700)
            except Exception as e:
                print(f"[浏览器系统缓存清理] 异常: {e}", flush=True)

        threading.Thread(target=_run, daemon=True).start()
        try:
            if self.winfo_exists():
                self.after(1800000, self._schedule_browser_scoped_temp_cleanup)
        except Exception:
            pass

    def _maybe_clear_stale_pause_after_idle_close(self):
        """
        正常退出时 on_closing 会把所有任务 task_paused=True；若 session_resume 未标记待继续（如已达目标空闲），
        重启后应解除「已达目标空闲」任务的暂停，避免无「继续进程」且仍暂停。
        """
        if read_session_resume_pending():
            return
        for t in list(self.task_manager.tasks):
            tid = t.get("task_id")
            if not tid or not t.get("task_paused"):
                continue
            if not task_is_fully_idle_at_goal(self.task_manager, t):
                continue
            try:
                self.task_manager.set_task_paused(tid, False)
            except Exception:
                pass

    def _pause_tasks_needing_resume_on_startup(self):
        """
        冷启动：凡需要「继续进程」的任务一律先暂停。
        避免 restore 右侧列表 / 聊天区时立刻 download_thumbnail（主线程卡）或
        因缓存仍为「提取中」而调用 borrow_extract_driver 创建帧提取浏览器。
        用户点击「继续进程」后再解除暂停并真正跑队列。
        仅在存在「上次正常退出时写入的 session_resume 待继续」时执行，避免无退出记录时误暂停且又无「继续进程」入口。
        """
        if not read_session_resume_pending():
            _trace("_pause_tasks_needing_resume_on_startup skip", "no session_resume pending")
            return
        _trace("_pause_tasks_needing_resume_on_startup enter")
        for t in self.task_manager.tasks:
            tid = t.get("task_id")
            if not tid:
                continue
            need_pause = False
            if t.get("ever_confirmed_to_download") and (
                task_has_incomplete_extraction(t) or t.get("recommendation_resume_pending")
            ):
                need_pause = True
            elif task_has_user_first_request_message(self.task_manager, tid):
                if not task_is_fully_idle_at_goal(self.task_manager, t):
                    need_pause = True
            if need_pause:
                _trace("startup pause task", str(tid))
                self.task_manager.set_task_paused(tid, True)

    def _deferred_initial_task_load(self):
        """首屏绘制后再加载当前任务（消息区、帧列表、缩略图等），避免阻塞构造函数。"""
        _trace("_deferred_initial_task_load enter")
        try:
            if not self.winfo_exists():
                _trace("_deferred_initial_task_load abort", "winfo_exists false")
                return
        except Exception:
            _trace("_deferred_initial_task_load abort", "winfo_exists exception")
            return
        # 上次退出未写入「待继续」时，解除因 on_closing 全任务暂停导致的「已达目标空闲」任务卡住
        try:
            self._maybe_clear_stale_pause_after_idle_close()
        except Exception:
            pass
        # 若用户在首帧已切换/新建任务，不再强制切回列表第一项
        if self.current_task_id is None:
            if not self.task_manager.tasks:
                _trace("_deferred_initial_task_load", "create_new_task (no tasks)")
                self.create_new_task()
            else:
                _trace("_deferred_initial_task_load", "cold start switch first task incremental")
                self._pause_tasks_needing_resume_on_startup()
                self._startup_resume_hooks_after_load = True
                self.switch_task(self.task_manager.tasks[0]['task_id'], incremental_load=True)
                return

        self.pending_resume_click = task_should_show_session_resume_banner(
            self.task_manager.get_task(self.current_task_id),
            read_session_resume_pending(),
            self.task_manager,
        )
        self._refresh_session_resume_row_visibility()

        if self.pending_resume_click:
            _trace("_deferred_initial_task_load", "apply resume session row")
            self._apply_resume_session_startup_after_load()
        _trace("_deferred_initial_task_load done", f"current_task_id={self.current_task_id}")

    def _apply_resume_session_startup_after_load(self):
        if hasattr(self, "resume_row") and not getattr(self, "frame_extract_view_visible", False):
            self.resume_row.grid()
        for tid in list(self.task_download_frames.keys()):
            try:
                self.task_download_frames[tid].set_startup_resume_pending_ui()
            except Exception:
                pass

    def _refresh_session_resume_row_visibility(self):
        if not hasattr(self, "resume_row"):
            return
        if getattr(self, "frame_extract_view_visible", False):
            self.resume_row.grid_remove()
            return
        t = self.task_manager.get_task(self.current_task_id) if self.current_task_id else None
        show = task_should_show_session_resume_banner(t, read_session_resume_pending(), self.task_manager)
        self.pending_resume_click = show
        if show:
            # 每个任务切换时恢复可点击样式，避免某个任务点过后其他任务也显示灰色不可点。
            try:
                self.resume_continue_btn.configure(text=self.tr("resume_session"), text_color="#165DFF", cursor="hand2")
            except Exception:
                pass
            self.resume_row.grid()
        else:
            self.resume_row.grid_remove()

    def _try_chat_control_command(self, txt):
        """
        聊天输入控制（不经过模型）：
        - 暂停搜索 / 继续搜索、暂停推荐 / 继续推荐：与首轮搜索、二次推荐共用同一协作 Event。
        - 暂停下载 / 继续下载：帧提取列表的暂停下载 / 继续下载。
        返回 True 表示本条输入已消费。
        """
        tid = self.current_task_id
        if not tid:
            return False
        s = (txt or "").strip()
        if not s:
            return False
        compact = re.sub(r"\s+", "", s)
        compact = re.sub(r"[。，、！？\.!,?]+$", "", compact)
        compact_en = re.sub(r"\s+", " ", s.lower()).strip().rstrip(".!,?")

        def _reply(zh):
            self.add_msg({"type": "text", "content": zh, "is_user": False})

        def _push_user_cmd():
            try:
                msg = {"type": "text", "content": s, "is_user": True}
                st = self.get_task_state(tid)
                msgs = list(st.get("current_messages") or [])
                msgs.append(msg)
                st["current_messages"] = msgs
                self.task_manager.save_task_messages(tid, msgs)
                if tid == getattr(self, "current_task_id", None):
                    self.current_messages = msgs
                msg_area = self.get_current_msg_area()
                if msg_area:
                    MessageBubble(msg_area, msg, self.task_manager, tid, True, app=self).pack(fill="x", pady=2)
                    msg_area._parent_canvas.yview_moveto(1)
            except Exception:
                pass

        if compact in ("暂停搜索", "暂停推荐") or compact_en in ("pause search", "pause recommendation"):
            _push_user_cmd()
            TaskThreadManager.pause_recommendation(tid)
            _reply(
                "Paused search and recommendation. Type \"resume search\" or \"resume recommendation\" to continue."
                if self.lang == "en"
                else "已暂停搜索与推荐进程。输入「继续搜索」或「继续推荐」可恢复。"
            )
            return True
        if compact in ("继续搜索", "继续推荐") or compact_en in ("resume search", "resume recommendation"):
            _push_user_cmd()
            TaskThreadManager.resume_recommendation(tid)
            _reply("Resumed search and recommendation." if self.lang == "en" else "已继续搜索与推荐进程。")
            return True
        if compact == "暂停下载" or compact_en == "pause download":
            _push_user_cmd()
            df = self.task_download_frames.get(tid)
            if df is not None:
                df.pause_all_downloads()
            else:
                self.task_manager.set_task_paused(tid, True)
            _reply("Download paused (frame extraction list)." if self.lang == "en" else "已暂停下载（帧提取列表）。")
            return True
        if compact == "继续下载" or compact_en == "resume download":
            _push_user_cmd()
            self.task_manager.set_task_paused(tid, False)
            df = self.task_download_frames.get(tid)
            if df is not None:
                df.resume_all_downloads()
            _reply("Download resumed (frame extraction list)." if self.lang == "en" else "已继续下载（帧提取列表）。")
            return True
        return False

    def _resume_bootstrap_from_confirmed_video_list_if_needed(self, tid):
        """
        兼容「点了已确认后立刻关闭」场景：
        若任务已有已确认视频列表，但尚未真正进入 ever_confirmed_to_download 流程，
        点击继续进程时自动补跑：勾选视频入下载列表 + 继续搜索推荐。
        """
        task = self.task_manager.get_task(tid) if tid else None
        if not task:
            return False
        if task.get("ever_confirmed_to_download"):
            return False
        msgs = []
        try:
            msgs = self.task_manager.load_task_messages(tid)
        except Exception:
            msgs = []
        confirmed_msg = None
        for m in reversed(msgs or []):
            if m.get("type") == "video_list" and m.get("selection_confirmed"):
                confirmed_msg = m
                break
        if not confirmed_msg:
            return False

        videos = list(confirmed_msg.get("videos") or [])
        selected_videos = [v for v in videos if (v or {}).get("selected", True)]
        deselected_videos = [v for v in videos if not (v or {}).get("selected", True)]
        if not selected_videos:
            return False

        self.task_manager.update_task_info(tid, {"ever_confirmed_to_download": True})
        self.task_manager.set_task_paused(tid, False)
        df = self._ensure_download_list_frame(tid)
        for v in selected_videos:
            try:
                df.add_video_to_queue(v)
            except Exception:
                pass

        st = self.get_task_state(tid)
        kw_map = dict(st.get("keyword_video_map") or {})
        if not kw_map:
            kw_map = dict((task.get("ui_state") or {}).get("keyword_video_map") or {})
        seen = set(st.get("seen_video_ids") or set())
        if not seen:
            seen = set((task.get("ui_state") or {}).get("seen_video_ids") or [])
        user_request = st.get("current_user_request") or (task.get("ui_state") or {}).get("current_user_request") or ""
        prefs = {
            "cached_preferences": task.get("cached_preferences") or {},
            "selection_analysis": (task.get("preferences") or ""),
        }
        try:
            target_count = int(task.get("target_video_count")) if task.get("target_video_count") is not None else 20
        except (TypeError, ValueError):
            target_count = 20
        current_count = self._current_download_total_for_task(tid)
        if kw_map and target_count > current_count:
            self.start_secondary_search(
                tid,
                user_request,
                prefs,
                deselected_videos,
                target_count,
                current_count,
                keyword_video_map=kw_map,
                seen_video_ids=seen,
            )
        return True

    def cancel_task_workers(self, task_id):
        TaskThreadManager.cancel_task(task_id)
        if task_id in self.task_download_frames:
            self.task_download_frames[task_id].stop_all_threads_for_delete_task()

    def on_recommendation_flow_toggle_click(self):
        """底部推荐控制条已移除；保留空实现避免旧绑定报错。"""
        pass

    def _show_recommendation_control_bar(self, task_id):
        """已废弃：暂停/继续搜索与推荐改为聊天输入指令，不再显示底部「暂停进程」条。"""
        self._recommendation_bar_task_id = None
        self.recommendation_ui_paused = False

    def _hide_recommendation_control_bar(self, task_id=None):
        """兼容旧调用：不再展示推荐控制条。"""
        self._recommendation_bar_task_id = None

    def _sync_recommendation_bar_visibility(self):
        """兼容旧调用：底部推荐控制条已移除。"""
        pass

    def _persist_recommendation_payload(self, task_id, payload):
        try:
            self.task_manager.update_task_info(task_id, {
                "recommendation_resume_payload": payload,
                "recommendation_resume_pending": True
            })
        except Exception:
            pass

    def _clear_recommendation_resume_state(self, task_id):
        try:
            self.task_manager.update_task_info(task_id, {
                "recommendation_resume_pending": False,
                "recommendation_resume_payload": None
            })
        except Exception:
            pass

    def _finish_recommendation_flow(self, task_id):
        """
        推荐流程结束（含因达到目标而停）：保留 recommendation_resume_payload，记录停在哪（关键词/已见集合等），
        仅清除「会话中途暂停」用的 pending 标记，便于用户调高目标后接着搜，而无需重开一轮「二次推荐」。
        """
        try:
            snap = self._build_recommendation_resume_payload_dict(task_id)
            if snap and isinstance(snap, dict):
                if snap.get("keyword_video_map") is None:
                    snap["keyword_video_map"] = {}
                self.task_manager.update_task_info(
                    task_id,
                    {
                        "recommendation_resume_payload": snap,
                        "recommendation_resume_pending": False,
                    },
                )
            else:
                self.task_manager.update_task_info(task_id, {"recommendation_resume_pending": False})
        except Exception as e:
            print(f"[推荐] 写入继续搜索缓存失败: {e}", flush=True)
            try:
                self.task_manager.update_task_info(task_id, {"recommendation_resume_pending": False})
            except Exception:
                pass
        self.recommendation_flow_active = False
        self.after(0, self._refresh_search_hint_placeholder)
        self._hide_recommendation_control_bar(task_id)
        write_session_resume_pending(compute_any_task_needs_session_resume(self.task_manager))
        self.after(0, self._refresh_session_resume_row_visibility)

    def _current_download_total_for_task(self, task_id):
        """任务当前下载总量（优先 UI 列表 _video_order，其次 frame_extract_cache 非删除条目）。"""
        df = self.task_download_frames.get(task_id)
        if df is not None:
            try:
                return len(getattr(df, "_video_order", []) or [])
            except Exception:
                pass
        t = self.task_manager.get_task(task_id)
        fec = (t or {}).get("frame_extract_cache", {}) or {}
        n = 0
        for ent in fec.values():
            if not ent:
                continue
            if ent.get("status_text") == "已删除":
                continue
            if ent.get("video"):
                n += 1
        return n

    def _collect_all_seen_video_ids_for_task(self, task_id):
        """任务内所有曾出现/推荐过的视频 ID（含首轮取消勾选、判别不通过、已删除等），用于全局去重。"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return set()
        s = set()
        for vid in task.get("seen_video_ids") or []:
            if vid:
                s.add(vid)
        ui = task.get("ui_state") or {}
        for vid in ui.get("seen_video_ids") or []:
            if vid:
                s.add(vid)
        fec = task.get("frame_extract_cache") or {}
        for k in fec.keys():
            if k:
                s.add(k)
        for vid in task.get("selected_videos") or []:
            if vid:
                s.add(vid)
        for vid in task.get("deselected_videos") or []:
            if vid:
                s.add(vid)
        for v in task.get("videos") or []:
            if v and v.get("video_id"):
                s.add(v["video_id"])
        msgs = []
        try:
            st = self.get_task_state(task_id)
            msgs = list(st.get("current_messages") or [])
        except Exception:
            msgs = []
        if not msgs:
            try:
                msgs = self.task_manager.load_task_messages(task_id)
            except Exception:
                msgs = []
        try:
            if task_id == getattr(self, "current_task_id", None) and getattr(self, "current_messages", None):
                msgs = list(self.current_messages or msgs)
        except Exception:
            pass
        for msg in msgs or []:
            if msg.get("type") != "video_list":
                continue
            for v in msg.get("videos") or []:
                if v and v.get("video_id"):
                    s.add(v["video_id"])
        try:
            df = self.task_download_frames.get(task_id)
            if df:
                for vid in getattr(df, "_video_order", []) or []:
                    if vid:
                        s.add(vid)
        except Exception:
            pass
        return s

    def _register_seen_video(self, task_id, video_id):
        if not video_id:
            return
        self.task_manager.merge_seen_video_ids_batch(task_id, [video_id])
        if task_id == getattr(self, "current_task_id", None):
            if not hasattr(self, "seen_video_ids") or self.seen_video_ids is None:
                self.seen_video_ids = set()
            self.seen_video_ids.add(video_id)
        try:
            st = self.get_task_state(task_id)
            sid = st.get("seen_video_ids")
            if isinstance(sid, list):
                st["seen_video_ids"] = set(sid)
            elif sid is None:
                st["seen_video_ids"] = set()
            st["seen_video_ids"].add(video_id)
        except Exception:
            pass

    def _merge_seen_into_runtime_and_disk(self, task_id):
        """启动推荐/搜索前：合并磁盘+内存中的已见集合到本次 run 的 seen_video_ids。"""
        merged = self._collect_all_seen_video_ids_for_task(task_id)
        try:
            st = self.get_task_state(task_id)
            if isinstance(st.get("seen_video_ids"), set):
                merged |= st["seen_video_ids"]
            elif st.get("seen_video_ids"):
                merged |= set(st["seen_video_ids"])
        except Exception:
            pass
        if task_id == getattr(self, "current_task_id", None) and getattr(self, "seen_video_ids", None):
            merged |= set(self.seen_video_ids or set())
        self.task_manager.merge_seen_video_ids_batch(task_id, list(merged))
        try:
            st = self.get_task_state(task_id)
            st["seen_video_ids"] = set(merged)
        except Exception:
            pass
        try:
            self.persist_task_ui_state(task_id)
        except Exception:
            pass
        return merged

    def _expand_keyword_map_with_qwen(self, task_id, user_request, keyword_video_map):
        """
        在旧关键词搜尽或无效时，用 Qwen 生成一批新的英文关键词并并入 keyword_video_map（空列表占位）。
        返回新加入的关键词列表（可能为空）。
        """
        ur = user_request or ""
        try:
            task = self.task_manager.get_task(task_id)
            if not ur and task:
                ur = (
                    (self.get_task_state(task_id).get("current_user_request"))
                    or (task.get("ui_state") or {}).get("current_user_request")
                    or task.get("name")
                    or ""
                )
        except Exception:
            pass
        existing = set(keyword_video_map.keys())
        added = []
        try:
            ext = _extract_keywords_and_preferences(ur)
            kws = ext.get("keywords") or []
        except Exception:
            kws = []
        if len(kws) < 5:
            try:
                kws = (kws or []) + list(generate_search_keywords(ur))
            except Exception:
                pass
        for kw in kws:
            ks = str(kw).strip()
            if ks and ks not in existing:
                keyword_video_map[ks] = []
                existing.add(ks)
                added.append(ks)
            if len(added) >= 10:
                break
        return added

    def _expand_keyword_map_with_qwen_fresh(self, task_id, user_request, keyword_video_map):
        """
        侧边栏/旧词路径搜尽后专用：让模型在明确避开已有搜索词的前提下生成新词，并入 keyword_video_map。
        """
        ur = user_request or ""
        try:
            task = self.task_manager.get_task(task_id)
            if not ur and task:
                ur = (
                    (self.get_task_state(task_id).get("current_user_request"))
                    or (task.get("ui_state") or {}).get("current_user_request")
                    or task.get("name")
                    or ""
                )
        except Exception:
            pass
        existing = set(keyword_video_map.keys())
        added = []
        try:
            fresh = generate_search_keywords_avoid(ur, existing)
        except Exception:
            fresh = []
        for kw in fresh:
            ks = str(kw).strip()
            if ks and ks not in existing:
                keyword_video_map[ks] = []
                existing.add(ks)
                added.append(ks)
            if len(added) >= 12:
                break
        if len(added) < 6:
            try:
                ext = _extract_keywords_and_preferences(ur)
                for kw in ext.get("keywords") or []:
                    ks = str(kw).strip()
                    if ks and ks not in existing:
                        keyword_video_map[ks] = []
                        existing.add(ks)
                        added.append(ks)
                    if len(added) >= 12:
                        break
            except Exception:
                pass
        return added

    def _build_recommendation_resume_payload_dict(self, task_id):
        """
        构造「继续搜索推荐」用的快照：关键词→视频列表、已见 id、需求与偏好等。
        在推荐结束或需要续跑时写入 recommendation_resume_payload。
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            return {}
        st = self.get_task_state(task_id)
        kw_map = dict(st.get("keyword_video_map") or {})
        if task_id == getattr(self, "current_task_id", None):
            kw_map = dict(getattr(self, "keyword_video_map", {}) or kw_map)
        if not kw_map:
            kw_map = dict((task.get("ui_state") or {}).get("keyword_video_map") or {})
        merged_seen = set(self._collect_all_seen_video_ids_for_task(task_id))
        try:
            sx = st.get("seen_video_ids")
            if isinstance(sx, set):
                merged_seen |= sx
            elif sx:
                merged_seen |= set(sx)
        except Exception:
            pass
        if task_id == getattr(self, "current_task_id", None) and getattr(self, "seen_video_ids", None):
            merged_seen |= set(self.seen_video_ids or set())
        seen = list(merged_seen)
        ur = st.get("current_user_request") or (task.get("ui_state") or {}).get("current_user_request")
        if task_id == getattr(self, "current_task_id", None) and getattr(self, "current_user_request", None):
            ur = self.current_user_request
        if not ur:
            ur = task.get("name") or ""
        try:
            tc = int(task.get("target_video_count")) if task.get("target_video_count") is not None else 20
        except (TypeError, ValueError):
            tc = 20
        cur = self._current_download_total_for_task(task_id)
        combined_preferences = {
            "cached_preferences": task.get("cached_preferences") or {},
            "selection_analysis": (task.get("preferences") or ""),
        }
        deselected = self._deselected_video_dicts_for_task(task_id)
        prev_pl = task.get("recommendation_resume_payload") or {}
        return {
            "user_request": ur,
            "preferences": combined_preferences,
            "deselected_videos": deselected,
            "target_count": tc,
            "current_count": cur,
            "keyword_video_map": kw_map,
            "seen_video_ids": seen,
            "secondary_kw_order": prev_pl.get("secondary_kw_order"),
            "secondary_kw_cursor": prev_pl.get("secondary_kw_cursor", 0),
        }

    def _resume_recommendation_worker(self, tid, payload):
        try:
            try:
                self.task_manager.set_task_paused(tid, False)
            except Exception:
                pass
            task = self.task_manager.get_task(tid)
            # 目标数以 info.json 为准（用户可能在对话里改了 target_video_count）
            target = (task or {}).get("target_video_count")
            try:
                target = int(target) if target is not None else None
            except (TypeError, ValueError):
                target = None
            if target is None:
                try:
                    target = int(payload.get("target_count")) if payload.get("target_count") is not None else None
                except (TypeError, ValueError):
                    target = None
            current_total = self._current_download_total_for_task(tid)
            if target is not None and target > 0 and current_total >= target:
                print(f"[推荐停止] 任务 {tid} 当前 {current_total} 已达到/超过目标 {target}，跳过恢复推荐")
                self._finish_recommendation_flow(tid)
                return
            TaskThreadManager.ensure_recommendation_event(tid)
            TaskThreadManager.resume_recommendation(tid)
            user_request = payload.get("user_request")
            preferences = payload.get("preferences") or {}
            deselected = payload.get("deselected_videos") or []
            target_count = target if target is not None else (payload.get("target_count") or 20)
            dl = self.task_download_frames.get(tid)
            current_count = len(getattr(dl, "_video_order", []) or []) if dl else payload.get("current_count", 0)
            kw_map = dict(payload.get("keyword_video_map") or {})
            seen = set(payload.get("seen_video_ids") or [])
            seen |= self._merge_seen_into_runtime_and_disk(tid)
            self.start_secondary_search(
                tid, user_request, preferences, deselected, target_count, current_count,
                keyword_video_map=kw_map, seen_video_ids=seen)
        except Exception as e:
            print(f"恢复推荐失败: {e}")

    def _refresh_chat_video_thumbnails_for_task(self, task_id, gap_ms=400):
        """遍历消息区，错峰启动 VideoCardFrame 的缩略图加载（延迟很久后调用，避免与队列/布局抢主线程）。"""
        if not task_id or task_id not in self.task_widgets:
            return
        msg_area = self.task_widgets[task_id].get("msg_area")
        if not msg_area:
            return
        cards = []
        from collections import deque
        dq = deque([msg_area])
        seen = set()
        while dq:
            w = dq.popleft()
            wid = id(w)
            if wid in seen:
                continue
            seen.add(wid)
            try:
                for c in w.winfo_children():
                    if isinstance(c, VideoCardFrame):
                        cards.append(c)
                    dq.append(c)
            except Exception:
                pass
        gap = max(80, int(gap_ms))
        for i, card in enumerate(cards):
            delay = i * gap
            self.after(delay, lambda cc=card: cc.schedule_thumbnail_after_resume())

    def _begin_deferred_thumbnail_recovery(self, task_id):
        """用户点击「继续进程」且搜索/帧提取已恢复后：解除右侧列表封面抑制，并错峰补图（与任务并行、慢慢加载）。"""
        if not task_id or not self.winfo_exists():
            return
        if self.current_task_id != task_id:
            return
        df = self.task_download_frames.get(task_id)
        if df:
            try:
                df._suppress_covers_until_resume = False
                df._RIGHT_LIST_THUMB_STAGGER_MS = max(8000, int(getattr(df, "_RIGHT_LIST_THUMB_STAGGER_MS", 180)))
                df.schedule_missing_thumbnails_in_order(slow=True)
            except Exception:
                pass
        try:
            self._refresh_chat_video_thumbnails_for_task(task_id, gap_ms=600)
        except Exception:
            pass

    def _schedule_later_thumbnail_recovery(self, tid):
        delay = int(getattr(self, "_THUMB_RECOVERY_DELAY_MS", 120000))

        def _go():
            try:
                if self.winfo_exists() and self.current_task_id == tid:
                    self._begin_deferred_thumbnail_recovery(tid)
            except Exception:
                pass

        self.after(delay, _go)

    def _finish_resume_after_fec_list(self, tid):
        _resume_flow_dbg("_finish_resume_after_fec_list.enter", f"tid={tid}")
        df = self.task_download_frames.get(tid)
        if df:
            try:
                df.resume_from_first_incomplete_after_relaunch(schedule_thumbnails=False)
            except Exception:
                pass
        _resume_flow_dbg("_finish_resume_after_fec_list.after_resume_incomplete", f"tid={tid}")
        self._schedule_later_thumbnail_recovery(tid)
        _resume_flow_dbg("_finish_resume_after_fec_list.done", f"tid={tid}")

    def _resume_session_after_deferred_fec(self, tid):
        """会话继续：轻量恢复 fec 顺序与队列（不预建宫格），再 resume；封面延后由 _schedule_later_thumbnail_recovery 处理。"""
        _resume_flow_dbg("_resume_session_after_deferred_fec.enter", f"tid={tid}")
        self.reload_download_list_for_task(tid)
        _resume_flow_dbg("_resume_session_after_deferred_fec.after_reload", f"tid={tid}")
        self._finish_resume_after_fec_list(tid)
        _resume_flow_dbg("_resume_session_after_deferred_fec.done", f"tid={tid}")

    def _task_needs_search_browser_on_resume(self, tid):
        """恢复二次推荐等搜索流程时需要搜索专用浏览器。"""
        task = self.task_manager.get_task(tid) if tid else None
        if not task:
            return False
        return bool(task.get("recommendation_resume_pending") and task.get("recommendation_resume_payload"))

    def _task_needs_extract_browser_on_resume(self, tid):
        """恢复帧提取/下载队列时需要帧提取专用浏览器。"""
        task = self.task_manager.get_task(tid) if tid else None
        if not task or not task.get("ever_confirmed_to_download"):
            return False
        return task_has_incomplete_extraction(task)

    def _prewarm_resume_browsers_worker(self, tid):
        """在后台线程中分步创建浏览器，避免阻塞 Tk 主线程；控制台打印便于定位卡点。"""
        need_search = self._task_needs_search_browser_on_resume(tid)
        need_extract = self._task_needs_extract_browser_on_resume(tid)
        print(f"[继续进程] 待预创建: 搜索浏览器={need_search}, 下载/帧提取浏览器={need_extract}")
        try:
            if need_search:
                print("[继续进程] 正在创建搜索浏览器...")
                TaskBrowserManager(tid).get_search_driver()
                print("[继续进程] 搜索浏览器创建成功")
            else:
                print("[继续进程] 无待恢复搜索任务，跳过搜索浏览器")
            if need_extract:
                print(
                    f"[继续进程] 正在预创建全部帧提取/下载浏览器（共 {MAX_PARALLEL_EXTRACT_BROWSERS} 台，与并行提取上限一致）..."
                )
                mgr = TaskBrowserManager(tid)
                borrowed = []
                try:
                    for i in range(MAX_PARALLEL_EXTRACT_BROWSERS):
                        d = mgr.borrow_extract_driver()
                        if not d:
                            print(f"[继续进程] 第 {i + 1} 台 borrow 返回空，停止预创建")
                            break
                        borrowed.append(d)
                finally:
                    for d in borrowed:
                        mgr.release_extract_driver(d)
                print(
                    f"[继续进程] 帧提取/下载浏览器预创建完成：{len(borrowed)}/{MAX_PARALLEL_EXTRACT_BROWSERS} 台（已归还空闲池，可供并行提取）"
                )
            else:
                print("[继续进程] 无待恢复下载/帧提取任务，跳过帧提取浏览器")
        except Exception as e:
            print(f"[继续进程] 浏览器预创建异常: {e}")
            traceback.print_exc()

    def _resume_session_continue(self, tid, deferred, rec_payload):
        """浏览器预创建完成后在主线程执行：恢复列表与队列，再启动推荐续跑线程。"""
        try:
            if getattr(self, "_resume_session_continue_done", False):
                return
            self._resume_session_continue_done = True
            _resume_flow_dbg_reset()
            print("[继续进程] 主线程开始执行 _resume_session_continue（reload / 恢复队列）")
            _resume_flow_dbg(
                "_resume_session_continue.enter",
                f"tid={tid} deferred={deferred} rec_payload={'yes' if rec_payload is not None else 'no'}",
            )
            if deferred and tid:
                self._resume_session_after_deferred_fec(tid)
            else:
                _resume_flow_dbg("_resume_session_continue.branch_non_deferred", f"tid={tid}")
                if tid and tid in self.task_download_frames:
                    try:
                        self.task_download_frames[tid].resume_from_first_incomplete_after_relaunch(schedule_thumbnails=False)
                    except Exception:
                        pass
                    _resume_flow_dbg("_resume_session_continue.after_resume_incomplete", f"tid={tid}")
                    self._schedule_later_thumbnail_recovery(tid)
                    _resume_flow_dbg("_resume_session_continue.after_thumb_schedule", f"tid={tid}")
            if tid and rec_payload is not None:
                _resume_flow_dbg("_resume_session_continue.spawn_recommendation_thread", f"tid={tid}")
                threading.Thread(
                    target=lambda: self._resume_recommendation_worker(tid, rec_payload), daemon=True
                ).start()
            # 无 recommendation_resume_payload 但该任务已确认首轮列表且未真正入队时，补跑一次恢复。
            if tid and rec_payload is None:
                try:
                    booted = self._resume_bootstrap_from_confirmed_video_list_if_needed(tid)
                    if booted:
                        _resume_flow_dbg("_resume_session_continue.bootstrap_from_confirmed_list", f"tid={tid}")
                except Exception:
                    pass
            # 尚无右侧下载列表时 resume_from_first_incomplete 不会执行，需显式解除暂停；
            # 若仅有首轮对话、尚未出现 video_list，则补跑首轮搜索。
            if tid and self.task_manager.is_task_paused(tid) and tid not in self.task_download_frames:
                try:
                    self.task_manager.set_task_paused(tid, False)
                except Exception:
                    pass
                try:
                    self._maybe_restart_first_search_after_resume_if_needed(tid)
                except Exception:
                    pass
            _resume_flow_dbg("_resume_session_continue.exit_ok", f"tid={tid}")
        finally:
            self._resume_session_in_progress = False

    def _maybe_restart_first_search_after_resume_if_needed(self, tid):
        """冷启动点击「继续进程」：若用户已给过需求与数量但尚未出现首轮 video_list，则重新跑首轮搜索。"""
        task = self.task_manager.get_task(tid) if tid else None
        if not task or task.get("ever_confirmed_to_download"):
            return
        msgs = []
        try:
            msgs = self.task_manager.load_task_messages(tid)
        except Exception:
            msgs = []
        if any(m.get("type") == "video_list" for m in (msgs or [])):
            return
        state = self.get_task_state(tid)
        ur = state.get("current_user_request")
        if not ur:
            ur = (task.get("ui_state") or {}).get("current_user_request")
        if not ur:
            for m in msgs or []:
                if m.get("is_user") and m.get("type") == "text":
                    ur = m.get("content")
                    break
        if not ur:
            return
        if getattr(self, "is_searching", False):
            return
        threading.Thread(target=self._run_post_count_flow, args=(ur, tid), daemon=True).start()

    def _schedule_resume_session_on_main_after_prewarm(self, tid, deferred, rec_payload):
        """预热线程结束后调度主线程恢复。非主线程调用 self.after(0) 在多数 Windows 环境下可用；
        若不可用，由主线程对 Queue 的短间隔轮询兜底（见 on_resume_session_clicked）。"""
        def _on_main():
            try:
                self._resume_session_continue(tid, deferred, rec_payload)
            except Exception as e:
                print(f"[继续进程] _resume_session_continue 异常: {e}")
                traceback.print_exc()

        try:
            self.after(0, _on_main)
        except Exception as e:
            print(f"[继续进程] after(0) 调度失败: {e}")
            traceback.print_exc()

    def on_resume_session_clicked(self):
        if getattr(self, "_resume_session_in_progress", False):
            print("[继续进程] 已在进行中，忽略重复点击")
            return
        self._resume_session_in_progress = True
        tid = self.current_task_id
        self._resume_session_continue_done = False
        write_session_resume_pending(compute_any_task_needs_session_resume(self.task_manager))
        self.pending_resume_click = False
        if hasattr(self, "resume_continue_btn"):
            self.resume_continue_btn.configure(text=self.tr("resume_session"), text_color="#999999", cursor="")
        if hasattr(self, "resume_row"):
            self.resume_row.grid_remove()

        deferred = bool(tid and getattr(self, "_fec_list_deferred_for_task", None) == tid)
        if deferred:
            self._fec_list_deferred_for_task = None

        rec_payload = None
        if tid:
            task = self.task_manager.get_task(tid)
            if task and task.get("recommendation_resume_pending") and task.get("recommendation_resume_payload"):
                rec_payload = dict(task.get("recommendation_resume_payload") or {})

        delivery = Queue()

        def _bg():
            try:
                self._prewarm_resume_browsers_worker(tid)
                delivery.put((tid, deferred, rec_payload))
                print("[继续进程] 预创建结束，向主线程投递恢复（after 0 + Queue 兜底）")
                self._schedule_resume_session_on_main_after_prewarm(tid, deferred, rec_payload)
            except Exception as e:
                print(f"[继续进程] 预热线程异常: {e}")
                traceback.print_exc()
                try:
                    self.after(0, lambda: setattr(self, "_resume_session_in_progress", False))
                except Exception:
                    self._resume_session_in_progress = False

        def _poll_delivery_from_main(_attempt=0):
            """主线程轮询：在大量 after 积压时，比单次 after(30)+get_nowait 更早取到预热线程的 put。"""
            if getattr(self, "_resume_session_continue_done", False):
                return
            try:
                tid2, deferred2, rec_payload2 = delivery.get_nowait()
            except Empty:
                if _attempt < 12000:
                    self.after(5, lambda: _poll_delivery_from_main(_attempt + 1))
                else:
                    print("[继续进程] 警告：Queue 兜底轮询超时（60s），请检查预热线程是否未 put")
                    self._resume_session_in_progress = False
                return
            self._resume_session_continue(tid2, deferred2, rec_payload2)

        threading.Thread(target=_bg, daemon=True).start()
        self.after(0, lambda: _poll_delivery_from_main(0))

    def _run_post_count_flow(self, txt, task_id):
        """task_id 在发起搜索时固定，避免生成关键词/首轮搜索过程中切换任务导致串台。"""
        tid = task_id
        if not tid:
            return
        self.after(0, lambda x=tid: self.update_status_message("🤖 正在生成搜索关键词...", task_id=x))
        try:
            if TaskThreadManager.is_task_cancelled(tid):
                return
            info = _extract_keywords_and_preferences(txt)
            kws = info.get("keywords", []) or []
            prefs = info.get("preferences") or {}
            try:
                self.task_manager.update_task_info(tid, {"cached_preferences": prefs})
            except Exception:
                pass
            state = self.get_task_state(tid)
            state["generated_keywords"] = kws

            task = self.task_manager.get_task(tid)
            if task and not task.get("auto_named"):
                if str(task.get("name", "")).startswith("任务 ") or str(task.get("name", "")).startswith("Task "):
                    suggested = info.get("task_name")
                    if self.lang == "en":
                        suggested = self._suggest_english_task_name(txt) or suggested
                    if suggested:
                        unique = self.task_manager.generate_unique_task_name(
                            suggested, exclude_task_id=tid
                        )
                        if self.task_manager.rename_task(tid, unique):
                            self.task_manager.update_task_info(tid, {"auto_named": True})
                            self.after(0, self.load_tasks_to_sidebar)
                            self.after(0, self.update_task_items_style)

            self.after(0, lambda x=tid, n=len(kws): self.update_status_message(
                f"✅ 已生成 {n} 个搜索关键词", task_id=x))
        except Exception as e:
            print(f"生成关键词错误: {e}")
            self.after(0, lambda x=tid, err=str(e): self.update_status_message(
                f"❌ 生成关键词出错: {err}", task_id=x))

        def _start():
            if TaskThreadManager.is_task_cancelled(tid):
                return
            self.start_search(txt, tid)

        self.after(120, _start)

    def on_closing(self):
        # 先暂停运行中的提取/推荐，再落盘；不调用浏览器 quit，进程可保留。
        for t in self.task_manager.tasks:
            tid = t["task_id"]
            try:
                self.task_manager.set_task_paused(tid, True)
            except Exception:
                pass
            if tid in self.task_download_frames:
                try:
                    self.task_download_frames[tid].pause_all_downloads()
                except Exception:
                    pass
        try:
            self.update_idletasks()
        except Exception:
            pass
        try:
            self.persist_all_running_task_cache_before_exit()
        except Exception as e:
            print(f"关闭前写入缓存失败: {e}")
        TaskThreadManager.shutdown_all()
        try:
            self.task_manager.refresh_all_info_baks()
        except Exception as e:
            print(f"更新 info.json.bak 备份失败: {e}")
        write_session_resume_pending(compute_any_task_needs_session_resume(self.task_manager))

        def _exit_browser_temp_cleanup():
            try:
                cleanup_browser_scoped_temp_dirs(
                    min_age_seconds=600, log_prefix="[退出时·浏览器系统缓存清理]"
                )
            except Exception:
                pass

        threading.Thread(target=_exit_browser_temp_cleanup, daemon=True).start()
        # 关闭时需要删空 system/temp：先关闭所有 WebDriver 释放目录占用。
        try:
            TaskBrowserManager.close_all_drivers()
        except Exception:
            pass
        try:
            if os.path.isdir(browser_temp_root):
                ok = remove_dir_with_retries(browser_temp_root, attempts=16, wait_sec=0.25)
                if not ok:
                    print(f"[退出清理] 未能完全删除目录: {browser_temp_root}", flush=True)
        except Exception:
            pass
        self.destroy()

    def get_task_state(self, task_id):
        if task_id not in self.task_states:
            task = self.task_manager.get_task(task_id) if hasattr(self, "task_manager") else None
            ui_state = task.get("ui_state") if task else None
            self.task_states[task_id] = {
                'current_messages': [],
                'current_user_request': (ui_state.get("current_user_request") if ui_state else None),
                'keyword_video_map': dict(ui_state.get("keyword_video_map") or {}) if ui_state else {},
                'seen_video_ids': set(ui_state.get("seen_video_ids", [])) if ui_state else set(),
                'status_frame': None,
                # 与 info.json 对齐，避免重启后会话状态只有内存默认值
                'target_video_count': (task.get("target_video_count") if task else None),
                'generated_keywords': (ui_state.get("generated_keywords", []) if ui_state else []),
                # 任务级输入/流程状态（避免跨任务串台）
                'is_searching': False,
                'waiting_for_video_count': False,
                'pending_search_request': None,
                'input_draft': "",
            }
        return self.task_states[task_id]

    def _set_task_searching(self, task_id, searching):
        st = self.get_task_state(task_id)
        st["is_searching"] = bool(searching)
        any_searching = any((s.get("is_searching") for s in self.task_states.values()))
        self.is_searching = bool(any_searching)

    def _snapshot_outgoing_task_ui_state(self, outgoing_task_id):
        """切换任务前把当前界面上的会话快照写入该任务的 task_states（多任务关闭时与落盘一致）。"""
        if not outgoing_task_id:
            return
        try:
            st = self.get_task_state(outgoing_task_id)
            st["current_messages"] = list(self.current_messages)
            st["keyword_video_map"] = dict(self.keyword_video_map or {})
            st["seen_video_ids"] = set(self.seen_video_ids or set())
            if self.current_user_request is not None:
                st["current_user_request"] = self.current_user_request
        except Exception:
            pass

    def persist_task_ui_state(self, task_id):
        """把推荐/搜索相关的易丢状态写入 info.json，避免直接退出丢失进度"""
        if not task_id:
            return
        state = self.task_states.get(task_id)
        if not state:
            return
        ui_state = {
            "current_user_request": state.get("current_user_request"),
            "generated_keywords": list(state.get("generated_keywords") or []),
            "seen_video_ids": list(state.get("seen_video_ids") or []),
            "keyword_video_map": state.get("keyword_video_map") or {},
            "updated_at": datetime.now().isoformat()
        }
        self.task_manager.update_task_info(task_id, {"ui_state": ui_state})

    def persist_all_task_ui_states(self):
        for tid in list(self.task_states.keys()):
            self.persist_task_ui_state(tid)

    def _flush_recommendation_resume_on_exit(self):
        """每个标记了推荐续跑的任务都刷新 payload（多任务时不能只刷当前任务）。"""
        for t in list(self.task_manager.tasks):
            tid = t.get("task_id")
            if not tid:
                continue
            task = self.task_manager.get_task(tid)
            if not task or not task.get("recommendation_resume_pending"):
                continue
            base = dict(task.get("recommendation_resume_payload") or {})
            dl = self.task_download_frames.get(tid)
            base["current_count"] = len(dl.download_items) if dl else base.get("current_count", 0)
            st = self.task_states.get(tid)
            if tid == self.current_task_id:
                base["keyword_video_map"] = dict(self.keyword_video_map or base.get("keyword_video_map") or {})
                base["seen_video_ids"] = list(self.seen_video_ids or base.get("seen_video_ids") or [])
                if self.current_user_request:
                    base["user_request"] = self.current_user_request
            elif st:
                base["keyword_video_map"] = dict(st.get("keyword_video_map") or base.get("keyword_video_map") or {})
                base["seen_video_ids"] = list(st.get("seen_video_ids") or base.get("seen_video_ids") or [])
                ur = st.get("current_user_request")
                if ur:
                    base["user_request"] = ur
            self._persist_recommendation_payload(tid, base)

    def persist_all_running_task_cache_before_exit(self):
        """
        右上角关闭窗口前：把内存里的对话、搜索 UI 状态、右侧帧提取进度落盘。
        不关闭浏览器；先暂停任务再写盘，减少后台线程与写入交错。
        """
        if self.current_task_id:
            try:
                self._snapshot_outgoing_task_ui_state(self.current_task_id)
            except Exception:
                pass
        # 凡已创建过聊天区的任务都落盘（不仅限于 task_states 里曾写入的键）
        persist_tids = set(self.task_states.keys()) | set(self.task_widgets.keys())
        for tid in list(persist_tids):
            if tid not in self.task_states:
                continue
            state = self.task_states[tid]
            try:
                msgs = state.get("current_messages")
                if msgs is not None:
                    self.task_manager.save_task_messages(tid, list(msgs))
            except Exception as e:
                print(f"保存任务 {tid} 对话失败: {e}")
            try:
                self.persist_task_ui_state(tid)
            except Exception as e:
                print(f"保存任务 {tid} UI 状态失败: {e}")
            try:
                tc = state.get("target_video_count")
                if tc is not None:
                    self.task_manager.update_task_info(tid, {"target_video_count": tc})
            except Exception as e:
                print(f"保存任务 {tid} 目标数量失败: {e}")
        for tid, dlf in list(self.task_download_frames.items()):
            try:
                dlf.flush_items_status_to_disk()
            except Exception as e:
                print(f"刷新任务 {tid} 帧列表缓存失败: {e}")
        try:
            self._flush_recommendation_resume_on_exit()
        except Exception as e:
            print(f"保存推荐续跑状态失败: {e}")

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#f8f9fa")
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)

        self.create_task_btn = ctk.CTkButton(
            side,
            text=self.tr("create_task"),
            height=45,
            font=ctk.CTkFont(family="Microsoft YaHei", weight="bold"),
            corner_radius=20,
            fg_color="#e6f2ff",
            hover_color="#4a7acc",
            text_color="#5c9eff",
            command=self.create_new_task,
        )
        self.create_task_btn.pack(pady=20, padx=15, fill="x")
        self.task_list_title = ctk.CTkLabel(
            side,
            text=self.tr("task_list"),
            font=ctk.CTkFont(family="Microsoft YaHei", weight="bold"),
            text_color="#666",
        )
        self.task_list_title.pack(anchor="w", padx=20)
        self.task_list = ctk.CTkScrollableFrame(side, fg_color="transparent")
        self.task_list.pack(fill="both", expand=True, padx=10, pady=5)
        lang_row = ctk.CTkFrame(side, fg_color="transparent")
        lang_row.pack(side="bottom", fill="x", padx=12, pady=10)
        self.lang_title_label = ctk.CTkLabel(
            lang_row,
            text=self.tr("language_label"),
            font=ctk.CTkFont(family="Microsoft YaHei", size=12),
            text_color="#666",
            anchor="w",
        )
        self.lang_title_label.pack(side="left", padx=(4, 8))
        lang_values = [I18N["zh"]["language_option_zh"], I18N["en"]["language_option_en"]]
        self.lang_option = ctk.CTkOptionMenu(
            lang_row,
            values=lang_values,
            width=110,
            height=28,
            command=self.on_language_changed,
        )
        self.lang_option.pack(side="right")
        self.lang_option.set(
            I18N["zh"]["language_option_zh"] if self.lang == "zh" else I18N["en"]["language_option_en"]
        )

        chat = ctk.CTkFrame(self, corner_radius=0, fg_color="#f8f9fa")
        chat.grid(row=0, column=1, sticky="nsew")
        chat.grid_columnconfigure(0, weight=1)
        chat.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(chat, height=60, corner_radius=0, fg_color="#f5f5f5")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=0)

        self.title_bar = ctk.CTkLabel(top, text=self.tr("app_title"),
                                      font=ctk.CTkFont(family="Microsoft YaHei", size=16, weight="bold"))
        self.title_bar.grid(row=0, column=0, pady=15, sticky="n")

        top_btns = ctk.CTkFrame(top, fg_color="transparent")
        top_btns.grid(row=0, column=1, padx=20, pady=15, sticky="e")
        self._download_btn_wrap = ctk.CTkFrame(top_btns, fg_color="transparent")
        self._download_btn_wrap.pack(side="left", padx=(0, 6))
        self._download_count_text = ctk.CTkLabel(
            self._download_btn_wrap,
            text="",
            font=ctk.CTkFont(family="Microsoft YaHei", size=11),
            text_color="#b0b0b0",
        )
        self.toggle_download_btn = ctk.CTkButton(
            self._download_btn_wrap, text="📥", width=34, height=30, font=ctk.CTkFont(size=14),
            fg_color="#165DFF", hover_color="#0E42D2", text_color="#ffffff",
            corner_radius=6, command=self.toggle_frame_extract_view)
        self.toggle_download_btn.pack(side="left")
        self.toggle_chat_btn = ctk.CTkButton(
            top_btns, text="💬", width=34, height=30, font=ctk.CTkFont(size=14),
            fg_color="#07c160", hover_color="#059A4C", text_color="#ffffff",
            corner_radius=6, command=self.toggle_frame_extract_view)
        self.toggle_chat_btn.pack(side="left")
        self.toggle_chat_btn.pack_forget()

        self.chat_container = ctk.CTkFrame(chat, fg_color="transparent")
        self.chat_container.grid(row=1, column=0, sticky="nsew")
        self.chat_container.grid_columnconfigure(0, weight=1)
        self.chat_container.grid_rowconfigure(0, weight=1)

        self.resume_row = ctk.CTkFrame(chat, fg_color="transparent")
        self.resume_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.resume_continue_btn = ctk.CTkLabel(
            self.resume_row, text=self.tr("resume_session"),
            font=ctk.CTkFont(family="Microsoft YaHei", size=13, weight="bold"),
            text_color="#165DFF", cursor="hand2")
        self.resume_continue_btn.pack(anchor="center")
        self.resume_continue_btn.bind("<Button-1>", lambda _e: self.on_resume_session_clicked())
        if not getattr(self, "pending_resume_click", False):
            self.resume_row.grid_remove()

        # 搜索/推荐的暂停与继续改为聊天输入指令，不再显示底部控制条
        self.recommendation_control_row = None
        self.rec_flow_toggle_btn = None

        self.input_box = ctk.CTkFrame(chat, height=80, corner_radius=0, fg_color="#f8f9fa")
        self.input_box.grid(row=3, column=0, sticky="ew", padx=20, pady=0)
        self.input_box.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_box, height=50, font=ctk.CTkFont(family="Microsoft YaHei", size=14),
                                  placeholder_text="",
                                  border_width=1, border_color="#e0e0e0", fg_color="#ffffff")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=15)
        self.entry.bind("<Return>", lambda _: self.send())
        self.entry.bind("<KeyRelease>", self.on_entry_change)

        self.send_btn = ctk.CTkButton(self.input_box, text="⬆", width=40, height=40,
                                      font=ctk.CTkFont(family="Microsoft YaHei", size=18, weight="bold"),
                                      fg_color="#8cb5ff", hover_color="#4a7acc", corner_radius=20, command=self.send)
        self.send_btn.grid(row=0, column=1, padx=5, pady=15)
        self._refresh_search_hint_placeholder()

    def on_entry_change(self, event):
        try:
            if self.current_task_id:
                self.get_task_state(self.current_task_id)["input_draft"] = self.entry.get()
        except Exception:
            pass
        if self.entry.get().strip():
            self.send_btn.configure(fg_color="#4a7acc", hover_color="#2c5cb5")
        else:
            self.send_btn.configure(fg_color="#8cb5ff", hover_color="#4a7acc")

    def _refresh_search_hint_placeholder(self):
        """搜索/推荐进行中时，在输入框显示可用指令提示。"""
        if not hasattr(self, "entry"):
            return
        active = bool(getattr(self, "recommendation_flow_active", False))
        try:
            if self.current_task_id:
                active = active or bool(self.get_task_state(self.current_task_id).get("is_searching"))
        except Exception:
            pass
        try:
            self.entry.configure(
                placeholder_text=self.tr("search_placeholder_active") if active else self.tr("search_placeholder_idle")
            )
        except Exception:
            pass

    def toggle_frame_extract_view(self):
        """📥：用帧提取宫格替换聊天区；💬：切回聊天。"""
        if not self.current_task_id:
            return
        self.frame_extract_view_visible = not self.frame_extract_view_visible
        tid = self.current_task_id
        if self.frame_extract_view_visible and getattr(self, "_fec_list_deferred_for_task", None) == tid:
            # 即使未点「继续进程」，用户主动点开 📥 也要先加载当前帧提取列表（仅加载，不自动继续跑队列）
            self._fec_list_deferred_for_task = None
            try:
                self.reload_download_list_for_task(tid)
            except Exception:
                pass
        df = self._ensure_download_list_frame(tid)
        self.download_list_frame = df

        if self.frame_extract_view_visible:
            # 用户主动进入帧提取页时，允许当前页封面立即加载（不等待「继续进程」）。
            try:
                df._suppress_covers_until_resume = False
            except Exception:
                pass
            df.set_frame_extract_visible(True)
            for t, w in self.task_widgets.items():
                if "msg_area" in w:
                    w["msg_area"].grid_remove()
            df.grid(row=0, column=0, sticky="nsew")
            self.input_box.grid_remove()
            self.resume_row.grid_remove()
            self._download_btn_wrap.pack_forget()
            self.toggle_chat_btn.pack(side="left")
            try:
                df.jump_to_active_processing_page()
                df.render_current_page()
            except Exception:
                pass
            self._refresh_download_badge()
        else:
            df.set_frame_extract_visible(False)
            df.grid_remove()
            if tid in self.task_widgets:
                self.task_widgets[tid]["msg_area"].grid(row=0, column=0, sticky="nsew")
            self.input_box.grid()
            self._refresh_session_resume_row_visibility()
            self._sync_recommendation_bar_visibility()
            self.toggle_chat_btn.pack_forget()
            self._download_btn_wrap.pack(side="left", padx=(0, 6))
            self._refresh_download_badge()

    def get_current_msg_area(self):
        if self.current_task_id and self.current_task_id in self.task_widgets:
            return self.task_widgets[self.current_task_id]['msg_area']
        return None

    def get_msg_area_for_task(self, task_id):
        if task_id and task_id in self.task_widgets:
            return self.task_widgets[task_id]['msg_area']
        return None

    def get_current_status_frame(self):
        if self.current_task_id and self.current_task_id in self.task_widgets:
            return self.task_widgets[self.current_task_id].get('status_frame')
        return None

    def update_status_message(self, text, text_color="#888888", task_id=None):
        """更新状态消息（task_id 指定写入哪个任务的聊天区；默认当前选中任务）"""
        if getattr(self, "lang", "zh") == "en":
            text = self._localize_runtime_text_en(text)
        tid = task_id if task_id is not None else self.current_task_id
        msg_area = self.get_msg_area_for_task(tid)
        if not msg_area:
            return
        state = self.get_task_state(tid)
        status_frame = state.get('status_frame')

        if status_frame is None:
            status_frame = StatusMessageFrame(msg_area, text)
            status_frame.pack(fill="x", pady=2)
            state['status_frame'] = status_frame
            if tid in self.task_widgets:
                self.task_widgets[tid]['status_frame'] = status_frame
            msg_area._parent_canvas.yview_moveto(1)
        else:
            status_frame.update_text(text, text_color)
            msg_area._parent_canvas.yview_moveto(1)

    def _localize_runtime_text_en(self, text):
        s = str(text or "")
        mapping = [
            ("正在分析搜索需求", "Analyzing search intent"),
            ("已生成", "Generated"),
            ("个搜索关键词", "search keywords"),
            ("准备开始搜索", "ready to search"),
            ("正在使用浏览器", "Launching browser"),
            ("正在搜索关键词", "Searching keyword"),
            ("已找到", "Found"),
            ("个视频，继续搜索", "videos, continue searching"),
            ("已达到目标数量", "Target reached"),
            ("停止搜索", "stop searching"),
            ("正在下载", "Downloading"),
            ("个视频的缩略图", "video thumbnails"),
            ("下载缩略图", "Downloading thumbnail"),
            ("搜索完成", "Search completed"),
            ("搜索出错", "Search error"),
            ("继续搜索推荐", "Continue recommendation search"),
            ("目标还差约", "about remaining"),
            ("个视频", "videos"),
            ("搜索关键词", "Search keyword"),
            ("没有找到新视频", "found no new videos"),
            ("找到", "Found"),
            ("个新视频，开始分析", "new videos, start analyzing"),
            ("分析中", "Analyzing"),
            ("符合偏好", "Matches preference"),
            ("不符合", "Not matched"),
            ("本轮搜索推荐完成", "This recommendation round completed"),
            ("多轮推荐完成", "Multi-round recommendation completed"),
            ("正在根据偏好推荐更多视频", "Recommending more videos based on your preference"),
            ("正在根据偏好推荐更多视频...", "Recommending more videos based on your preference..."),
            ("暂停搜索", "Pause search"),
            ("继续搜索", "Resume search"),
        ]
        for zh, en in mapping:
            s = s.replace(zh, en)
        return s

    def clear_status_message(self, task_id=None):
        tid = task_id if task_id is not None else self.current_task_id
        state = self.get_task_state(tid)
        status_frame = state.get('status_frame')
        if status_frame is not None:
            status_frame.destroy()
            state['status_frame'] = None
            if tid in self.task_widgets:
                self.task_widgets[tid]['status_frame'] = None

    def on_video_selection_change(self, video, selected):
        self.update_task_video_selection(video['video_id'], selected)
        self.update_stats_display()

    def update_task_video_selection(self, video_id, selected):
        task = self.task_manager.get_task(self.current_task_id)
        if task:
            if 'selected_videos' not in task:
                task['selected_videos'] = []
            if 'deselected_videos' not in task:
                task['deselected_videos'] = []
            if selected:
                if video_id in task['deselected_videos']:
                    task['deselected_videos'].remove(video_id)
                if video_id not in task['selected_videos']:
                    task['selected_videos'].append(video_id)
            else:
                if video_id in task['selected_videos']:
                    task['selected_videos'].remove(video_id)
                if video_id not in task['deselected_videos']:
                    task['deselected_videos'].append(video_id)
            self.task_manager.update_task_info(self.current_task_id, {
                'selected_videos': task['selected_videos'],
                'deselected_videos': task['deselected_videos']
            })

    def get_selected_videos_count(self):
        state = self.get_task_state(self.current_task_id)
        count = 0
        for msg in state['current_messages']:
            if msg.get('type') == 'video_list':
                for video in msg.get('videos', []):
                    if video.get('selected', True):
                        count += 1
        return count

    def get_total_videos_count(self):
        state = self.get_task_state(self.current_task_id)
        count = 0
        for msg in state['current_messages']:
            if msg.get('type') == 'video_list':
                count += len(msg.get('videos', []))
        return count

    def update_stats_display(self):
        """左下角「已选」统计已移除，保留空实现避免旧调用报错。"""
        pass

    def load_tasks_to_sidebar(self):
        for widget in self.task_list.winfo_children():
            widget.destroy()
        self.task_items = []
        for task in self.task_manager.tasks:
            task_item = TaskItemFrame(self.task_list, task, self)
            self.task_items.append(task_item)

    def update_task_items_style(self):
        for item in self.task_items:
            item.update_selected_style()

    def create_new_task(self):
        t = self.task_manager.create_task()
        self.load_tasks_to_sidebar()
        self.switch_task(t['task_id'])

    def create_task_message_area(self, task_id):
        msg_area = ctk.CTkScrollableFrame(self.chat_container, fg_color="#ffffff")
        msg_area.grid(row=0, column=0, sticky="nsew")
        msg_area.grid_columnconfigure(0, weight=1)
        self.task_widgets[task_id] = {'msg_area': msg_area, 'status_frame': None}
        return msg_area

    def _bump_load_job(self):
        self._load_job_seq += 1
        return self._load_job_seq

    def prepare_messages_state_for_area(self, task_id, msg_area):
        for widget in msg_area.winfo_children():
            widget.destroy()
        messages = self.task_manager.load_task_messages(task_id)
        state = self.get_task_state(task_id)
        if not messages:
            welcome_msg = {"type": "text", "content": self.tr("welcome_text"),
                           "is_user": False}
            messages = [welcome_msg]
            self.task_manager.save_task_messages(task_id, messages)
        state['current_messages'] = messages
        task = self.task_manager.get_task(task_id)
        selected_videos = task.get('selected_videos', []) if task else []
        deselected_videos = task.get('deselected_videos', []) if task else []
        for msg in messages:
            if msg.get('type') == 'video_list':
                for video in msg.get('videos', []):
                    if video['video_id'] in deselected_videos:
                        video['selected'] = False
                    elif video['video_id'] in selected_videos:
                        video['selected'] = True
                    else:
                        video['selected'] = True
        return messages

    def load_messages_to_area(self, task_id, msg_area):
        messages = self.prepare_messages_state_for_area(task_id, msg_area)
        state = self.get_task_state(task_id)
        for msg in messages:
            is_user = msg.get('is_user', False)
            bubble = MessageBubble(msg_area, msg, self.task_manager, task_id, is_user=is_user,
                                   on_selection_change=self.on_video_selection_change,
                                   on_confirm=self.on_confirm_analysis if msg.get('type') == 'video_list' else None,
                                   user_request=state.get('current_user_request') if msg.get(
                                       'type') == 'video_list' else None, app=self)
            bubble.pack(fill="x", pady=2)
        msg_area._parent_canvas.yview_moveto(1)

    def _ordered_fec_videos_for_task(self, task):
        if not task:
            return []
        fec = task.get('frame_extract_cache', {}) or {}
        ordered_ids = []
        for v in (task.get('videos', []) or []):
            vid = v.get('video_id')
            if vid and vid in fec:
                ordered_ids.append(vid)
        for vid in list(fec.keys()):
            if vid not in ordered_ids:
                ordered_ids.append(vid)
        out = []
        for vid in ordered_ids:
            entry = fec.get(vid) or {}
            video = entry.get('video')
            if not video:
                continue
            if entry.get('status_text') == "已删除":
                continue
            out.append(video)
        return out

    def _switch_task_finalize(self, task_id):
        _trace("_switch_task_finalize", f"task_id={task_id}")
        state = self.get_task_state(task_id)
        self.current_messages = state.get('current_messages', [])
        self.current_user_request = state.get('current_user_request')
        kwm = state.get('keyword_video_map') or {}
        self.keyword_video_map = dict(kwm) if isinstance(kwm, dict) else {}
        sid = state.get('seen_video_ids')
        if isinstance(sid, set):
            self.seen_video_ids = set(sid)
        else:
            self.seen_video_ids = set(sid or [])
        self.update_stats_display()
        try:
            self.update_idletasks()
        except RecursionError:
            pass
        self._sync_recommendation_bar_visibility()
        self._refresh_session_resume_row_visibility()
        if task_should_show_session_resume_banner(
                self.task_manager.get_task(task_id), read_session_resume_pending(), self.task_manager
        ) and task_id in self.task_download_frames:
            try:
                self.task_download_frames[task_id].set_startup_resume_pending_ui()
            except Exception:
                pass
        if getattr(self, "_startup_resume_hooks_after_load", False):
            self._startup_resume_hooks_after_load = False
            self.pending_resume_click = task_should_show_session_resume_banner(
                self.task_manager.get_task(self.current_task_id),
                read_session_resume_pending(),
                self.task_manager,
            )
            self._refresh_session_resume_row_visibility()
            if self.pending_resume_click:
                self._apply_resume_session_startup_after_load()
        try:
            self._refresh_download_badge()
        except Exception:
            pass

    def _incremental_chain_done(self, job, task_id):
        if job != self._load_job_seq:
            return
        self._incr_chains_remaining -= 1
        if self._incr_chains_remaining <= 0:
            self._switch_task_finalize(task_id)
            # 非「待继续进程」任务：启动链结束后仍可错峰补右侧封面；待继续进程的任务由点击后继续 + _THUMB_RECOVERY_DELAY_MS 补图
            df = self.task_download_frames.get(task_id)
            if df and not getattr(df, "_suppress_covers_until_resume", False):

                def _startup_right_thumbs_non_deferred():
                    try:
                        if self.winfo_exists() and self.current_task_id == task_id:
                            df.schedule_missing_thumbnails_in_order(slow=True)
                    except Exception:
                        pass

                self.after(8000, _startup_right_thumbs_non_deferred)

    def _incremental_mark_dl_done(self, job, task_id):
        if job != self._load_job_seq:
            return
        cb = getattr(self, "_incr_after_dl_batch_callback", None)
        if cb:
            self._incr_after_dl_batch_callback = None
            try:
                cb()
            except Exception:
                pass
        self._incremental_chain_done(job, task_id)

    def _task_should_defer_right_panel_covers(self, task):
        """任务处于暂停且仍有未完成帧提取/推荐时：右侧列表仅占位封面，直至用户点击主界面「继续进程」。"""
        if not task:
            return False
        if not task.get("task_paused"):
            return False
        if not task.get("ever_confirmed_to_download"):
            return False
        return bool(task_has_incomplete_extraction(task) or task.get("recommendation_resume_pending"))

    def _ensure_download_list_frame(self, task_id):
        """右侧帧列表：首次需要再加行时才创建控件树，未点开 📥 时可推迟创建以减轻冷启动。"""
        if task_id not in self.task_download_frames:
            task = self.task_manager.get_task(task_id)
            defer = self._task_should_defer_right_panel_covers(task)
            self.task_download_frames[task_id] = DownloadListFrame(
                self.chat_container, self.task_manager, task_id, defer_covers_until_resume=defer, app=self
            )
        else:
            self.task_download_frames[task_id]._app = self
        return self.task_download_frames[task_id]

    def _refresh_download_badge(self):
        """聊天视图下：📥 左侧浅灰字显示 已完成 已完成数/列表总数；帧提取全屏、无数或已全部完成且无进行中时隐藏。"""
        if not hasattr(self, "_download_count_text"):
            return
        tid = self.current_task_id
        if (
            not tid
            or getattr(self, "frame_extract_view_visible", False)
            or tid not in self.task_download_frames
        ):
            self._download_count_text.pack_forget()
            return
        df = self.task_download_frames[tid]
        order = getattr(df, "_video_order", None) or []
        total = len(order)
        if total == 0:
            self._download_count_text.pack_forget()
            return
        completed = 0
        for video_id in order:
            cached = self.task_manager.get_video_extraction_status(tid, video_id)
            if cached.get("status") == "已完成":
                completed += 1
                continue
            item = df.download_items.get(video_id)
            if item and hasattr(item, "get_status_text"):
                if "已完成" in (item.get_status_text() or ""):
                    completed += 1
        has_active = bool(getattr(df, "active_extractions", set()))
        if not has_active:
            try:
                with df.queue_lock:
                    has_active = bool(df.extraction_queue)
            except Exception:
                pass
        if completed >= total and not has_active:
            self._download_count_text.pack_forget()
            return
        target_total = None
        try:
            t = self.task_manager.get_task(tid) if tid else None
            target_total = (t or {}).get("target_video_count")
        except Exception:
            target_total = None
        self._download_count_text.configure(
            text=format_completed_progress_text(completed, total, target_total)
        )
        try:
            if not self._download_count_text.winfo_ismapped():
                self._download_count_text.pack(side="left", padx=(0, 6), before=self.toggle_download_btn)
        except Exception:
            self._download_count_text.pack(side="left", padx=(0, 6))

    def _run_incremental_dl_batch(self, job, task_id):
        if job != self._load_job_seq:
            _trace("_run_incremental_dl_batch skip", f"stale job {job} != {self._load_job_seq}")
            return
        _trace("_run_incremental_dl_batch", f"job={job} task={task_id} hydrate (no per-row UI)")
        self.reload_download_list_for_task(task_id)
        self._incremental_mark_dl_done(job, task_id)

    def _defer_fec_list_until_resume_click(self, task_id):
        """冷启动且任务暂停、用户已发过需求且仍有未完成帧/推荐时：不预填右侧 fec 行，等用户点击「继续进程」后再加载。"""
        t = self.task_manager.get_task(task_id)
        if not t or not t.get("task_paused"):
            return False
        if not task_has_user_first_request_message(self.task_manager, task_id):
            return False
        if not t.get("ever_confirmed_to_download"):
            return False
        return bool(task_has_incomplete_extraction(t) or t.get("recommendation_resume_pending"))

    def _incremental_mark_msg_done(self, job, task_id):
        if job != self._load_job_seq:
            return
        # 聊天区（含 defer 的 video_list 卡片）全部就绪后再跑右侧 fec 列表，避免 1000 行与消息区抢主线程导致长时间「假死」
        _trace("_incremental_mark_msg_done", f"chat done, starting dl batch task={task_id}")
        try:
            self._switch_task_finalize(task_id)
        except Exception:
            pass
        if self._defer_fec_list_until_resume_click(task_id):
            self._fec_list_deferred_for_task = task_id
            self._incremental_mark_dl_done(job, task_id)
            return
        fec_delay = max(400, int(getattr(self, "_STARTUP_FEC_DELAY_AFTER_CHAT_MS", 2200)))
        self.after(fec_delay, lambda: self._run_incremental_dl_batch(job, task_id))

    def _run_incremental_msg_step(self, job, task_id):
        if job != self._load_job_seq:
            _trace("_run_incremental_msg_step skip", f"stale job {job}")
            return
        msg_area = self._incr_msg_area
        messages = self._incr_msg_list
        i = self._incr_msg_index
        if i >= len(messages):
            _trace("_run_incremental_msg_step done", f"task={task_id} total_msgs={len(messages)}")
            self._incremental_mark_msg_done(job, task_id)
            return
        msg = messages[i]
        _trace("_run_incremental_msg_step", f"task={task_id} msg[{i}] type={msg.get('type')}")
        self._incr_msg_index = i + 1
        state = self.get_task_state(task_id)
        is_user = msg.get('is_user', False)

        def schedule_next():
            if job != self._load_job_seq:
                return
            self.after(self._STARTUP_MSG_DELAY_MS, lambda: self._run_incremental_msg_step(job, task_id))

        defer_cards = msg.get('type') == 'video_list' and len(msg.get('videos') or []) > 0
        if defer_cards:
            def on_cards_done():
                if job != self._load_job_seq:
                    return
                schedule_next()

            bubble = MessageBubble(
                msg_area, msg, self.task_manager, task_id, is_user=is_user,
                on_selection_change=self.on_video_selection_change,
                on_confirm=self.on_confirm_analysis if msg.get('type') == 'video_list' else None,
                user_request=state.get('current_user_request') if msg.get('type') == 'video_list' else None,
                app=self, defer_video_cards=True, on_video_cards_finished=on_cards_done)
            bubble.pack(fill="x", pady=2)
        else:
            bubble = MessageBubble(
                msg_area, msg, self.task_manager, task_id, is_user=is_user,
                on_selection_change=self.on_video_selection_change,
                on_confirm=self.on_confirm_analysis if msg.get('type') == 'video_list' else None,
                user_request=state.get('current_user_request') if msg.get('type') == 'video_list' else None,
                app=self)
            bubble.pack(fill="x", pady=2)
            schedule_next()

        try:
            msg_area._parent_canvas.yview_moveto(1)
        except Exception:
            pass

    def _switch_task_incremental(self, job, task_id):
        _trace("_switch_task_incremental", f"job={job} task={task_id}")
        # 仅右侧列表链参与「完成计数」；消息链结束后会再启动该链（见 _incremental_mark_msg_done）
        self._incr_chains_remaining = 1
        task = self.task_manager.get_task(task_id)
        self._incr_dl_list = self._ordered_fec_videos_for_task(task)
        self._incr_dl_index = 0

        if not self.frame_extract_view_visible and task_id in self.task_download_frames:
            try:
                self.task_download_frames[task_id].clear_all()
            except Exception:
                pass

        if task_id not in self.task_widgets:
            msg_area = self.create_task_message_area(task_id)
        else:
            msg_area = self.task_widgets[task_id]['msg_area']
            msg_area.grid()
            if 'scroll_position' in self.task_widgets[task_id]:
                try:
                    msg_area._parent_canvas.yview_moveto(self.task_widgets[task_id]['scroll_position'])
                except Exception:
                    pass

        self._incr_msg_list = self.prepare_messages_state_for_area(task_id, msg_area)
        self._incr_msg_index = 0
        self._incr_msg_area = msg_area

        if self.frame_extract_view_visible:
            try:
                msg_area.grid_remove()
            except Exception:
                pass
            df = self._ensure_download_list_frame(task_id)
            try:
                df.clear_all()
            except Exception:
                pass
            self.download_list_frame = df
            df.set_frame_extract_visible(True)
            df.grid(row=0, column=0, sticky="nsew")
            self.input_box.grid_remove()
            self.resume_row.grid_remove()

        self.after(1, lambda: self._run_incremental_msg_step(job, task_id))

    def reload_download_list_for_task(self, task_id):
        _resume_flow_dbg("reload_download_list.enter", f"task_id={task_id}")
        download_frame = self._ensure_download_list_frame(task_id)
        _resume_flow_dbg("reload_download_list.after_ensure_frame", f"task_id={task_id}")
        download_frame.clear_all()
        _resume_flow_dbg("reload_download_list.after_clear_all", f"task_id={task_id}")
        task = self.task_manager.get_task(task_id)
        if not task:
            _resume_flow_dbg("reload_download_list.no_task_return", f"task_id={task_id}")
            return

        # 从帧提取专用缓存恢复右侧列表（成员+状态）。
        # task['videos'] 仅表示聊天区「首轮搜索结果」，未点确认不得入队；空 fec 不能从 videos 迁移，否则会误启动下载。
        fec = task.get('frame_extract_cache', {}) or {}

        # 先按 task['videos'] 顺序恢复已在 fec 中的条目；再补 fec 中其余（二次推荐等）
        ordered_ids = []
        for v in (task.get('videos', []) or []):
            vid = v.get('video_id')
            if vid and vid in fec:
                ordered_ids.append(vid)
        for vid in fec.keys():
            if vid not in ordered_ids:
                ordered_ids.append(vid)

        n_ids = len(ordered_ids)
        _resume_flow_dbg("reload_download_list.loop_start", f"n={n_ids} fec_keys={len(fec)}")
        for i, vid in enumerate(ordered_ids):
            if i == 0 or i == n_ids - 1 or (n_ids > 12 and i % max(1, n_ids // 6) == 0):
                _resume_flow_dbg("reload_download_list.loop", f"i={i + 1}/{n_ids} vid={vid}")
            entry = fec.get(vid) or {}
            video = entry.get('video')
            if not video:
                continue
            if entry.get('status_text') == "已删除":
                continue
            download_frame._video_payload[vid] = video
            download_frame._video_order.append(vid)

            tsk = self.task_manager.get_task(task_id)
            if tsk and not tsk.get("task_paused"):
                # 仅对「未完成」条目做进度回填；已完成状态绝不能在切换任务时被覆盖成“提取中”。
                cached_status = self.task_manager.get_video_extraction_status(task_id, vid) or {}
                cached_state = cached_status.get("status")
                if cached_state in ("已完成", "失败"):
                    continue
                if str(cached_status.get("status_text") or "") == "已删除":
                    continue
                t0 = time.perf_counter()
                frames_info = self.task_manager.get_video_frames_info(task_id, video.get('title', ''), vid)
                gi = (time.perf_counter() - t0) * 1000.0
                if gi > 50.0:
                    _resume_flow_dbg("reload_download_list.slow_get_frames_info", f"vid={vid} {gi:.0f}ms")
                total_frames = frames_info.get('total_frames', 0)
                if total_frames > 0:
                    last_t = frames_info.get('last_extracted_time', -1)
                    progress = min(100, int(total_frames * 100 / 1000))
                    status_text = f"⏳ 提取中... ({total_frames}帧)"
                    self.task_manager.update_video_extraction_status(
                        task_id, vid, status_text, progress, total_frames, last_t
                    )

        _resume_flow_dbg("reload_download_list.after_ordered_loop", f"n={n_ids}")

        show_frame = bool(self.frame_extract_view_visible and self.current_task_id == task_id)
        if show_frame:
            download_frame.jump_to_active_processing_page()
            _resume_flow_dbg("reload_download_list.after_jump_page", f"page={download_frame._current_page}")
        else:
            download_frame._current_page = 0
            _resume_flow_dbg("reload_download_list.page_reset", "show_frame=False")
        download_frame.update_count_display()
        _resume_flow_dbg("reload_download_list.after_update_count")

        def _defer_refill():
            try:
                _resume_flow_dbg("reload_download_list._defer_refill.enter", f"task_id={task_id}")
                if self.current_task_id != task_id:
                    _resume_flow_dbg("reload_download_list._defer_refill.skip_wrong_task", "")
                    return
                download_frame._refill_extraction_queue_after_hydrate()
                _resume_flow_dbg("reload_download_list._defer_refill.done", f"task_id={task_id}")
            except Exception:
                pass

        self.after(0, _defer_refill)
        _resume_flow_dbg("reload_download_list.scheduled_after0_refill", "")

        download_frame.set_frame_extract_visible(show_frame)
        _resume_flow_dbg("reload_download_list.after_set_visible", f"show={show_frame}")
        try:
            if show_frame:
                _resume_flow_dbg("reload_download_list.before_render_current_page", "")
                download_frame.render_current_page()
                _resume_flow_dbg("reload_download_list.after_render_current_page", "")
        except Exception:
            pass

        def _reload_list_thumbs():
            try:
                if (
                    self.current_task_id == task_id
                    and self.frame_extract_view_visible
                    and not getattr(download_frame, "_suppress_covers_until_resume", False)
                ):
                    download_frame.schedule_missing_thumbnails_in_order(slow=True)
            except Exception:
                pass

        self.after(5000, _reload_list_thumbs)
        _resume_flow_dbg("reload_download_list.exit", f"task_id={task_id}")

    def switch_task(self, task_id, incremental_load=False):
        _trace("switch_task", f"task_id={task_id} incremental={incremental_load} job={getattr(self, '_load_job_seq', '?')}")
        if not incremental_load:
            self._startup_resume_hooks_after_load = False
        self._bump_load_job()
        old_tid = self.current_task_id
        try:
            if old_tid:
                self.get_task_state(old_tid)["input_draft"] = self.entry.get()
        except Exception:
            pass
        if old_tid != task_id:
            self._fec_list_deferred_for_task = None
        if old_tid and old_tid != task_id:
            self._snapshot_outgoing_task_ui_state(old_tid)
        if self.current_task_id and self.current_task_id in self.task_widgets:
            current_area = self.task_widgets[self.current_task_id]['msg_area']
            current_area.grid_remove()
            try:
                scroll_pos = current_area._parent_canvas.yview()[0]
                self.task_widgets[self.current_task_id]['scroll_position'] = scroll_pos
            except Exception:
                pass
        self.current_task_id = task_id
        try:
            draft = self.get_task_state(task_id).get("input_draft", "")
            self.entry.delete(0, "end")
            if draft:
                self.entry.insert(0, draft)
            self.on_entry_change(None)
        except Exception:
            pass
        self.update_task_items_style()
        task = self.task_manager.get_task(task_id)
        if task:
            self.title_bar.configure(text=self.tr("app_title"))

        if incremental_load:
            self._switch_task_incremental(self._load_job_seq, task_id)
            return

        if task_id not in self.task_widgets:
            msg_area = self.create_task_message_area(task_id)
            self.load_messages_to_area(task_id, msg_area)
        else:
            msg_area = self.task_widgets[task_id]['msg_area']
            msg_area.grid(row=0, column=0, sticky="nsew")
            if "scroll_position" in self.task_widgets[task_id]:
                try:
                    msg_area._parent_canvas.yview_moveto(self.task_widgets[task_id]["scroll_position"])
                except Exception:
                    pass

        self.reload_download_list_for_task(task_id)

        if self.frame_extract_view_visible:
            try:
                msg_area.grid_remove()
            except Exception:
                pass
            df = self._ensure_download_list_frame(task_id)
            self.download_list_frame = df
            df.set_frame_extract_visible(True)
            df.grid(row=0, column=0, sticky="nsew")
            try:
                df.jump_to_active_processing_page()
                df.render_current_page()
            except Exception:
                pass
            self.input_box.grid_remove()
            self.resume_row.grid_remove()
        else:
            for _df in list(self.task_download_frames.values()):
                try:
                    _df.set_frame_extract_visible(False)
                    _df.grid_remove()
                except Exception:
                    pass
            self.input_box.grid()

        self._switch_task_finalize(task_id)

    def on_confirm_analysis(self, user_request, selected_videos, deselected_videos):
        tid = self.current_task_id
        if not tid:
            return
        # 后台线程里 self.current_task_id / keyword 映射可能已随切换任务而变，这里在主线程快照
        kw_capture = dict(self.keyword_video_map or {})
        seen_capture = set(self.seen_video_ids or set())

        # 首轮列表里出现过的视频（含取消勾选）一律记入任务级 uid，后续任何轮次不得再推荐
        try:
            _all_round1 = []
            for v in selected_videos or []:
                if v and v.get("video_id"):
                    _all_round1.append(v["video_id"])
            for v in deselected_videos or []:
                if v and v.get("video_id"):
                    _all_round1.append(v["video_id"])
            if _all_round1:
                self.task_manager.merge_seen_video_ids_batch(tid, _all_round1)
                for _vid in _all_round1:
                    self._register_seen_video(tid, _vid)
        except Exception:
            pass

        current_download_frame = self._ensure_download_list_frame(tid)

        try:
            self.task_manager.update_task_info(tid, {"ever_confirmed_to_download": True})
        except Exception:
            pass

        # 关闭窗口时 on_closing 会把所有任务标为暂停并写入磁盘；不解除则 add_video_to_queue 会直接 return，永远不会 _process_queue
        self.task_manager.set_task_paused(tid, False)
        try:
            current_download_frame.update_button_states()
        except Exception:
            pass

        # 未在帧提取视图时自动切到宫格，便于看到进度
        if not self.frame_extract_view_visible:
            self.toggle_frame_extract_view()

        for video in selected_videos:
            current_download_frame.add_video_to_queue(video)

        def analyze_and_continue():
            target_count = self.task_manager.get_task(tid).get('target_video_count')
            if target_count is None:
                target_count = 20

            # download_items 只包含已实例化条目（分页/延迟加载时会偏小），应按 _video_order 计总数
            current_count = len(getattr(current_download_frame, "_video_order", []) or [])

            if current_count >= target_count:
                self.after(0, lambda x=tid, tc=target_count: self.update_status_message(
                    f"✅ 已达到目标数量 {tc} 个，停止推荐", task_id=x))
                self.after(2000, lambda x=tid: self.clear_status_message(task_id=x))
                return

            analysis = analyze_user_preferences(user_request, selected_videos, deselected_videos)

            # 把“缓存偏好 + 本次分析偏好”组合起来给后续匹配用
            cached = {}
            try:
                task = self.task_manager.get_task(tid)
                cached = task.get("cached_preferences") if task else {}
            except Exception:
                cached = {}
            combined_preferences = {
                "cached_preferences": cached or {},
                "selection_analysis": analysis
            }

            self.start_secondary_search(
                tid, user_request, combined_preferences, deselected_videos, target_count, current_count,
                keyword_video_map=kw_capture, seen_video_ids=seen_capture)

            task = self.task_manager.get_task(tid)
            if task:
                preferences = {
                    'analysis': analysis,
                    'user_request': user_request,
                    'selected_count': len(selected_videos),
                    'deselected_count': len(deselected_videos),
                    'timestamp': datetime.now().isoformat()
                }
                if 'search_history' not in task:
                    task['search_history'] = []
                task['search_history'].append(preferences)
                self.task_manager.update_task_info(tid,
                                                   {'preferences': analysis, 'search_history': task['search_history']})
            analysis_msg = {"type": "text",
                            "content": self.tr("msg_pref_analyzed", analysis=analysis),
                            "is_user": False}
            self.after(0, lambda m=analysis_msg, x=tid: self.add_msg(m, task_id=x))

        threading.Thread(target=analyze_and_continue, daemon=True).start()

    def start_secondary_search(self, task_id, user_request, preferences, deselected_videos, target_count, current_count,
                               keyword_video_map=None, seen_video_ids=None):
        """
        继续搜索推荐（必须传入 task_id）。
        达到目标时视为「暂停」：secondary_kw_order / secondary_kw_cursor 写入暂停缓存；
        提高目标后再调用则从上次关键词进度接着跑，不刻意跳过或改顺序。
        """
        if not task_id:
            return
        self.recommendation_flow_active = True
        self.after(0, self._refresh_search_hint_placeholder)
        try:
            self.task_manager.set_task_paused(task_id, False)
        except Exception:
            pass
        try:
            target_count = int(target_count) if target_count is not None else None
        except (TypeError, ValueError):
            target_count = None
        current_count = self._current_download_total_for_task(task_id)
        if target_count is not None and target_count > 0 and current_count >= target_count:
            print(f"[推荐停止] 任务 {task_id} 当前 {current_count} 已达到/超过目标 {target_count}，不再继续搜索推荐")
            self.after(0, lambda tid=task_id: self._finish_recommendation_flow(tid))
            return
        if keyword_video_map is None:
            keyword_video_map = dict(self.keyword_video_map or {})
        else:
            keyword_video_map = dict(keyword_video_map)
        if seen_video_ids is None:
            seen_video_ids = set(self.seen_video_ids or set())
        else:
            seen_video_ids = set(seen_video_ids)
        merged_seen = self._merge_seen_into_runtime_and_disk(task_id)
        seen_video_ids |= merged_seen

        TaskThreadManager.ensure_recommendation_event(task_id)
        TaskThreadManager.resume_recommendation(task_id)

        def _snapshot_payload(extra=None):
            dl = self.task_download_frames.get(task_id)
            cc = len(getattr(dl, "_video_order", []) or []) if dl else current_count
            base = {
                "user_request": user_request,
                "preferences": preferences,
                "deselected_videos": deselected_videos,
                "target_count": target_count,
                "current_count": cc,
                "keyword_video_map": dict(keyword_video_map),
                "seen_video_ids": list(seen_video_ids),
            }
            if extra:
                base.update(extra)
            return base

        self._persist_recommendation_payload(task_id, _snapshot_payload())

        deselected_by_keyword = {}
        for video in deselected_videos:
            keyword = video.get('keyword')
            if keyword:
                if keyword not in deselected_by_keyword:
                    deselected_by_keyword[keyword] = []
                deselected_by_keyword[keyword].append(video)

        abandoned_keywords = set()
        for keyword, videos in deselected_by_keyword.items():
            keyword_videos = keyword_video_map.get(keyword, [])
            if len(videos) >= 2 and len(videos) >= len(keyword_videos):
                abandoned_keywords.add(keyword)

        all_keywords = list(keyword_video_map.keys())
        if not all_keywords:
            self._expand_keyword_map_with_qwen(task_id, user_request, keyword_video_map)
            all_keywords = list(keyword_video_map.keys())

        first_active_index = None
        for i, kw in enumerate(all_keywords):
            if kw not in abandoned_keywords:
                first_active_index = i
                break

        if first_active_index is None:
            self._expand_keyword_map_with_qwen(task_id, user_request, keyword_video_map)
            all_keywords = list(keyword_video_map.keys())
            for i, kw in enumerate(all_keywords):
                if kw not in abandoned_keywords:
                    first_active_index = i
                    break

        if first_active_index is None:
            self.after(0, lambda x=task_id: self.update_status_message(
                "⚠️ 无可用搜索词，已尝试用模型生成新词仍不足", task_id=x))
            self.after(2000, lambda x=task_id: self.clear_status_message(x))
            self.after(0, lambda tid=task_id: self._finish_recommendation_flow(tid))
            return

        ordered_keywords = all_keywords[first_active_index:] + all_keywords[:first_active_index]
        ordered_keywords = [kw for kw in ordered_keywords if kw not in abandoned_keywords]

        prev_snap = {}
        try:
            pt = self.task_manager.get_task(task_id)
            prev_snap = dict(pt.get("recommendation_resume_payload") or {})
        except Exception:
            prev_snap = {}
        prev_order = prev_snap.get("secondary_kw_order")
        try:
            prev_cursor = int(prev_snap.get("secondary_kw_cursor") or 0)
        except (TypeError, ValueError):
            prev_cursor = 0
        if (
            isinstance(prev_order, list)
            and prev_order
            and set(prev_order) == set(ordered_keywords)
            and all(k in keyword_video_map for k in prev_order)
            and 0 <= prev_cursor <= len(prev_order)
        ):
            ordered_keywords = list(prev_order)
            main_kw_resume = prev_cursor
        else:
            main_kw_resume = 0

        self.after(0, lambda x=task_id, need=max(0, (target_count or 0) - current_count), mc=main_kw_resume, nk=len(ordered_keywords): self.update_status_message(
            f"🔍 继续搜索推荐，目标还差约 {need} 个视频（关键词 {mc}/{nk}）...", task_id=x
        ))

        def process_keywords():
            nonlocal seen_video_ids
            seen_video_ids |= self._collect_all_seen_video_ids_for_task(task_id)
            total_analyzed = 0
            total_matched = 0

            try:
                cur0 = self._current_download_total_for_task(task_id)
                tc0 = int(target_count) if target_count is not None else 0
                if tc0 > 0 and cur0 < tc0:
                    self._recursive_sidebar_recommend(
                        task_id, user_request, preferences, tc0, cur0
                    )
            except Exception as e:
                print(f"[推荐] 前置侧边栏推荐异常: {e}", flush=True)

            def _persist_kw(cursor_idx):
                self._persist_recommendation_payload(
                    task_id,
                    _snapshot_payload(
                        {
                            "secondary_kw_order": list(ordered_keywords),
                            "secondary_kw_cursor": max(0, min(cursor_idx, len(ordered_keywords))),
                        }
                    ),
                )

            def run_keyword_loop(keyword_list, start_abs_index=0):
                nonlocal total_analyzed, total_matched
                for j, keyword in enumerate(keyword_list):
                    abs_k = start_abs_index + j
                    TaskThreadManager.wait_recommendation_running(task_id)
                    if TaskThreadManager.is_task_cancelled(task_id):
                        return
                    current_now = self._current_download_total_for_task(task_id)
                    remaining_needed = max(0, (target_count or 0) - current_now)
                    if remaining_needed <= 0:
                        print(f"[推荐停止] 任务 {task_id} 已达目标：{current_now}/{target_count}")
                        _persist_kw(abs_k)
                        break

                    _persist_kw(abs_k)

                    self.after(0, lambda kw=keyword, x=task_id: self.update_status_message(
                        f"🔍 搜索关键词: {kw}", task_id=x))

                    videos = get_all_videos_from_search(task_id, keyword, seen_video_ids)

                    if not videos:
                        self.after(0, lambda kw=keyword, x=task_id: self.update_status_message(
                            f"⚠️ 关键词 {kw} 没有找到新视频", task_id=x))
                        _persist_kw(abs_k + 1)
                        continue

                    self.after(0, lambda kw=keyword, count=len(videos), x=task_id: self.update_status_message(
                        f"📹 关键词 {kw} 找到 {count} 个新视频，开始分析...", task_id=x))

                    matched_videos = []
                    hit_target_mid_videos = False
                    for idx, video in enumerate(videos, 1):
                        TaskThreadManager.wait_recommendation_running(task_id)
                        if TaskThreadManager.is_task_cancelled(task_id):
                            return
                        current_now = self._current_download_total_for_task(task_id)
                        remaining_needed = max(0, (target_count or 0) - current_now)
                        if remaining_needed <= 0:
                            print(f"[推荐停止] 任务 {task_id} 已达目标：{current_now}/{target_count}")
                            _persist_kw(abs_k)
                            hit_target_mid_videos = True
                            break

                        vid_mark = (video or {}).get("video_id")
                        if vid_mark:
                            self._register_seen_video(task_id, vid_mark)
                            seen_video_ids.add(vid_mark)

                        self.after(0, lambda v=video, i=idx, t=len(videos), x=task_id: self.update_status_message(
                            f"🔍 分析中 ({i}/{t}): {v['title'][:50]}...", "#888888", x))

                        try:
                            _tsk = self.task_manager.get_task(task_id)
                            if not video_passes_duration_filter(_tsk, video):
                                continue
                        except Exception:
                            pass

                        total_analyzed += 1

                        try:
                            is_match = check_video_match(user_request, preferences, video['title'])
                        except Exception as e:
                            print(f"分析失败: {e}")
                            is_match = False

                        if is_match:
                            total_matched += 1
                            remaining_needed = max(0, remaining_needed - 1)
                            matched_videos.append(video)
                            self.after(0, lambda v=video, x=task_id: self.update_status_message(
                                f"✅ 符合偏好: {v['title'][:50]}...", "#52c41a", x))

                            if task_id in self.task_download_frames:
                                def add_secondary_video(v=video, tid=task_id):
                                    self.task_download_frames[tid].add_video(v)

                                self.after(0, add_secondary_video)
                        else:
                            self.after(0, lambda v=video, x=task_id: self.update_status_message(
                                f"❌ 不符合: {v['title'][:50]}...", "#ff4d4f", x))

                        time.sleep(0.3)

                    if hit_target_mid_videos:
                        break

                    if matched_videos:
                        self.after(0, lambda kw=keyword, count=len(matched_videos), x=task_id: self.update_status_message(
                            f"✅ 关键词 {kw} 找到 {count} 个符合偏好的视频", task_id=x))
                    else:
                        self.after(0, lambda kw=keyword, x=task_id: self.update_status_message(
                            f"⚠️ 关键词 {kw} 没有找到符合偏好的视频", task_id=x))

                    time.sleep(1)
                    _persist_kw(abs_k + 1)

            tail = ordered_keywords[main_kw_resume:]
            run_keyword_loop(tail, start_abs_index=main_kw_resume)

            need_remain = max(0, (target_count or 0) - self._current_download_total_for_task(task_id))
            if need_remain > 0:
                added = self._expand_keyword_map_with_qwen(task_id, user_request, keyword_video_map)
                if added:
                    self.after(0, lambda x=task_id: self.update_status_message(
                        "🤖 已用模型补充新的英文关键词，正在继续搜索...", task_id=x))
                    n_main = len(ordered_keywords)
                    ordered_keywords.extend(added)
                    run_keyword_loop(added, start_abs_index=n_main)
                    _persist_kw(len(ordered_keywords))

            need_remain2 = max(0, (target_count or 0) - self._current_download_total_for_task(task_id))
            if need_remain2 > 0:
                added_fresh = self._expand_keyword_map_with_qwen_fresh(task_id, user_request, keyword_video_map)
                if added_fresh:
                    self.after(0, lambda x=task_id: self.update_status_message(
                        "🤖 仍不足，已用模型生成与旧词不同的新搜索词…", task_id=x))
                    n_base = len(ordered_keywords)
                    ordered_keywords.extend(added_fresh)
                    run_keyword_loop(added_fresh, start_abs_index=n_base)
                    _persist_kw(len(ordered_keywords))

            self.after(0, lambda x=task_id, ta=total_analyzed, tm=total_matched: self.update_status_message(
                f"✅ 本轮搜索推荐完成，分析了 {ta} 个视频，找到 {tm} 个符合偏好的视频", task_id=x))

            hint_msg = {
                "type": "text",
                "content": self.tr("msg_auto_search_hint"),
                "is_user": False,
            }
            self.after(0, lambda m=hint_msg, x=task_id: self.add_msg(m, task_id=x))

            def start_recursive_recommend():
                try:
                    dl_frame = self.task_download_frames.get(task_id)
                    if not dl_frame:
                        return
                    target = self.task_manager.get_task(task_id).get('target_video_count') or target_count
                    current_total = len(getattr(dl_frame, "_video_order", []) or [])
                    if not target or current_total >= target:
                        return
                    self._recursive_sidebar_recommend(task_id, user_request, preferences, target, current_total)
                finally:
                    self.after(2000, lambda x=task_id: self.clear_status_message(x))
                    self.after(0, lambda tid=task_id: self._finish_recommendation_flow(tid))

            threading.Thread(target=start_recursive_recommend, daemon=True).start()

        threading.Thread(target=process_keywords, daemon=True).start()

    def _recursive_sidebar_recommend(self, task_id, user_request, preferences, target_count, current_count,
                                     max_depth=5):
        """多轮抓取：对一次推荐+二次推荐得到的视频，循环抓取其侧边栏推荐作为三次、四次..."""
        self.after(0, lambda x=task_id, tc=target_count, cc=current_count: self.update_status_message(
            f"🔁 开始多轮推荐抓取（目标还差 {max(0, tc - cc)} 个）", task_id=x))

        dl_frame = self.task_download_frames.get(task_id)
        if not dl_frame:
            return

        queue = []
        for vid in getattr(dl_frame, "_video_order", []) or []:
            vobj = (getattr(dl_frame, "_video_payload", {}) or {}).get(vid)
            if vobj:
                queue.append((vobj, 1))

        seen_ids = set(self._collect_all_seen_video_ids_for_task(task_id))
        processed_ids = set()
        total_matched = 0

        while queue and self._current_download_total_for_task(task_id) < target_count:
            TaskThreadManager.wait_recommendation_running(task_id)
            if TaskThreadManager.is_task_cancelled(task_id):
                break
            video, depth = queue.pop(0)
            if depth > max_depth:
                break

            video_id = video.get("video_id")
            if not video_id or video_id in processed_ids:
                continue
            processed_ids.add(video_id)
            current_count = self._current_download_total_for_task(task_id)
            if current_count >= target_count:
                break

            level_name = f"{depth + 2}次推荐" if depth >= 1 else "三次推荐"
            self.after(0, lambda v=video, ln=level_name, x=task_id: self.update_status_message(
                f"📺 {ln}：抓取「{v.get('title', '')[:30]}」的侧边栏推荐...", task_id=x))

            try:
                recommended = get_recommended_videos_from_watch(
                    task_id,
                    video.get("url"),
                    current_video_id=video_id,
                    seen_ids=seen_ids,
                    max_videos=20
                )
            except Exception as e:
                print(f"侧边栏抓取失败: {e}")
                continue

            if not recommended:
                continue

            for rec_v in recommended:
                TaskThreadManager.wait_recommendation_running(task_id)
                if TaskThreadManager.is_task_cancelled(task_id):
                    return
                if self._current_download_total_for_task(task_id) >= target_count:
                    break

                vid2 = rec_v.get("video_id")
                if vid2:
                    self._register_seen_video(task_id, vid2)
                    seen_ids.add(vid2)

                order_set = set(getattr(dl_frame, "_video_order", []) or [])
                if not vid2 or vid2 in order_set:
                    continue

                title = rec_v.get("title", "")
                if not title:
                    continue
                try:
                    _tsk2 = self.task_manager.get_task(task_id)
                    if not video_passes_duration_filter(_tsk2, rec_v):
                        continue
                except Exception:
                    pass

                # 先显示“分析中”
                self.after(0, lambda t=title, x=task_id: self.update_status_message(
                    f"🔍 分析推荐视频是否符合偏好: {t[:40]}...", "#888888", x))

                try:
                    is_match = check_video_match(user_request, preferences, title)
                except Exception as e:
                    print(f"多轮分析失败: {e}")
                    is_match = False

                # 新增：像二次推荐一样，实时红绿反馈
                if is_match:
                    self.after(0, lambda t=title, x=task_id: self.update_status_message(
                        f"✅ 符合偏好: {t[:40]}...", "#52c41a", x))
                else:
                    self.after(0, lambda t=title, x=task_id: self.update_status_message(
                        f"❌ 不符合偏好: {t[:40]}...", "#ff4d4f", x))
                    continue

                total_matched += 1

                def add_rec(v=rec_v):
                    if task_id in self.task_download_frames:
                        self.task_download_frames[task_id].add_video(v)

                self.after(0, add_rec)

                # 把符合条件的视频继续放入队列，用于下一层（四次、五次...）
                queue.append((rec_v, depth + 1))

        _cc_done = self._current_download_total_for_task(task_id)
        self.after(0, lambda x=task_id, tm=total_matched, cc=_cc_done, tc=target_count: self.update_status_message(
            f"✅ 多轮推荐完成，额外找到 {tm} 个符合偏好的视频（当前总数 {cc}/{tc}）", task_id=x))

    def _has_pending_video_list_confirm(self):
        tid = self.current_task_id
        if not tid:
            return False
        state = self.get_task_state(tid)
        for msg in state.get("current_messages") or []:
            if msg.get("type") == "video_list" and not msg.get("selection_confirmed"):
                return True
        return False

    def _merge_preference_patch(self, task, patch):
        base = dict(task.get("cached_preferences") or {})
        if not isinstance(patch, dict):
            return base
        for key in ["scene", "subject", "camera", "style", "avoid"]:
            arr = patch.get(key)
            if not isinstance(arr, list):
                continue
            cur = list(base.get(key) or [])
            for x in arr:
                s = str(x).strip()
                if s and s not in cur:
                    cur.append(s)
            base[key] = cur
        return base

    def _handle_followup_instruction(self, txt):
        tid = self.current_task_id
        self.entry.delete(0, "end")
        self.send_btn.configure(fg_color="#8cb5ff", hover_color="#4a7acc")
        msg = {"type": "text", "content": txt, "is_user": True}
        self.current_messages.append(msg)
        self.task_manager.save_task_messages(tid, self.current_messages)
        state = self.get_task_state(tid)
        state["current_messages"] = self.current_messages
        msg_area = self.get_current_msg_area()
        if msg_area:
            MessageBubble(msg_area, msg, self.task_manager, tid, True, app=self).pack(fill="x", pady=2)
            msg_area._parent_canvas.yview_moveto(1)
        threading.Thread(target=self._followup_worker, args=(txt, tid), daemon=True).start()

    def _followup_worker(self, txt, tid):
        task = self.task_manager.get_task(tid)
        if not task:
            return
        st = self.get_task_state(tid)
        snapshot = {
            "task_name": task.get("name"),
            "user_request": st.get("current_user_request")
            or (task.get("ui_state") or {}).get("current_user_request"),
            "cached_preferences": task.get("cached_preferences") or {},
            "target_video_count": task.get("target_video_count"),
            "video_duration_min_sec": task.get("video_duration_min_sec"),
            "video_duration_max_sec": task.get("video_duration_max_sec"),
            "has_unfinished_extraction": task_has_incomplete_extraction(task),
        }
        try:
            data = analyze_followup_instruction(txt, snapshot)
        except Exception as e:
            print(f"followup 分析失败: {e}")
            data = {"intent": "none", "reply_zh": "分析失败，请稍后再试。"}
        self.after(0, lambda: self._apply_followup_result(tid, data))

    def _deselected_video_dicts_for_task(self, task_id):
        """从聊天记录中还原带 keyword 等字段的取消勾选视频，供继续搜索推荐逻辑使用。"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return []
        des_ids = set(task.get("deselected_videos") or [])
        if not des_ids:
            return []
        msgs = []
        try:
            st = self.get_task_state(task_id)
            msgs = list(st.get("current_messages") or [])
        except Exception:
            msgs = []
        if not msgs:
            try:
                msgs = self.task_manager.load_task_messages(task_id)
            except Exception:
                msgs = []
        if task_id == getattr(self, "current_task_id", None) and getattr(self, "current_messages", None):
            msgs = list(self.current_messages or msgs)
        out = []
        seen = set()
        for msg in msgs:
            if msg.get("type") != "video_list":
                continue
            for v in msg.get("videos") or []:
                vid = (v or {}).get("video_id")
                if vid and vid in des_ids and vid not in seen:
                    seen.add(vid)
                    out.append(v)
        return out

    def _followup_resume_secondary_search_if_below_target(self, tid, target_count):
        """
        对话里修改目标总数后：
        - 若新目标高于当前列表条数：优先用「上次推荐结束时的缓存」接着搜索推荐（同一关键词与已见集合）；
        - 若缩小目标且列表已超过新目标：不处理（由 cur >= tc 直接返回）。
        """
        task = self.task_manager.get_task(tid)
        if not task or not task.get("ever_confirmed_to_download"):
            return
        try:
            tc = int(target_count) if target_count is not None else None
        except (TypeError, ValueError):
            tc = None
        if tc is None or tc <= 0:
            return
        cur = self._current_download_total_for_task(tid)
        if cur >= tc:
            return

        chk = task.get("recommendation_resume_payload")
        if isinstance(chk, dict) and chk.get("keyword_video_map") is not None:
            payload = dict(chk)
        else:
            payload = self._build_recommendation_resume_payload_dict(tid)
        if payload.get("keyword_video_map") is None:
            payload["keyword_video_map"] = {}
        payload["target_count"] = tc
        payload["current_count"] = cur
        print(
            f"[followup] 目标 {tc} 当前列表 {cur}，从暂停缓存继续搜索推荐（关键词进度 {payload.get('secondary_kw_cursor')}/{len(payload.get('secondary_kw_order') or [])}）",
            flush=True,
        )
        try:
            self.task_manager.update_task_info(tid, {"recommendation_resume_payload": payload})
        except Exception:
            pass
        threading.Thread(
            target=lambda: self._resume_recommendation_worker(tid, payload), daemon=True
        ).start()

    def _apply_followup_result(self, tid, data):
        task = self.task_manager.get_task(tid)
        if not task:
            return
        intent = (data.get("intent") or "none").lower().strip()
        try:
            gap = float(data.get("off_topic_gap") or 0)
        except (TypeError, ValueError):
            gap = 0.0
        if (intent == "off_topic" and gap >= 0.45) or gap >= 0.72:
            self.add_msg(
                {
                    "type": "text",
                    "content": data.get("reply_zh")
                    or "当前需求与任务主题差距较大，建议新建任务来下载这一类素材。",
                    "is_user": False,
                }
            )
            return

        updates = {}
        followup_new_target = None  # intent==count 时解析后的新目标，用于自动续推
        if intent == "duration":
            lo = data.get("duration_min_sec")
            hi = data.get("duration_max_sec")
            if lo is not None:
                try:
                    updates["video_duration_min_sec"] = int(lo)
                except (TypeError, ValueError):
                    pass
            if hi is not None:
                try:
                    updates["video_duration_max_sec"] = int(hi)
                except (TypeError, ValueError):
                    pass

        if intent == "count":
            tc = task.get("target_video_count")
            if tc is None:
                tc = 20
            try:
                tc = int(tc)
            except (TypeError, ValueError):
                tc = 20
            abs_v = data.get("target_absolute")
            delta = data.get("target_delta")
            try:
                if abs_v is not None:
                    tc = max(1, int(abs_v))
                elif delta is not None:
                    tc = max(1, tc + int(delta))
            except (TypeError, ValueError):
                pass
            updates["target_video_count"] = tc
            st = self.get_task_state(tid)
            st["target_video_count"] = tc
            followup_new_target = tc

        if intent == "preference":
            merged = self._merge_preference_patch(task, data.get("preference_patch") or {})
            updates["cached_preferences"] = merged

        if updates:
            try:
                self.task_manager.update_task_info(tid, updates)
            except Exception as e:
                print(f"followup 写 info 失败: {e}")

        self.add_msg({"type": "text", "content": data.get("reply_zh") or "已记录。", "is_user": False})
        try:
            self.persist_task_ui_state(tid)
        except Exception:
            pass

        if intent == "count" and followup_new_target is not None:
            try:
                self._followup_resume_secondary_search_if_below_target(tid, followup_new_target)
            except Exception as e:
                print(f"followup 自动续推异常: {e}", flush=True)
                traceback.print_exc()

    def send(self):
        txt = self.entry.get().strip()
        if not txt or not self.current_task_id:
            return
        tid = self.current_task_id
        state = self.get_task_state(tid)
        # 暂停/继续 搜索·推荐·下载：允许在「正在搜索」时发送
        if not state.get("waiting_for_video_count") and self._try_chat_control_command(txt):
            self.entry.delete(0, "end")
            state["input_draft"] = ""
            self.send_btn.configure(fg_color="#8cb5ff", hover_color="#4a7acc")
            return
        if state.get("is_searching"):
            return
        if state.get("waiting_for_video_count"):
            self.process_video_count_input(txt, tid)
            return

        task = self.task_manager.get_task(tid)
        ever_confirmed = task and task.get("ever_confirmed_to_download")
        if ever_confirmed:
            self._handle_followup_instruction(txt)
            return

        if self._has_pending_video_list_confirm():
            self.add_msg(
                {
                    "type": "text",
                    "content": self.tr("msg_confirm_first"),
                    "is_user": False,
                }
            )
            self.entry.delete(0, "end")
            return

        self.entry.delete(0, "end")
        state["input_draft"] = ""
        self.send_btn.configure(fg_color="#8cb5ff", hover_color="#4a7acc")
        msg = {"type": "text", "content": txt, "is_user": True}
        self.current_messages.append(msg)
        self.task_manager.save_task_messages(self.current_task_id, self.current_messages)
        state = self.get_task_state(self.current_task_id)
        state['current_messages'] = self.current_messages
        state['current_user_request'] = txt
        msg_area = self.get_current_msg_area()
        if msg_area:
            MessageBubble(msg_area, msg, self.task_manager, self.current_task_id, True, app=self).pack(fill="x", pady=2)
            msg_area._parent_canvas.yview_moveto(1)
        self.current_user_request = txt
        count_question_msg = {"type": "text", "content": self.tr("ask_video_count"), "is_user": False}
        self.add_msg(count_question_msg)
        state["waiting_for_video_count"] = True
        state["pending_search_request"] = txt

    def process_video_count_input(self, user_input, task_id=None):
        tid = task_id if task_id is not None else self.current_task_id
        state = self.get_task_state(tid)
        state["waiting_for_video_count"] = False
        self.entry.delete(0, "end")
        state["input_draft"] = ""
        self.send_btn.configure(fg_color="#8cb5ff", hover_color="#4a7acc")
        try:
            match = re.search(r'(\d+)', user_input)
            if match:
                target_count = int(match.group(1))
            else:
                target_count = None
        except:
            target_count = None
        user_msg = {"type": "text", "content": user_input, "is_user": True}
        self.current_messages.append(user_msg)
        self.task_manager.save_task_messages(self.current_task_id, self.current_messages)
        state = self.get_task_state(self.current_task_id)
        state['current_messages'] = self.current_messages
        state['target_video_count'] = target_count
        msg_area = self.get_current_msg_area()
        if msg_area:
            MessageBubble(msg_area, user_msg, self.task_manager, self.current_task_id, True, app=self).pack(fill="x", pady=2)
            msg_area._parent_canvas.yview_moveto(1)
        if target_count is not None:
            self.task_manager.update_task_info(self.current_task_id, {'target_video_count': target_count})
            self.add_msg(
                {"type": "text", "content": self.tr("count_recorded", count=target_count), "is_user": False})
        else:
            self.add_msg({"type": "text", "content": self.tr("count_recorded_no_n"), "is_user": False})
        pending = state.get("pending_search_request")
        state["pending_search_request"] = None
        threading.Thread(
            target=self._run_post_count_flow,
            args=(pending, tid),
            daemon=True,
        ).start()

    def start_search(self, txt, task_id):
        self._set_task_searching(task_id, True)
        self._active_search_task_id = task_id
        self.after(0, self._refresh_search_hint_placeholder)
        threading.Thread(target=self.search_worker, args=(txt, task_id), daemon=True).start()

    def search_worker(self, prompt, task_id):
        tid = task_id
        if not tid:
            self.is_searching = False
            return
        try:
            # 与二次推荐一致：首轮关键词搜索通过聊天「暂停搜索/继续搜索」协作暂停
            TaskThreadManager.ensure_recommendation_event(tid)
            TaskThreadManager.resume_recommendation(tid)
            if TaskThreadManager.is_task_cancelled(tid):
                return
            state = self.get_task_state(tid)
            kws = state.get('generated_keywords', [])
            if not kws:
                self.after(0, lambda x=tid: self.update_status_message("🤖 正在分析搜索需求...", task_id=x))
                info = _extract_keywords_and_preferences(prompt)
                kws = info.get("keywords", []) or generate_search_keywords(prompt)
                state["generated_keywords"] = kws
            self.after(0, lambda x=tid, n=len(kws): self.update_status_message(
                f"🔍 已生成 {n} 个搜索关键词，准备开始搜索...", task_id=x))
            self.after(0, lambda x=tid: self.update_status_message("🌐 正在使用浏览器...", task_id=x))
            seen_video_ids = set()
            try:
                task = self.task_manager.get_task(tid)
                cached = (task.get("ui_state", {}) or {}).get("seen_video_ids") if task else None
                if cached:
                    seen_video_ids = set(cached)
            except Exception:
                pass
            seen_video_ids |= self._collect_all_seen_video_ids_for_task(tid)
            all_videos = []
            keyword_video_map = {}
            total_keywords = len(kws)
            for idx, k in enumerate(kws, 1):
                TaskThreadManager.wait_recommendation_running(tid)
                if TaskThreadManager.is_task_cancelled(tid):
                    return
                self.after(0, lambda i=idx, t=total_keywords, kw=k, x=tid: self.update_status_message(
                    f"🔎 正在搜索关键词 ({i}/{t}): {kw}", task_id=x))
                vs = get_videos_from_search(tid, k, 2, seen_video_ids)
                all_videos.extend(vs)
                keyword_video_map[k] = vs
                state["seen_video_ids"] = set(seen_video_ids)
                state["keyword_video_map"] = dict(keyword_video_map)
                if all_videos:
                    self.after(0, lambda count=len(all_videos), x=tid: self.update_status_message(
                        f"📹 已找到 {count} 个视频，继续搜索...", task_id=x))
                if len(all_videos) >= 20:
                    self.after(0, lambda x=tid: self.update_status_message(
                        "✅ 已达到目标数量(20个)，停止搜索", task_id=x))
                    break
            if not all_videos:
                self.after(0, lambda x=tid: self.update_status_message(
                    "⚠️ 未找到任何视频，请尝试其他关键词", task_id=x))
                self.after(2000, lambda x=tid: self.clear_status_message(task_id=x))
                no_result = {"type": "text", "content": self.tr("msg_no_result"), "is_user": False}
                self.after(0, lambda m=no_result, x=tid: self.add_msg(m, task_id=x))
                return
            batch_ids = [v["video_id"] for v in all_videos if v.get("video_id")]
            self.task_manager.merge_seen_video_ids_batch(tid, batch_ids)
            self.after(0, lambda x=tid, n=len(all_videos): self.update_status_message(
                f"📸 正在下载 {n} 个视频的缩略图...", task_id=x))
            for idx, v in enumerate(all_videos):
                TaskThreadManager.wait_recommendation_running(tid)
                if TaskThreadManager.is_task_cancelled(tid):
                    return
                self.after(0, lambda i=idx, total=len(all_videos), title=v['title'], x=tid: self.update_status_message(
                    f"📸 下载缩略图 ({i + 1}/{total}): {title[:30]}...", task_id=x))
                img = download_thumbnail(v['video_id'])
                self.task_manager.save_thumbnail(tid, v['video_id'], img)
                v['selected'] = True
            self.after(0, lambda x=tid, n=len(all_videos): self.update_status_message(
                f"✅ 搜索完成！共找到 {n} 个视频", task_id=x))
            self.after(1000, lambda x=tid: self.clear_status_message(task_id=x))
            hint_msg = {"type": "text", "content": self.tr("msg_recommend_hint"), "is_user": False}
            self.after(0, lambda m=hint_msg, x=tid: self.add_msg(m, task_id=x))
            time.sleep(0.5)
            video_msg = {"type": "video_list", "videos": all_videos, "is_user": False}
            self.after(0, lambda m=video_msg, x=tid: self.add_msg(m, is_video_list=True, task_id=x))
            self.task_manager.update_task_info(tid, {"videos": all_videos})
            st = self.get_task_state(tid)
            st["keyword_video_map"] = keyword_video_map
            st["seen_video_ids"] = set(seen_video_ids)
            self.persist_task_ui_state(tid)
            if tid == self.current_task_id:
                self.keyword_video_map = dict(keyword_video_map)
                self.seen_video_ids = set(seen_video_ids)
        except Exception as e:
            print(f"搜索错误: {e}")
            self.after(0, lambda x=tid, err=str(e): self.update_status_message(
                f"❌ 搜索出错: {err}", task_id=x))
            self.after(3000, lambda x=tid: self.clear_status_message(task_id=x))
            error_msg = {"type": "text", "content": self.tr("msg_search_error", err=str(e)), "is_user": False}
            self.after(0, lambda m=error_msg, x=tid: self.add_msg(m, task_id=x))
        finally:
            self._set_task_searching(tid, False)
            self.after(0, self._refresh_search_hint_placeholder)
            if getattr(self, "_active_search_task_id", None) == tid:
                self._active_search_task_id = None
            self.after(0, self._refresh_session_resume_row_visibility)

    def add_msg(self, m, is_video_list=False, task_id=None):
        tid = task_id if task_id is not None else self.current_task_id
        if not tid:
            return
        if tid == self.current_task_id:
            if self.get_current_status_frame() is not None:
                self.clear_status_message(tid)
        else:
            st = self.get_task_state(tid)
            if st.get("status_frame") is not None:
                self.clear_status_message(tid)
        state = self.get_task_state(tid)
        messages = list(state.get("current_messages") or [])
        messages.append(m)
        state["current_messages"] = messages
        self.task_manager.save_task_messages(tid, messages)
        if tid == self.current_task_id:
            self.current_messages = messages
        msg_area = self.get_msg_area_for_task(tid)
        if not msg_area:
            return
        user_req = state.get("current_user_request")
        if is_video_list:
            bubble = MessageBubble(msg_area, m, self.task_manager, tid, is_user=False,
                                   on_selection_change=self.on_video_selection_change,
                                   on_confirm=self.on_confirm_analysis,
                                   user_request=user_req, app=self)
            bubble.pack(fill="x", pady=2)
        else:
            MessageBubble(msg_area, m, self.task_manager, tid, is_user=False,
                          on_selection_change=self.on_video_selection_change, app=self).pack(fill="x", pady=2)
        msg_area._parent_canvas.yview_moveto(1)
        if tid == self.current_task_id:
            self.update_stats_display()
        self.update_idletasks()


if __name__ == "__main__":
    sys.excepthook = _debug_excepthook
    _install_thread_excepthook()
    app = EasyDatasetApp()
    app.mainloop()