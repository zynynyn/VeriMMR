import logging
import os
from rich.console import Console
from rich.logging import RichHandler
from typing import Literal, Optional
from datetime import datetime

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
logging_dict = _LOG_LEVELS

_LOGGING_INITIALIZED = False
_LOGFILE_PATH: Optional[str] = None


def _level_from_str(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return _LOG_LEVELS.get(str(level).lower(), logging.INFO)


def get_logger(
    name: str,
    level: Literal["debug", "info", "warn", "error"] | str = "info",
    enable_rich_tracebacks: bool = True,
    log_file: Optional[str] = None,
):
    global _LOGGING_INITIALIZED, _LOGFILE_PATH

    lvl = _level_from_str(level)
    base = logging.getLogger("UltraRAG")

    if not _LOGGING_INITIALIZED:
        # --- [关键修改 1] 彻底清空所有 Logger 的 Handler ---
        # 不仅仅是 UltraRAG，我们要让整个 Python 环境的控制台都安静
        for logger_name in logging.root.manager.loggerDict.keys():
            logging.getLogger(logger_name).handlers = []
        logging.root.handlers = [] 
        
        os.makedirs("logs", exist_ok=True)
        if log_file:
            _LOGFILE_PATH = log_file
        else:
            ts = os.environ.get("ULTRARAG_LOG_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
            _LOGFILE_PATH = os.path.join("logs", f"{ts}.log")

        # --- [关键修改 2] 只创建 FileHandler，不创建任何 Console Handler ---
        file_handler = logging.FileHandler(_LOGFILE_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(lvl)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                datefmt="%m/%d/%y %H:%M:%S",
            )
        )

        # 把 root logger 指向文件，这样所有库的日志都会进文件而不会出控制台
        root_logger = logging.getLogger()
        root_logger.setLevel(lvl)
        root_logger.addHandler(file_handler)

        # UltraRAG 的 base 也只加 file_handler
        base.setLevel(lvl)
        base.addHandler(file_handler)
        
        # --- [关键修改 3] 针对 MiniCPM 等库的特殊静默 ---
        # 强制这些容易喷日志的库只报 ERROR，并且关闭向上传递
        quiet_list = ["transformers", "huggingface_hub", "vllm", "sentence_transformers", "urllib3"]
        for q in quiet_list:
            l = logging.getLogger(q)
            l.setLevel(logging.ERROR)
            l.propagate = False  # 这一行很重要：切断它通往 root 的路径
            l.addHandler(file_handler) # 如果想看它们的错，进文件看
        # ---------------------------------------------------------
        # os.makedirs("logs", exist_ok=True)

        # if log_file:
        #     _LOGFILE_PATH = log_file
        # else:
        #     ts = os.environ.get("ULTRARAG_LOG_TS") or datetime.now().strftime(
        #         "%Y%m%d_%H%M%S"
        #     )
        #     _LOGFILE_PATH = os.path.join("logs", f"{ts}.log")

        # rich_handler = RichHandler(
        #     console=Console(stderr=True),
        #     rich_tracebacks=enable_rich_tracebacks,
        #     omit_repeated_times=False,
        # )
        # rich_handler.setLevel(lvl)
        # rich_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

        # file_handler = logging.FileHandler(_LOGFILE_PATH, mode="a", encoding="utf-8")
        # file_handler.setLevel(lvl)
        # file_handler.setFormatter(
        #     logging.Formatter(
        #         "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        #         datefmt="%m/%d/%y %H:%M:%S",
        #     )
        # )

        # base.setLevel(lvl)
        # base.addHandler(rich_handler)
        # base.addHandler(file_handler)
        base.propagate = False

        _LOGGING_INITIALIZED = True

    if lvl < base.level or lvl > base.level:
        base.setLevel(lvl)
        for h in base.handlers:
            h.setLevel(lvl)

    return base if name == "UltraRAG" else base.getChild(name)
