import logging
from inspect import getframeinfo, stack
from typing import Optional, Union

_default_logger = logging.getLogger("graspkit")

_name_to_level = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
_has_logged = set()


def log_once(
    message: str,
    level: Union[str, int] = logging.INFO,
    logger: Optional[logging.Logger] = None,
):
    """
    Log a message only once (based on the message content and the source code location).

    Args:
        message (str): message to log
        level (str or int): log level, could be "critical"/"error"/"warning"/"info"/"debug" or int (default: "info")
    """
    call_stack = stack()
    actual_frame = call_stack[1]
    for frame_info in call_stack[1:]:
        if not frame_info.filename.startswith("<"):  # found a valid caller frame instead of <@beartype(...)>
            actual_frame = frame_info
            break
    frame_info = getframeinfo(actual_frame.frame)
    key = (frame_info.filename, frame_info.lineno, message)
    if key not in _has_logged:
        if isinstance(level, str):
            level = _name_to_level[level.lower()]
        if logger is None:
            logger = _default_logger
        logger.log(level, message)
        _has_logged.add(key)
