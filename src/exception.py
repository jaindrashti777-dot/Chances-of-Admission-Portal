"""
Custom exception handling module for the ML pipeline.

Provides structured error messages including filename and line number
for easier debugging of production issues.
"""

from __future__ import annotations

import sys
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


def error_message_detail(error: Exception | str, error_detail: Any) -> str:
    """
    Extract detailed error information from the traceback.

    Parameters
    ----------
    error : Exception or str
        The original error.
    error_detail : sys
        The sys module to extract traceback details.

    Returns
    -------
    str
        Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"with error message: [{str(error)}]"
        )
    else:
        error_message = str(error)

    return error_message


class CustomException(Exception):
    """
    Custom exception class for the project.

    Captures the exact script name and line number where the error occurred
    and logs it automatically.
    """

    def __init__(self, error_message: Exception | str, error_detail: Any = sys):
        """
        Initialize the custom exception.

        Parameters
        ----------
        error_message : Exception or str
            The error message or exception object.
        error_detail : Any
            sys module to extract traceback.
        """
        super().__init__(str(error_message))
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
