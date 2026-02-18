# Copyright 2026 German Aerospace Center (DLR)
# Institute Systems Engineering for Future Mobility (SE)
#
# Contributors:
#   - Thies de Graaff <thies.degraaff@dlr.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging


CDR_LOG_FORMAT = '%(asctime)s.%(msecs)03d | [%(levelname)s] %(message)s'
CDR_LOG_DATEFORMAT = '%Y/%m/%d %H:%M:%S'


class CDRLogger(logging.Logger):
    """
    This customized Logger class overrides `makeRecord`, making it possible to set the timestamp of a log message
    explicitly. See `makeRecord` for more details.
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """
        This specialized implementation watches for the keyword "timestamp" in the optional `extra` dictionary.
        If both are present in the call's arguments, then the timestamp of the created `LogRecord` will be set to
        the specified timestamp, which has to be given in seconds since epoch (int or float, i.e. decimal precision
        is possible).
        """
        timestamp = None
        if extra is not None and 'timestamp' in extra:
            timestamp = extra['timestamp']
            extra = None if len(extra) == 1 else {k: v for k, v in extra.items() if k != 'timestamp'}

        rv = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)

        if timestamp is not None:
            assert isinstance(timestamp, (float, int))
            rv.created = timestamp
            rv.msecs = (timestamp - int(timestamp)) * 1000
            rv.relativeCreated = (timestamp - logging._startTime) * 1000  # type: ignore

        return rv


class CDRDefaultFormatter(logging.Formatter):
    """
    A default formatter class for the CDR to use with Python's logging module. It uses custom formats for the log
    messages and the datetime representation.
    """

    def __init__(self):
        super().__init__(CDR_LOG_FORMAT, CDR_LOG_DATEFORMAT)


ANSI_COLOR_CODES = {
    'reset': '\033[0m',
    'red': '\033[31m',
    'yellow': '\033[33m'
}

def _inject_color(fmt: str, color: str) -> str:
    color = color.lower()
    return ANSI_COLOR_CODES[color] + fmt + ANSI_COLOR_CODES['reset']


class CDRColoredFormatter(CDRDefaultFormatter):
    """
    A formatter class for the CDR that extends from `CDRDefaultFormatter` by colorizing log messages based on their
    log level.
    """

    def __init__(self):
        super().__init__()

        self.color_formatters = {
            logging.WARNING: logging.Formatter(_inject_color(CDR_LOG_FORMAT, 'yellow'), CDR_LOG_DATEFORMAT),
            logging.ERROR: logging.Formatter(_inject_color(CDR_LOG_FORMAT, 'red'), CDR_LOG_DATEFORMAT),
            logging.CRITICAL: logging.Formatter(_inject_color(CDR_LOG_FORMAT, 'red'), CDR_LOG_DATEFORMAT)
        }

        # Required for Windows to enable colors in terminal
        import os
        if os.name == 'nt':
            os.system('color')  # pragma: no cover

    def format(self, record) -> str:
        return self.color_formatters.get(record.levelno, super()).format(record)
