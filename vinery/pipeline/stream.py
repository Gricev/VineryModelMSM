"""Чтение потокового видео (файл или RTSP/USB-камера) с единой временной шкалой."""
from __future__ import annotations

import time
from typing import Iterator, Optional

from .models import Frame

try:
    import cv2
except ImportError:  # позволяет импортировать модуль без установленного opencv
    cv2 = None


class VideoStream:
    """Обёртка над cv2.VideoCapture, выдающая Frame с временной меткой.

    source: путь к файлу, RTSP-URL ('rtsp://...') или индекс камеры (0, 1, ...).

    Временная метка t (секунды от старта прохода):
      - realtime=False (файл): t = index / fps — детерминированно по номеру кадра;
      - realtime=True (живая камера): t = monotonic() - start_monotonic, то есть
        реальное время захвата. Передайте ОБЩИЙ start_monotonic обеим камерам,
        чтобы их часы совпадали — это основа честной синхронизации потоков.
    """

    def __init__(self, source: str | int, camera: str,
                 realtime: bool = False, start_monotonic: Optional[float] = None):
        if cv2 is None:
            raise RuntimeError("Установите opencv-python: pip install opencv-python")
        self.source = source
        self.camera = camera
        self.realtime = realtime
        self.start_monotonic = start_monotonic
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть поток: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self.cap.read()
            if not ok:
                break
            if self.realtime:
                now = time.monotonic()
                if self.start_monotonic is None:
                    self.start_monotonic = now
                t = now - self.start_monotonic
            else:
                t = index / self.fps
            yield Frame(camera=self.camera, t=t, index=index, image=image)
            index += 1

    def release(self) -> None:
        self.cap.release()