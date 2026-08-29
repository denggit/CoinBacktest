from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage

from .config import EmailConfig


class AsyncEmailSender:
    def __init__(self, config: EmailConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._tasks: set[asyncio.Task[bool]] = set()

    def submit(self, subject: str, body: str) -> None:
        if not self.config.enabled:
            return
        if not self.config.configured:
            self.logger.warning("[email] enabled but SMTP configuration is incomplete")
            return
        task = asyncio.create_task(asyncio.to_thread(self._send_sync, subject, body))
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[bool]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except Exception as exc:  # defensive: email must never terminate monitor
            self.logger.warning("[email] send failed: %s", exc)

    def _send_sync(self, subject: str, body: str) -> bool:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.sender
        message["To"] = ", ".join(self.config.recipients)
        message.set_content(body)
        try:
            if self.config.use_ssl:
                with smtplib.SMTP_SSL(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout_seconds,
                    context=ssl.create_default_context(),
                ) as client:
                    client.login(self.config.username, self.config.password)
                    client.send_message(message)
            else:
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=self.config.timeout_seconds) as client:
                    client.ehlo()
                    client.starttls(context=ssl.create_default_context())
                    client.ehlo()
                    client.login(self.config.username, self.config.password)
                    client.send_message(message)
            self.logger.info("[email] sent subject=%s", subject)
            return True
        except Exception as exc:
            self.logger.warning("[email] send failed: %s", exc)
            return False

    async def drain(self, timeout: float = 0.5) -> None:
        if not self._tasks:
            return
        done, _ = await asyncio.wait(tuple(self._tasks), timeout=timeout)
        for task in done:
            self._tasks.discard(task)
