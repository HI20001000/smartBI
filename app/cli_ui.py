# cli_ui.py
import os
import sys
import shutil
import platform
from datetime import datetime


def _supports_color() -> bool:
    """Basic ANSI color support detection (fallback when rich is unavailable)."""
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False

    # Windows terminals usually support ANSI nowadays, but keep it conservative.
    term = os.getenv("TERM", "")
    colorterm = os.getenv("COLORTERM", "")
    if term or colorterm:
        return True
    return os.name != "nt"


def _clear_screen() -> None:
    # Prefer ANSI clear to avoid flicker; fallback to cls/clear if needed.
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    else:
        os.system("cls" if os.name == "nt" else "clear")


def _truncate_middle(s: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    head = (max_len - 1) // 2
    tail = max_len - 1 - head
    return f"{s[:head]}…{s[-tail:]}"


def _wrap_text(s: str, width: int) -> list[str]:
    if width <= 0:
        return [s]
    out, line = [], ""
    for token in s.split(" "):
        if not line:
            line = token
            continue
        if len(line) + 1 + len(token) <= width:
            line += " " + token
        else:
            out.append(line)
            line = token
    if line:
        out.append(line)

    # If no spaces (e.g., URL), hard wrap:
    hard = []
    for ln in out:
        if len(ln) <= width:
            hard.append(ln)
        else:
            for i in range(0, len(ln), width):
                hard.append(ln[i : i + width])
    return hard or [""]


def print_startup_ui(
    model: str,
    base_url: str,
    *,
    app_name: str = "SmartBI Chat CLI",
    framework: str = "LangChain",
    version: str | None = None,
    clear_screen: bool = True,
    show_system: bool = False,
    prefer_rich: bool = True,
) -> None:
    """Pretty startup UI. Uses rich if available; otherwise falls back to ANSI."""
    if clear_screen:
        _clear_screen()

    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    term_w = shutil.get_terminal_size((88, 20)).columns
    w = min(max(70, min(term_w, 100)), 100)  # stable, not too narrow/wide

    # 1) Rich path (best)
    if prefer_rich:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.align import Align
            from rich import box

            console = Console()

            title = Text.assemble(
                ("🤖 ", "bold cyan"),
                (app_name, "bold cyan"),
                ("  ", ""),
                (f"({framework})", "dim"),
            )
            if version:
                title.append(f"  v{version}", style="dim")

            table = Table.grid(padding=(0, 1))
            table.add_column(justify="right", style="bold magenta", width=10)
            table.add_column(style="white")

            table.add_row("Model", f"[bold green]{model}[/bold green]")
            table.add_row("Base URL", f"[bold blue]{base_url}[/bold blue]")
            table.add_row("Time", f"[dim]{now}[/dim]")

            if show_system:
                sys_info = f"{platform.system()} {platform.release()} · Python {platform.python_version()}"
                table.add_row("System", f"[dim]{sys_info}[/dim]")

            help_lines = Text.assemble(
                ("• ", "bold"),
                ("Type a message and press Enter\n", ""),
                ("• ", "bold"),
                ("Type ", ""),
                ("exit", "bold red"),
                (" or ", ""),
                ("quit", "bold red"),
                (" to stop\n", ""),
                ("• ", "bold"),
                ("Tip: set ", "dim"),
                ("NO_COLOR=0", "dim bold"),
                (" to disable colors", "dim"),
            )

            panel = Panel(
                Align.center(
                    Text.assemble(title, "\n\n", table, "\n", help_lines),
                    vertical="middle",
                ),
                box=box.ROUNDED,
                border_style="cyan",
                padding=(1, 2),
                width=min(w, 96),
            )
            console.print(panel)
            return
        except Exception:
            # fall through to ANSI
            pass

    # 2) Fallback: ANSI
    use_color = _supports_color()

    def C(s: str, code: str) -> str:
        if not use_color:
            return s
        return f"\033[{code}m{s}\033[0m"

    inner = w - 2

    def pad_line(s: str) -> str:
        return "│" + s.ljust(inner)[:inner] + "│"

    top = "┌" + "─" * inner + "┐"
    mid = "├" + "─" * inner + "┤"
    bot = "└" + "─" * inner + "┘"

    # Centered header
    header = f"{C('🤖', '1;36')} {C(app_name, '1;36')} {C(f'({framework})', '2')}"
    if version:
        header += f" {C('v' + version, '2')}"
    header = header.strip()
    header = header.center(inner)

    # Key-value formatter
    def kv(label: str, value: str, value_color: str) -> list[str]:
        left = f"{C(label, '1;35'):<10}: "
        max_v = inner - len(left)
        wrapped = _wrap_text(value, max_v)
        out = []
        for i, ln in enumerate(wrapped):
            if i == 0:
                out.append(left + C(ln, value_color))
            else:
                out.append(" " * len(left) + C(ln, value_color))
        return out

    lines = [
        top,
        pad_line(header),
        pad_line(""),
    ]

    for ln in kv("Model", model, "1;32"):
        lines.append(pad_line(ln))
    for ln in kv("Base URL", base_url, "1;34"):
        lines.append(pad_line(ln))
    for ln in kv("Time", now, "2"):
        lines.append(pad_line(ln))

    if show_system:
        sys_info = f"{platform.system()} {platform.release()} · Python {platform.python_version()}"
        for ln in kv("System", sys_info, "2"):
            lines.append(pad_line(ln))

    lines += [
        mid,
        pad_line(f"{C('•', '1;37')} Type a message and press Enter"),
        pad_line(f"{C('•', '1;37')} Type {C('exit', '1;31')} or {C('quit', '1;31')} to stop"),
        pad_line(C("• Tip: set NO_COLOR=1 to disable colors", "2")),
        bot,
    ]
    print("\n".join(lines))
