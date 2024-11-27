from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console
from datetime import datetime
import traceback

# Initialize Rich console
console = Console()

class AutopilotError(Exception):
    """Base class for autopilot-related errors"""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ModelError(AutopilotError):
    """Error related to model loading or inference"""
    pass

class CameraError(AutopilotError):
    """Error related to camera operations"""
    pass

class ProcessingError(AutopilotError):
    """Error related to frame processing"""
    pass

def create_error_panel(error, title="🚨 Error Details"):
    """Create a rich panel for error display"""
    error_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return Panel(
        f"[red bold]Error at {error_time}[/red bold]\n"
        f"[red]{str(error)}[/red]\n\n"
        f"[yellow]Stack Trace:[/yellow]\n"
        f"{Syntax(traceback.format_exc(), 'python', background_color='default')}",
        title=title,
        border_style="red"
    )

def handle_camera_error(error):
    """Handle camera-related errors"""
    panel = Panel(
        f"[red bold]Camera Error[/red bold]\n"
        f"[red]{str(error)}[/red]\n\n"
        f"[yellow]Stack Trace:[/yellow]\n"
        f"{Syntax(traceback.format_exc(), 'python', background_color='default')}",
        title="🚨 Camera Initialization Failed",
        border_style="red"
    )
    console.print(panel)

def handle_runtime_error(error):
    """Handle runtime errors"""
    panel = Panel(
        f"[red bold]Runtime Error[/red bold]\n"
        f"[red]{str(error)}[/red]\n\n"
        f"[yellow]Stack Trace:[/yellow]\n"
        f"{Syntax(traceback.format_exc(), 'python', background_color='default')}",
        title="🚨 Main Loop Error",
        border_style="red"
    )
    console.print(panel)

def handle_fatal_error(error):
    """Handle fatal application errors"""
    panel = Panel(
        f"[red bold]Fatal Error[/red bold]\n"
        f"[red]{str(error)}[/red]\n\n"
        f"[yellow]Stack Trace:[/yellow]\n"
        f"{Syntax(traceback.format_exc(), 'python', background_color='default')}",
        title="🚨 Application Crashed",
        border_style="red"
    )
    console.print(panel)

def print_warning(message):
    """Print a warning message"""
    console.print(f"[yellow]⚠️ Warning: {message}[/yellow]")
