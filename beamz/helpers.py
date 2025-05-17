import sys
import datetime
from typing import List, Dict, Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.syntax import Syntax
import numpy as np
from beamz.const import LIGHT_SPEED

def calc_optimal_fdtd_params(wavelength, n_max, dims=2, safety_factor=0.45, points_per_wavelength=20):
    """
    Calculate optimal FDTD grid resolution and time step based on wavelength and material properties.
    
    Args:
        wavelength: Light wavelength in vacuum
        n_max: Maximum refractive index in the simulation
        safety_factor: Multiplier for Courant condition (0.5 recommended for stability)
        points_per_wavelength: Number of grid points per wavelength in the highest index material
        
    Returns:
        tuple: (resolution, dt) - optimal spatial resolution and time step
    """
    # Calculate wavelength in the highest index material
    lambda_material = wavelength / n_max
    # Calculate optimal grid resolution based on desired points per wavelength
    resolution = lambda_material / points_per_wavelength
    # Calculate speed of light in the material
    c_material = LIGHT_SPEED / n_max
    # Calculate time step using Courant condition for ND simulation
    dt_max = resolution / (c_material * np.sqrt(dims))
    # Apply safety factor
    dt = safety_factor * dt_max
    return resolution, dt

# Initialize rich console
console = Console()

def progress_bar(progress:int, total:int, length:int=50):
    """Print a progress bar to the console."""
    percent = 100 * (progress / float(total))
    filled_length = int(length * progress // total)
    bar = '█' * filled_length + '-' * (length - filled_length - 1)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')
    sys.stdout.flush()

def display_header(title: str, subtitle: Optional[str] = None) -> None:
    """Display a formatted header with optional subtitle."""
    console.print(Panel(f"[bold blue]{title}[/]", subtitle=subtitle, expand=False))

def display_status(status: str, status_type: str = "info") -> None:
    """
    Display a status message with appropriate styling.
    
    Args:
        status: The status message to display
        status_type: One of "info", "success", "warning", "error"
    """
    style_map = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }
    style = style_map.get(status_type, "white")
    console.print(f"[{style}]● {status}[/]")

def create_rich_progress() -> Progress:
    """Create and return a rich progress bar for tracking processes."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )

def display_parameters(params: Dict[str, Any], title: str = "Parameters") -> None:
    """
    Display a dictionary of parameters in a clean, formatted table.
    
    Args:
        params: Dictionary of parameter names and values
        title: Title for the parameters table
    """
    table = Table(title=title)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in params.items():
        table.add_row(str(key), str(value))
    
    console.print(table)

def display_results(results: Dict[str, Any], title: str = "Results") -> None:
    """
    Display simulation or optimization results in a formatted table.
    
    Args:
        results: Dictionary of result names and values
        title: Title for the results table
    """
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        # Format numeric values nicely
        if isinstance(value, (int, float)):
            value_str = f"{value:.6g}"
        else:
            value_str = str(value)
        table.add_row(str(key), value_str)
    
    console.print(table)

def display_design_summary(design_name: str, properties: Dict[str, Any]) -> None:
    """
    Display a summary of a design with its key properties.
    
    Args:
        design_name: Name of the design
        properties: Dictionary of design properties
    """
    console.print(Panel(
        f"[bold cyan]Design:[/] [yellow]{design_name}[/]",
        title="Design Summary",
        expand=False
    ))
    
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="blue")
    table.add_column("Value", style="white")
    
    for prop, value in properties.items():
        table.add_row(prop, str(value))
    
    console.print(table)

def display_simulation_status(progress: float, metrics: Dict[str, Any] = None) -> None:
    """
    Display current simulation status with progress and metrics.
    
    Args:
        progress: Progress percentage (0-100)
        metrics: Current simulation metrics
    """
    progress_text = f"Simulation Progress: [bold cyan]{progress:.1f}%[/]"
    console.print(progress_text)
    
    if metrics:
        metrics_panel = Panel(
            "\n".join([f"[blue]{k}:[/] {v}" for k, v in metrics.items()]),
            title="Current Metrics",
            expand=False
        )
        console.print(metrics_panel)

def display_optimization_progress(iteration: int, total: int, best_value: float, 
                                 parameters: Dict[str, Any] = None) -> None:
    """
    Display optimization progress information.
    
    Args:
        iteration: Current iteration number
        total: Total number of iterations
        best_value: Best objective value found so far
        parameters: Current best parameters
    """
    console.rule(f"[bold magenta]Optimization - Iteration {iteration}/{total}[/]")
    console.print(f"Best objective value: [bold green]{best_value:.6g}[/]")
    
    if parameters:
        table = Table(title="Best Parameters")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in parameters.items():
            if isinstance(value, float):
                table.add_row(str(key), f"{value:.6g}")
            else:
                table.add_row(str(key), str(value))
        
        console.print(table)

def display_time_elapsed(start_time: datetime.datetime) -> None:
    """
    Display the time elapsed since the start time.
    
    Args:
        start_time: The start datetime
    """
    elapsed = datetime.datetime.now() - start_time
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"[bold]Time elapsed:[/] "
    if hours > 0:
        time_str += f"{int(hours)}h "
    if minutes > 0 or hours > 0:
        time_str += f"{int(minutes)}m "
    time_str += f"{seconds:.1f}s"
    
    console.print(time_str)

def code_preview(code: str, language: str = "python") -> None:
    """
    Display formatted code with syntax highlighting.
    
    Args:
        code: The code to display
        language: The programming language for syntax highlighting
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def tree_view(data: Dict[str, Any], title: str = "Structure") -> None:
    """
    Display nested data in a tree view.
    
    Args:
        data: Nested dictionary to display as a tree
        title: Title for the tree
    """
    tree = Tree(f"[bold]{title}[/]")
    
    def _add_to_tree(tree_node, data_node):
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                if isinstance(value, (dict, list)):
                    branch = tree_node.add(f"[blue]{key}[/]")
                    _add_to_tree(branch, value)
                else:
                    tree_node.add(f"[blue]{key}:[/] {value}")
        elif isinstance(data_node, list):
            for i, item in enumerate(data_node):
                if isinstance(item, (dict, list)):
                    branch = tree_node.add(f"[green]{i}[/]")
                    _add_to_tree(branch, item)
                else:
                    tree_node.add(f"[green]{i}:[/] {item}")
    
    _add_to_tree(tree, data)
    console.print(tree)