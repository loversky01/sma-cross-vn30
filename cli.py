"""
cli.py
────────────────────────────────────────────────────────────
Command-Line Interface cho SMA Cross VN30 system

Usage:
    python cli.py --help
    python cli.py backtest MSN
    python cli.py optimize VCB
    python cli.py portfolio --max-positions 5
    python cli.py dashboard
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
import pandas as pd
import os

from utils.config_loader import get_config
from utils.logger import setup_logger, get_logger

# Initialize console
console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 SMA Cross VN30 - Quantitative Trading System"""
    pass


@cli.command()
@click.argument('symbol')
@click.option('--capital', type=float, help='Initial capital (VND)')
@click.option('--stop-loss', type=float, help='Stop loss percentage')
@click.option('--take-profit', type=float, help='Take profit percentage')
def backtest(symbol, capital, stop_loss, take_profit):
    """📊 Run backtest for a symbol"""
    from backtest_advanced import backtest_symbol
    
    console.print(f"\n[bold green]Running backtest for {symbol}...[/bold green]")
    
    # Load config
    cfg = get_config()
    
    config = {
        'initial_capital': capital if capital else cfg.initial_capital,
        'commission_pct': cfg.commission_pct,
        'tax_pct': cfg.tax_pct,
        'slippage_pct': cfg.slippage_pct,
        'stop_loss_pct': stop_loss if stop_loss else (cfg.stop_loss_pct if cfg.use_stop_loss else None),
        'take_profit_pct': take_profit if take_profit else (cfg.take_profit_pct if cfg.use_take_profit else None),
        'trailing_stop_pct': None,
        'position_size_pct': 100.0,
        'signal_dir': cfg.signal_dir
    }
    
    results = backtest_symbol(symbol.upper(), config)
    
    if results and 'metrics' in results:
        # Display results in table
        table = Table(title=f"Backtest Results - {symbol.upper()}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", width=20)
        
        metrics = results['metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                if '%' in key or 'pct' in key.lower() or 'rate' in key.lower():
                    table.add_row(key, f"{value:.2f}%")
                elif 'ratio' in key.lower():
                    table.add_row(key, f"{value:.3f}")
                else:
                    table.add_row(key, f"{value:,.2f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    else:
        console.print("[bold red]❌ Backtest failed[/bold red]")


@cli.command()
@click.argument('symbol')
@click.option('--short-range', type=str, help='Short SMA range (comma-separated)')
@click.option('--long-range', type=str, help='Long SMA range (comma-separated)')
@click.option('--metric', type=str, default='sharpe_ratio', help='Optimization metric')
def optimize(symbol, short_range, long_range, metric):
    """🔍 Optimize parameters for a symbol"""
    from optimize_parameters import optimize_symbol
    
    console.print(f"\n[bold green]Optimizing parameters for {symbol}...[/bold green]")
    
    # Parse ranges
    short = [int(x) for x in short_range.split(',')] if short_range else None
    long = [int(x) for x in long_range.split(',')] if long_range else None
    
    results = optimize_symbol(
        symbol=symbol.upper(),
        short_range=short,
        long_range=long,
        metric=metric
    )
    
    if len(results) > 0:
        console.print(f"\n[bold green]✅ Optimization completed[/bold green]")
        console.print(f"Results saved to: optimization_{symbol.upper()}.csv")
    else:
        console.print("[bold red]❌ Optimization failed[/bold red]")


@cli.command()
@click.option('--symbols', type=str, help='Comma-separated symbols (default: all VN30)')
@click.option('--max-positions', type=int, default=5, help='Maximum concurrent positions')
@click.option('--capital', type=float, help='Initial capital (VND)')
def portfolio(symbols, max_positions, capital):
    """💼 Run portfolio backtest"""
    from portfolio_backtest import run_portfolio_backtest
    
    console.print(f"\n[bold green]Running portfolio backtest...[/bold green]")
    
    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    
    # Get capital
    cfg = get_config()
    init_capital = capital if capital else cfg.initial_capital
    
    results = run_portfolio_backtest(
        symbols=symbol_list,
        max_positions=max_positions,
        initial_capital=init_capital
    )
    
    if results and 'metrics' in results:
        console.print(f"\n[bold green]✅ Portfolio backtest completed[/bold green]")
        console.print(f"Results saved to: portfolio_backtest_trades.csv")
    else:
        console.print("[bold red]❌ Portfolio backtest failed[/bold red]")


@cli.command()
@click.argument('symbol')
@click.option('--runs', type=int, default=1000, help='Number of simulation runs')
@click.option('--confidence', type=float, default=95.0, help='Confidence level for VaR')
def montecarlo(symbol, runs, confidence):
    """🎲 Run Monte Carlo simulation"""
    from monte_carlo import simulate_symbol
    
    console.print(f"\n[bold green]Running Monte Carlo simulation for {symbol}...[/bold green]")
    
    results = simulate_symbol(
        symbol=symbol.upper(),
        num_runs=runs,
        confidence=confidence
    )
    
    if results:
        console.print(f"\n[bold green]✅ Simulation completed[/bold green]")
        console.print(f"Plot saved to: monte_carlo_{symbol.upper()}.png")
    else:
        console.print("[bold red]❌ Simulation failed[/bold red]")


@cli.command()
@click.argument('symbol')
@click.option('--train-pct', type=float, default=70.0, help='Training percentage')
@click.option('--test-pct', type=float, default=30.0, help='Testing percentage')
def walkforward(symbol, train_pct, test_pct):
    """🚶 Run walk-forward analysis"""
    from walk_forward import walk_forward_symbol
    
    console.print(f"\n[bold green]Running walk-forward analysis for {symbol}...[/bold green]")
    
    results = walk_forward_symbol(
        symbol=symbol.upper(),
        train_pct=train_pct,
        test_pct=test_pct
    )
    
    if results:
        console.print(f"\n[bold green]✅ Walk-forward analysis completed[/bold green]")
    else:
        console.print("[bold red]❌ Analysis failed[/bold red]")


@cli.command()
@click.option('--port', type=int, default=8501, help='Port number')
def dashboard(port):
    """📱 Launch dashboard"""
    import subprocess
    
    console.print(f"\n[bold green]Launching dashboard on port {port}...[/bold green]")
    console.print(f"[cyan]→ http://localhost:{port}[/cyan]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
    
    try:
        subprocess.run(['streamlit', 'run', 'app_dashboard.py', '--server.port', str(port)])
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


@cli.command()
@click.option('--symbols', type=str, help='Comma-separated symbols (default: all VN30)')
def update(symbols):
    """📥 Update data for symbols"""
    from sma_cross_vn30 import download_price, add_sma_signals
    import time
    
    cfg = get_config()
    
    # Get symbols
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    else:
        symbol_list = cfg.vn30_symbols
    
    console.print(f"\n[bold green]Updating data for {len(symbol_list)} symbols...[/bold green]")
    
    for symbol in track(symbol_list, description="Downloading..."):
        df = download_price(symbol)
        
        if df is not None:
            df_signals = add_sma_signals(df)
            df_signals.to_csv(f"{cfg.signal_dir}/{symbol}_signals.csv", index=False)
            console.print(f"[green]✓[/green] {symbol}")
        else:
            console.print(f"[red]✗[/red] {symbol}")
        
        time.sleep(1.5)
    
    console.print(f"\n[bold green]✅ Update completed[/bold green]")


@cli.command()
def summary():
    """📊 Show backtest summary"""
    summary_path = "backtest_summary.csv"
    
    if not os.path.exists(summary_path):
        console.print("[bold red]❌ Summary file not found. Run 'python backtest_all_vn30.py' first.[/bold red]")
        return
    
    df = pd.read_csv(summary_path)
    
    # Display top 10
    table = Table(title="Top 10 Best Performing Stocks", show_header=True, header_style="bold magenta")
    table.add_column("Mã", style="cyan", width=10)
    table.add_column("Tỷ Suất (%)", style="green", width=15)
    table.add_column("Win Rate (%)", style="yellow", width=15)
    table.add_column("Số Lệnh", style="blue", width=12)
    
    for _, row in df.head(10).iterrows():
        table.add_row(
            row['Mã'],
            f"{row['Tỷ suất (%)']:.2f}",
            f"{row['Win rate (%)']:.2f}",
            str(row['Số lệnh'])
        )
    
    console.print(table)


@cli.command()
def config():
    """⚙️ Show current configuration"""
    cfg = get_config()
    
    table = Table(title="Current Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="green", width=40)
    
    # Strategy settings
    table.add_row("[bold]Strategy[/bold]", "")
    table.add_row("  SMA Short", str(cfg.sma_short))
    table.add_row("  SMA Long", str(cfg.sma_long))
    
    # Risk management
    table.add_row("[bold]Risk Management[/bold]", "")
    table.add_row("  Stop Loss", f"{cfg.stop_loss_pct}%" if cfg.use_stop_loss else "Disabled")
    table.add_row("  Take Profit", f"{cfg.take_profit_pct}%" if cfg.use_take_profit else "Disabled")
    
    # Backtest
    table.add_row("[bold]Backtest[/bold]", "")
    table.add_row("  Initial Capital", f"{cfg.initial_capital:,} VND")
    table.add_row("  Commission", f"{cfg.commission_pct}%")
    table.add_row("  Tax", f"{cfg.tax_pct}%")
    table.add_row("  Slippage", f"{cfg.slippage_pct}%")
    
    console.print(table)


@cli.command()
@click.argument('symbol')
def plot(symbol):
    """📈 Plot SMA chart for a symbol"""
    from plot_sma_chart import plot_sma_vi
    
    console.print(f"\n[bold green]Plotting chart for {symbol}...[/bold green]")
    
    plot_sma_vi(symbol.upper())


if __name__ == '__main__':
    cli()

