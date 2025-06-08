import os
import argparse
import subprocess
import time
import pandas as pd
import sys
from datetime import datetime

def run_command(cmd):
    """Run a command and return output."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Real-time output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Get any errors
        _, stderr = process.communicate()
        
        if return_code != 0:
            print(f"Error running command: {' '.join(cmd)}")
            print(f"Error details: {stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def generate_html_report(benchmark_dir, output_path):
    """Generate an HTML report from benchmark results."""
    # Load summary data
    summary_path = os.path.join(benchmark_dir, 'model_comparison_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return False
    
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return False
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Model Benchmark Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin: 30px 0;
            }}
            .chart {{
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                background: white;
                padding: 15px;
                border-radius: 5px;
            }}
            .chart img {{
                max-width: 100%;
                height: auto;
            }}
            .chart-title {{
                text-align: center;
                font-weight: bold;
                margin: 10px 0;
            }}
            .section {{
                margin: 40px 0;
            }}
            .model-details {{
                margin: 30px 0;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Face Recognition Model Benchmark Report</h1>
            <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="section">
            <h2>Overview</h2>
            <p>This report contains benchmark results for multiple face recognition models tested on the CFP dataset, which contains frontal and profile face images.</p>
        </div>
        
        <div class="section">
            <h2>Summary Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Frontal-Frontal Accuracy (%)</th>
                    <th>Profile-Profile Accuracy (%)</th>
                    <th>Frontal-Profile Accuracy (%)</th>
                    <th>ROC AUC</th>
                    <th>EER</th>
                    <th>Processing Time (s)</th>
                </tr>
    """
    
    # Add rows to the table
    for _, row in summary_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['Model']}</td>
                    <td>{row['FF_Accuracy']:.1f}</td>
                    <td>{row['PP_Accuracy']:.1f}</td>
                    <td>{row['FP_Accuracy']:.1f}</td>
                    <td>{row['ROC_AUC']:.3f}</td>
                    <td>{row['EER']:.3f}</td>
                    <td>{row['Processing_Time']:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            
            <div class="chart-container">
    """
    
    # Add all PNG files as charts
    chart_files = [f for f in os.listdir(benchmark_dir) if f.endswith('.png')]
    for chart_file in chart_files:
        chart_name = chart_file.replace('_', ' ').replace('.png', '').title()
        html_content += f"""
                <div class="chart">
                    <div class="chart-title">{chart_name}</div>
                    <img src="{chart_file}" alt="{chart_name}">
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.8em;">
            Face Recognition Model Benchmark Report<br>
            Generated using custom benchmarking tools
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report generated at {output_path}")
        return True
    except Exception as e:
        print(f"Error writing HTML report: {e}")
        return False

def check_prerequisites():
    """Check if the necessary prerequisites are installed."""
    try:
        import torch
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        from PIL import Image
        from tqdm import tqdm
        return True
    except ImportError as e:
        print(f"Missing prerequisite: {e}")
        print("Please install the required packages using: pip install -r requirements.txt")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for face recognition models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CFP dataset images')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated list of models to test (elasticface,adaface,facenet512,buffalo_l) or "all"')
    parser.add_argument('--use_demo', action='store_true',
                        help='Use demo models')
    parser.add_argument('--skip_vis', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Starting Face Recognition Model Benchmark")
    print("="*80)
    
    # Step 1: Run the benchmark
    benchmark_cmd = [
        sys.executable, 'benchmark_all_models.py',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--models', args.models
    ]
    
    if args.use_demo:
        benchmark_cmd.append('--use_demo')
    
    benchmark_success = run_command(benchmark_cmd)
    if not benchmark_success:
        print("\nBenchmark failed. Checking if partial results are available...")
        summary_path = os.path.join(args.output_dir, 'model_comparison_summary.csv')
        if not os.path.exists(summary_path):
            print("No results available. Exiting.")
            sys.exit(1)
        print("Partial results found. Continuing with report generation...")
    
    # Step 2: Generate visualizations (if not skipped)
    if not args.skip_vis:
        print("\n" + "="*80)
        print("Generating Visualizations")
        print("="*80)
        
        visualize_cmd = [
            sys.executable, 'visualize_benchmark_results.py',
            '--benchmark_dir', args.output_dir
        ]
        
        vis_success = run_command(visualize_cmd)
        if not vis_success:
            print("Visualization generation failed. Continuing with report...")
    
    # Step 3: Generate HTML report
    print("\n" + "="*80)
    print("Generating HTML Report")
    print("="*80)
    
    report_path = os.path.join(args.output_dir, 'benchmark_report.html')
    report_success = generate_html_report(args.output_dir, report_path)
    
    if report_success:
        print(f"\nBenchmark complete! Report available at {report_path}")
    else:
        print("\nFailed to generate HTML report.")
    
    print("\n" + "="*80)
    print(f"Results available in: {os.path.abspath(args.output_dir)}")
    print("="*80)

if __name__ == '__main__':
    main() 