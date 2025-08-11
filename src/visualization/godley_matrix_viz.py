#!/usr/bin/env python3
"""
Visualization for Godley matrices with bilateral flow detail.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Optional

class GodleyMatrixVisualizer:
    """Visualize Godley balance sheet and transaction matrices."""
    
    def __init__(self, output_dir: str = "output/godley"):
        self.output_dir = output_dir
        
    def plot_godley_flow_matrix(self, 
                                flow_matrices: Dict[str, pd.DataFrame],
                                period: str):
        """
        Plot Godley transaction flow matrices.
        
        Parameters:
        -----------
        flow_matrices : Dict[str, pd.DataFrame]
            Dictionary of instrument -> sector×sector flow matrix
        period : str
            Time period label
        """
        n_instruments = len(flow_matrices)
        fig, axes = plt.subplots(2, (n_instruments+1)//2, 
                                figsize=(6*(n_instruments+1)//2, 10))
        axes = axes.flatten() if n_instruments > 1 else [axes]
        
        for idx, (inst_code, matrix) in enumerate(flow_matrices.items()):
            ax = axes[idx]
            
            # Create heatmap
            sns.heatmap(matrix, 
                       annot=True, 
                       fmt='.0f',
                       cmap='RdBu_r',
                       center=0,
                       ax=ax,
                       cbar_kws={'label': 'Flow (Billions $)'})
            
            ax.set_title(f'Bilateral Flows: {inst_code}')
            ax.set_xlabel('Issuer (Source of Funds)')
            ax.set_ylabel('Holder (Use of Funds)')
        
        plt.suptitle(f'Godley Transaction Flow Matrices - {period}', fontsize=14)
        plt.tight_layout()
        
        # Save
        fig.savefig(f"{self.output_dir}/godley_flows_{period}.png", dpi=150)
        return fig
    
    def plot_flow_network(self,
                         bilateral_flows: pd.DataFrame,
                         instrument: str,
                         threshold: float = 100):
        """
        Plot bilateral flows as a network graph.
        """
        # Filter for specific instrument and significant flows
        inst_flows = bilateral_flows[
            (bilateral_flows['instrument'] == instrument) &
            (abs(bilateral_flows['flow']) > threshold)
        ]
        
        if len(inst_flows) == 0:
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        for _, flow in inst_flows.iterrows():
            G.add_edge(flow['issuer'], 
                      flow['holder'],
                      weight=abs(flow['flow']),
                      flow=flow['flow'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=2000,
                              node_color='lightblue',
                              ax=ax)
        
        # Draw edges with width based on flow size
        edges = G.edges()
        weights = [G[u][v]['weight']/100 for u, v in edges]
        
        nx.draw_networkx_edges(G, pos,
                              width=weights,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='-|>',
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Add edge labels with flow amounts
        edge_labels = {(u, v): f"${G[u][v]['flow']:.0f}B" 
                      for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        
        ax.set_title(f'Bilateral Flow Network: {instrument}')
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/flow_network_{instrument}.png", dpi=150)
        return fig
    
    def create_godley_report(self, godley_accounts: Dict, period: str):
        """
        Create comprehensive Godley accounting report.
        """
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Godley Accounting Report - {period}</title>
            <style>
                table {{ border-collapse: collapse; margin: 20px; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: right; }}
                th {{ background-color: #f0f0f0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Godley Accounting Report - Period: {period}</h1>
        """
        
        # Add balance sheet
        html += "<h2>Balance Sheet Matrix</h2>"
        html += godley_accounts['balance_sheet'].to_html(classes='table')
        
        # Add transaction matrices
        html += "<h2>Transaction Flow Matrices</h2>"
        for inst, matrix in godley_accounts['transactions'].items():
            html += f"<h3>Instrument: {inst}</h3>"
            html += matrix.to_html(classes='table')
        
        # Add validation metrics
        html += "<h2>Validation Results</h2>"
        html += "<ul>"
        for key, value in godley_accounts['validation'].items():
            html += f"<li>{key}: {value}</li>"
        html += "</ul>"
        
        html += "</body></html>"
        
        # Save report
        with open(f"{self.output_dir}/godley_report_{period}.html", 'w') as f:
            f.write(html)
        
        return html
