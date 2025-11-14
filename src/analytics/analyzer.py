"""
Multi-Model Analytics & Visualization Module
=============================================
Advanced analytics, visualization, and reporting for LLM evaluation results

Author: Rosalina Torres
License: MIT
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st
from wordcloud import WordCloud
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# Analytics Classes
# ============================================================================

class PerformanceAnalyzer:
    """Analyze model performance metrics"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Per-model metrics
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]
            
            metrics[model] = {
                'avg_quality': model_df['quality_score'].mean(),
                'std_quality': model_df['quality_score'].std(),
                'avg_bias': model_df['bias_score'].mean(),
                'std_bias': model_df['bias_score'].std(),
                'avg_response_time': model_df['response_time'].mean(),
                'p95_response_time': model_df['response_time'].quantile(0.95),
                'error_rate': (model_df['error'].notna()).mean(),
                'best_response_rate': model_df['is_best'].mean(),
                'consistency': 1 - model_df['quality_score'].std() if len(model_df) > 1 else 1.0
            }
        
        # Overall metrics
        metrics['overall'] = {
            'total_evaluations': len(self.df),
            'unique_prompts': self.df['prompt_id'].nunique(),
            'categories_tested': self.df['category'].nunique(),
            'avg_consensus': self.df['consensus_score'].mean(),
            'total_errors': self.df['error'].notna().sum()
        }
        
        return metrics
    
    def get_model_rankings(self) -> pd.DataFrame:
        """Generate model rankings across different metrics"""
        rankings = []
        
        for model, stats in self.metrics.items():
            if model != 'overall':
                rankings.append({
                    'model': model,
                    'quality_rank': 0,  # Will be calculated
                    'speed_rank': 0,
                    'reliability_rank': 0,
                    'overall_rank': 0
                })
        
        rankings_df = pd.DataFrame(rankings)
        
        # Calculate ranks
        rankings_df['quality_rank'] = rankings_df.index + 1  # Placeholder
        rankings_df['speed_rank'] = rankings_df.index + 1
        rankings_df['reliability_rank'] = rankings_df.index + 1
        rankings_df['overall_rank'] = (
            rankings_df[['quality_rank', 'speed_rank', 'reliability_rank']].mean(axis=1)
        ).rank().astype(int)
        
        return rankings_df.sort_values('overall_rank')
    
    def statistical_comparison(self, metric: str = 'quality_score') -> Dict[str, Any]:
        """Perform statistical comparison between models"""
        models = self.df['model'].unique()
        
        # ANOVA test
        groups = [self.df[self.df['model'] == m][metric].values for m in models]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Pairwise comparisons
        pairwise = {}
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                group1 = self.df[self.df['model'] == model1][metric].values
                group2 = self.df[self.df['model'] == model2][metric].values
                
                t_stat, p_val = stats.ttest_ind(group1, group2)
                pairwise[f"{model1} vs {model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        return {
            'anova': {'f_statistic': f_stat, 'p_value': p_value},
            'pairwise': pairwise
        }

class BiasAnalyzer:
    """Analyze bias patterns across models"""
    
    def __init__(self, bias_data: Dict[str, Dict[str, float]]):
        self.bias_data = bias_data
        self.bias_df = self._create_bias_dataframe()
    
    def _create_bias_dataframe(self) -> pd.DataFrame:
        """Convert bias data to DataFrame"""
        records = []
        for model, biases in self.bias_data.items():
            for bias_type, score in biases.items():
                records.append({
                    'model': model,
                    'bias_type': bias_type,
                    'score': score
                })
        return pd.DataFrame(records)
    
    def get_bias_summary(self) -> pd.DataFrame:
        """Get summary of bias scores by model and type"""
        return self.bias_df.pivot_table(
            index='model',
            columns='bias_type',
            values='score',
            aggfunc='mean'
        ).round(3)
    
    def identify_problematic_biases(self, threshold: float = 0.5) -> List[Dict]:
        """Identify biases above threshold"""
        problematic = self.bias_df[self.bias_df['score'] > threshold]
        return problematic.to_dict('records')
    
    def bias_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation between different bias types"""
        pivot = self.bias_df.pivot(index='model', columns='bias_type', values='score')
        return pivot.corr().round(3)

class CostAnalyzer:
    """Analyze cost implications of model usage"""
    
    def __init__(self):
        self.pricing = {
            'gpt-4': {'per_1k_tokens': 0.03, 'per_request': 0.0},
            'gpt-3.5-turbo': {'per_1k_tokens': 0.002, 'per_request': 0.0},
            'claude-3-sonnet': {'per_1k_tokens': 0.015, 'per_request': 0.0},
            'gemini-pro': {'per_1k_tokens': 0.001, 'per_request': 0.0},
            'llama-2-70b': {'per_1k_tokens': 0.0008, 'per_request': 0.01}
        }
    
    def calculate_costs(
        self,
        results_df: pd.DataFrame,
        avg_tokens_per_request: int = 500
    ) -> pd.DataFrame:
        """Calculate costs for model usage"""
        cost_data = []
        
        for model in results_df['model'].unique():
            model_df = results_df[results_df['model'] == model]
            num_requests = len(model_df)
            
            pricing = self.pricing.get(model, {'per_1k_tokens': 0.01, 'per_request': 0})
            
            token_cost = (avg_tokens_per_request / 1000) * pricing['per_1k_tokens'] * num_requests
            request_cost = pricing['per_request'] * num_requests
            total_cost = token_cost + request_cost
            
            cost_data.append({
                'model': model,
                'requests': num_requests,
                'token_cost': token_cost,
                'request_cost': request_cost,
                'total_cost': total_cost,
                'cost_per_request': total_cost / num_requests if num_requests > 0 else 0
            })
        
        return pd.DataFrame(cost_data)
    
    def cost_quality_analysis(
        self,
        results_df: pd.DataFrame,
        quality_col: str = 'quality_score'
    ) -> pd.DataFrame:
        """Analyze cost vs quality trade-offs"""
        costs = self.calculate_costs(results_df)
        quality = results_df.groupby('model')[quality_col].mean().reset_index()
        
        analysis = costs.merge(quality, on='model')
        analysis['quality_per_dollar'] = (
            analysis[quality_col] / analysis['total_cost']
        ).replace([np.inf, -np.inf], 0)
        
        return analysis

# ============================================================================
# Visualization Functions
# ============================================================================

class AdvancedVisualizer:
    """Advanced visualization for evaluation results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.analyzer = PerformanceAnalyzer(results_df)
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Quality Scores by Model',
                'Response Time Distribution',
                'Error Rates',
                'Best Response Rate'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'violin'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        models = self.df['model'].unique()
        colors = px.colors.qualitative.Plotly
        
        # Quality scores box plot
        for i, model in enumerate(models):
            model_df = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_df['quality_score'],
                    name=model,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=1
            )
        
        # Response time violin plot
        for i, model in enumerate(models):
            model_df = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Violin(
                    y=model_df['response_time'],
                    name=model,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
        
        # Error rates
        error_rates = self.df.groupby('model')['error'].apply(
            lambda x: (x.notna()).mean()
        ).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=error_rates['model'],
                y=error_rates['error'],
                marker_color=colors[:len(error_rates)]
            ),
            row=2, col=1
        )
        
        # Best response rate
        best_rates = self.df.groupby('model')['is_best'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=best_rates['model'],
                y=best_rates['is_best'],
                marker_color=colors[:len(best_rates)]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Multi-Model Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_bias_heatmap(self, bias_analyzer: BiasAnalyzer) -> go.Figure:
        """Create bias heatmap"""
        bias_summary = bias_analyzer.get_bias_summary()
        
        fig = go.Figure(data=go.Heatmap(
            z=bias_summary.values,
            x=bias_summary.columns,
            y=bias_summary.index,
            colorscale='RdYlGn_r',
            text=bias_summary.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Bias Score")
        ))
        
        fig.update_layout(
            title="Bias Analysis Heatmap",
            xaxis_title="Bias Type",
            yaxis_title="Model",
            height=500
        )
        
        return fig
    
    def create_category_performance(self) -> go.Figure:
        """Create category-wise performance comparison"""
        category_perf = self.df.groupby(['category', 'model']).agg({
            'quality_score': 'mean',
            'response_time': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            category_perf,
            x='response_time',
            y='quality_score',
            color='model',
            size='quality_score',
            facet_col='category',
            title="Performance by Category",
            labels={
                'response_time': 'Avg Response Time (s)',
                'quality_score': 'Avg Quality Score'
            }
        )
        
        return fig
    
    def create_consensus_analysis(self) -> go.Figure:
        """Visualize consensus patterns"""
        consensus_by_category = self.df.groupby('category')['consensus_score'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=consensus_by_category['category'],
            y=consensus_by_category['mean'],
            error_y=dict(
                type='data',
                array=consensus_by_category['std']
            ),
            marker_color='lightblue',
            name='Mean Consensus'
        ))
        
        fig.update_layout(
            title="Consensus Score by Category",
            xaxis_title="Category",
            yaxis_title="Consensus Score",
            showlegend=False
        )
        
        return fig
    
    def create_response_clustering(self) -> go.Figure:
        """Create response clustering visualization"""
        # Prepare data for clustering
        features = self.df[['quality_score', 'response_time', 'consensus_score']].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Create visualization
        fig = px.scatter(
            x=features_pca[:, 0],
            y=features_pca[:, 1],
            color=self.df['model'],
            symbol=clusters.astype(str),
            title="Response Clustering Analysis",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        
        return fig
    
    def create_timeline_analysis(self) -> go.Figure:
        """Create timeline analysis of response times"""
        # Add index as pseudo-timestamp
        self.df['evaluation_order'] = range(len(self.df))
        
        fig = px.line(
            self.df,
            x='evaluation_order',
            y='response_time',
            color='model',
            title="Response Time Timeline",
            labels={
                'evaluation_order': 'Evaluation Sequence',
                'response_time': 'Response Time (s)'
            }
        )
        
        # Add rolling average
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model].copy()
            model_df['rolling_avg'] = model_df['response_time'].rolling(
                window=5, center=True
            ).mean()
            
            fig.add_trace(go.Scatter(
                x=model_df['evaluation_order'],
                y=model_df['rolling_avg'],
                mode='lines',
                name=f'{model} (avg)',
                line=dict(dash='dash'),
                showlegend=False
            ))
        
        return fig

class ReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        bias_data: Optional[Dict] = None
    ):
        self.df = results_df
        self.bias_data = bias_data
        self.analyzer = PerformanceAnalyzer(results_df)
        self.visualizer = AdvancedVisualizer(results_df)
        
    def generate_html_report(self, output_path: str = "evaluation_report.html"):
        """Generate interactive HTML report"""
        html_content = []
        
        # Header
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Model Evaluation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #555;
                    margin-top: 30px;
                }
                .metric-card {
                    display: inline-block;
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 8px;
                    min-width: 200px;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-label {
                    color: #666;
                    margin-top: 5px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background: #667eea;
                    color: white;
                }
                tr:hover {
                    background: #f5f5f5;
                }
            </style>
        </head>
        <body>
            <div class="container">
        """)
        
        # Title and metadata
        html_content.append(f"""
            <h1>🚀 Multi-Model LLM Evaluation Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Evaluations:</strong> {len(self.df)}</p>
        """)
        
        # Key metrics cards
        html_content.append("<h2>📊 Key Metrics</h2>")
        html_content.append('<div class="metrics-container">')
        
        metrics = self.analyzer.metrics['overall']
        for key, value in metrics.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                value = f"{value:.3f}"
            html_content.append(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """)
        
        html_content.append("</div>")
        
        # Model rankings table
        html_content.append("<h2>🏆 Model Rankings</h2>")
        rankings = self.analyzer.get_model_rankings()
        html_content.append(rankings.to_html(classes='ranking-table', index=False))
        
        # Performance dashboard
        html_content.append("<h2>📈 Performance Dashboard</h2>")
        perf_fig = self.visualizer.create_performance_dashboard()
        html_content.append(f'<div id="performance-dashboard"></div>')
        html_content.append(f"""
            <script>
                Plotly.newPlot('performance-dashboard', {perf_fig.to_json()});
            </script>
        """)
        
        # Category performance
        html_content.append("<h2>📂 Category Performance</h2>")
        cat_fig = self.visualizer.create_category_performance()
        html_content.append(f'<div id="category-performance"></div>')
        html_content.append(f"""
            <script>
                Plotly.newPlot('category-performance', {cat_fig.to_json()});
            </script>
        """)
        
        # Statistical comparison
        html_content.append("<h2>📊 Statistical Analysis</h2>")
        stats_comparison = self.analyzer.statistical_comparison()
        html_content.append("<h3>ANOVA Results</h3>")
        html_content.append(f"""
            <p>F-statistic: {stats_comparison['anova']['f_statistic']:.4f}</p>
            <p>P-value: {stats_comparison['anova']['p_value']:.4f}</p>
        """)
        
        # Footer
        html_content.append("""
            </div>
        </body>
        </html>
        """)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        return output_path
    
    def generate_excel_report(self, output_path: str = "evaluation_report.xlsx"):
        """Generate Excel report with multiple sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([self.analyzer.metrics['overall']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Model performance sheet
            model_metrics_df = pd.DataFrame(self.analyzer.metrics).T
            model_metrics_df.to_excel(writer, sheet_name='Model Performance')
            
            # Raw data sheet
            self.df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Rankings sheet
            rankings = self.analyzer.get_model_rankings()
            rankings.to_excel(writer, sheet_name='Rankings', index=False)
            
            # Cost analysis sheet
            cost_analyzer = CostAnalyzer()
            costs = cost_analyzer.cost_quality_analysis(self.df)
            costs.to_excel(writer, sheet_name='Cost Analysis', index=False)
        
        return output_path

# ============================================================================
# Streamlit Dashboard
# ============================================================================

class StreamlitDashboard:
    """Interactive Streamlit dashboard for evaluation results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.analyzer = PerformanceAnalyzer(results_df)
        self.visualizer = AdvancedVisualizer(results_df)
    
    def run(self):
        """Run Streamlit dashboard"""
        st.set_page_config(
            page_title="Multi-Model LLM Evaluation Dashboard",
            page_icon="🚀",
            layout="wide"
        )
        
        # Title
        st.title("🚀 Multi-Model LLM Evaluation Dashboard")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            selected_models = st.multiselect(
                "Select Models",
                options=self.df['model'].unique(),
                default=self.df['model'].unique()
            )
            
            # Category selection
            selected_categories = st.multiselect(
                "Select Categories",
                options=self.df['category'].unique(),
                default=self.df['category'].unique()
            )
            
            # Metric selection
            metric = st.selectbox(
                "Primary Metric",
                options=['quality_score', 'response_time', 'consensus_score']
            )
        
        # Filter data
        filtered_df = self.df[
            (self.df['model'].isin(selected_models)) &
            (self.df['category'].isin(selected_categories))
        ]
        
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Evaluations",
                len(filtered_df),
                delta=f"{len(filtered_df) - len(self.df)} from full dataset"
            )
        
        with col2:
            avg_quality = filtered_df['quality_score'].mean()
            st.metric(
                "Avg Quality Score",
                f"{avg_quality:.3f}",
                delta=f"{avg_quality - self.df['quality_score'].mean():.3f}"
            )
        
        with col3:
            avg_time = filtered_df['response_time'].mean()
            st.metric(
                "Avg Response Time",
                f"{avg_time:.2f}s",
                delta=f"{avg_time - self.df['response_time'].mean():.2f}s"
            )
        
        with col4:
            error_rate = (filtered_df['error'].notna()).mean()
            st.metric(
                "Error Rate",
                f"{error_rate:.1%}",
                delta=f"{error_rate - (self.df['error'].notna()).mean():.1%}"
            )
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance",
            "🎯 Bias Analysis",
            "💰 Cost Analysis",
            "📈 Advanced Analytics"
        ])
        
        with tab1:
            st.subheader("Performance Overview")
            
            # Performance dashboard
            perf_fig = self.visualizer.create_performance_dashboard()
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            comparison_df = filtered_df.groupby('model').agg({
                'quality_score': ['mean', 'std'],
                'response_time': ['mean', 'std'],
                'error': lambda x: (x.notna()).mean()
            }).round(3)
            st.dataframe(comparison_df)
        
        with tab2:
            st.subheader("Bias Analysis")
            
            if hasattr(self, 'bias_analyzer'):
                bias_fig = self.visualizer.create_bias_heatmap(self.bias_analyzer)
                st.plotly_chart(bias_fig, use_container_width=True)
                
                # Problematic biases
                st.subheader("⚠️ Problematic Biases")
                threshold = st.slider("Bias Threshold", 0.0, 1.0, 0.5)
                problematic = self.bias_analyzer.identify_problematic_biases(threshold)
                if problematic:
                    st.dataframe(pd.DataFrame(problematic))
                else:
                    st.success("No biases above threshold!")
        
        with tab3:
            st.subheader("Cost Analysis")
            
            cost_analyzer = CostAnalyzer()
            costs = cost_analyzer.calculate_costs(filtered_df)
            
            # Cost comparison chart
            fig = px.bar(
                costs,
                x='model',
                y='total_cost',
                title='Total Cost by Model'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost-quality analysis
            cost_quality = cost_analyzer.cost_quality_analysis(filtered_df)
            st.dataframe(cost_quality)
        
        with tab4:
            st.subheader("Advanced Analytics")
            
            # Clustering analysis
            clustering_fig = self.visualizer.create_response_clustering()
            st.plotly_chart(clustering_fig, use_container_width=True)
            
            # Timeline analysis
            timeline_fig = self.visualizer.create_timeline_analysis()
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Statistical tests
            st.subheader("Statistical Tests")
            stats_results = self.analyzer.statistical_comparison(metric)
            
            st.write("**ANOVA Results:**")
            st.write(f"F-statistic: {stats_results['anova']['f_statistic']:.4f}")
            st.write(f"P-value: {stats_results['anova']['p_value']:.4f}")
            
            if stats_results['anova']['p_value'] < 0.05:
                st.success("Significant differences found between models!")
            else:
                st.info("No significant differences found between models.")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution for analytics module"""
    
    # Load sample data (replace with actual results)
    sample_data = {
        'prompt_id': ['p1', 'p2', 'p3'] * 4,
        'category': ['reasoning', 'creative', 'technical'] * 4,
        'model': ['gpt-4'] * 3 + ['claude-3'] * 3 + ['gemini-pro'] * 3 + ['llama-2'] * 3,
        'provider': ['openai'] * 3 + ['anthropic'] * 3 + ['google'] * 3 + ['meta'] * 3,
        'quality_score': np.random.uniform(0.6, 1.0, 12),
        'bias_score': np.random.uniform(0, 0.3, 12),
        'response_time': np.random.uniform(0.5, 3.0, 12),
        'consensus_score': np.random.uniform(0.7, 1.0, 12),
        'is_best': np.random.choice([True, False], 12),
        'error': [None] * 10 + ['timeout', 'api_error']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize components
    analyzer = PerformanceAnalyzer(df)
    visualizer = AdvancedVisualizer(df)
    report_gen = ReportGenerator(df)
    
    # Generate reports
    print("Generating HTML report...")
    html_path = report_gen.generate_html_report()
    print(f"HTML report saved to: {html_path}")
    
    print("Generating Excel report...")
    excel_path = report_gen.generate_excel_report()
    print(f"Excel report saved to: {excel_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("ANALYTICS SUMMARY")
    print("="*60)
    
    for model, metrics in analyzer.metrics.items():
        if model != 'overall':
            print(f"\n{model}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Run 'streamlit run analytics.py' to launch interactive dashboard")

if __name__ == "__main__":
    main()
