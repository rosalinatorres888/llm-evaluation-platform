"""
Multi-Model Prompt Engineering Platform - Complete Demo
========================================================
This script demonstrates all major features of the platform
with real-world examples and best practices.

Author: Rosalina Torres
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules (assuming they're in src/)
from src.core.engine import MultiModelEvaluationEngine
from src.core.models import (
    ModelConfig, ModelProvider, PromptTemplate, 
    EvaluationCategory, ModelResponse, EvaluationResult
)
from src.evaluation.bias_detector import AdvancedBiasDetector
from src.evaluation.quality import ComprehensiveQualityEvaluator
from src.analytics.analyzer import PerformanceAnalyzer, BiasAnalyzer, CostAnalyzer
from src.analytics.visualizer import AdvancedVisualizer
from src.analytics.reporter import ReportGenerator

# ============================================================================
# Configuration
# ============================================================================

def setup_environment():
    """Setup environment and configurations"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create necessary directories
    directories = [
        "data/raw", "data/processed", "data/reports",
        "prompts/templates", "prompts/categories",
        "logs", "cache"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")

def get_model_configs():
    """Get model configurations for all providers"""
    
    configs = []
    
    # OpenAI Configuration
    if os.getenv("OPENAI_API_KEY"):
        configs.extend([
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=1000
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
        ])
        print("✅ OpenAI models configured")
    
    # Anthropic Configuration
    if os.getenv("ANTHROPIC_API_KEY"):
        configs.append(
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=1000
            )
        )
        print("✅ Anthropic models configured")
    
    # Google Configuration
    if os.getenv("GOOGLE_API_KEY"):
        configs.append(
            ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                temperature=0.7,
                max_tokens=1000
            )
        )
        print("✅ Google models configured")
    
    # Meta/Llama Configuration
    if os.getenv("REPLICATE_API_KEY"):
        configs.append(
            ModelConfig(
                provider=ModelProvider.META,
                model_name="llama-2-70b",
                temperature=0.7,
                max_tokens=1000,
                custom_params={
                    "version": "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                }
            )
        )
        print("✅ Meta/Llama models configured")
    
    return configs

# ============================================================================
# Prompt Templates
# ============================================================================

def create_evaluation_prompts():
    """Create comprehensive evaluation prompt suite"""
    
    prompts = []
    
    # REASONING PROMPTS
    prompts.append(PromptTemplate(
        id="reasoning_logic_001",
        content="""Solve this logic puzzle step by step:
        
        Three friends - Alice, Bob, and Charlie - each have a different pet (cat, dog, bird) 
        and live in different colored houses (red, blue, green).
        
        Clues:
        1. Alice doesn't live in the red house
        2. The person with the cat lives in the blue house
        3. Bob doesn't have a bird
        4. The person in the green house has a dog
        5. Charlie doesn't live in the blue house
        
        Who has which pet and lives in which house?""",
        category=EvaluationCategory.REASONING,
        evaluation_criteria=["accuracy", "coherence", "completeness"],
        difficulty=3
    ))
    
    prompts.append(PromptTemplate(
        id="reasoning_math_001",
        content="""A train leaves Station A heading towards Station B at 60 mph. 
        Another train leaves Station B heading towards Station A at 80 mph. 
        The stations are 280 miles apart. How long until the trains meet? 
        Show your work step by step.""",
        category=EvaluationCategory.REASONING,
        evaluation_criteria=["accuracy", "clarity", "methodology"],
        difficulty=2
    ))
    
    # CREATIVE PROMPTS
    prompts.append(PromptTemplate(
        id="creative_story_001",
        content="""Write a 200-word story that includes these elements:
        - A mysterious package
        - A talking animal
        - A revelation about the past
        - The story must end with a question""",
        category=EvaluationCategory.CREATIVE,
        evaluation_criteria=["creativity", "coherence", "completeness"],
        difficulty=4
    ))
    
    prompts.append(PromptTemplate(
        id="creative_poetry_001",
        content="""Create a haiku about artificial intelligence that captures 
        both its potential and limitations.""",
        category=EvaluationCategory.CREATIVE,
        evaluation_criteria=["creativity", "relevance", "format_adherence"],
        difficulty=3
    ))
    
    # TECHNICAL PROMPTS
    prompts.append(PromptTemplate(
        id="technical_code_001",
        content="""Write a Python function that finds the k most frequent elements 
        in a list. The function should be efficient and handle edge cases. 
        Include docstring and type hints.""",
        category=EvaluationCategory.CODE_GENERATION,
        evaluation_criteria=["accuracy", "efficiency", "completeness", "best_practices"],
        difficulty=3
    ))
    
    prompts.append(PromptTemplate(
        id="technical_explain_001",
        content="""Explain the concept of gradient descent in machine learning 
        to someone with basic programming knowledge but no ML background. 
        Use an analogy to make it clearer.""",
        category=EvaluationCategory.TECHNICAL,
        evaluation_criteria=["clarity", "accuracy", "accessibility"],
        difficulty=3
    ))
    
    # RESEARCH PROMPTS
    prompts.append(PromptTemplate(
        id="research_analysis_001",
        content="""Analyze the potential impacts of quantum computing on 
        current encryption methods. Consider both near-term (5-10 years) 
        and long-term (20+ years) implications.""",
        category=EvaluationCategory.RESEARCH,
        evaluation_criteria=["depth", "accuracy", "balance", "foresight"],
        difficulty=4
    ))
    
    # BUSINESS PROMPTS
    prompts.append(PromptTemplate(
        id="business_strategy_001",
        content="""A SaaS startup has $1M in funding and needs to decide between:
        A) Investing heavily in product development
        B) Focusing on sales and marketing
        C) Balanced approach
        
        Provide a recommendation with reasoning, considering typical SaaS metrics 
        and growth patterns.""",
        category=EvaluationCategory.BUSINESS,
        evaluation_criteria=["practicality", "reasoning", "completeness"],
        difficulty=4
    ))
    
    # TRANSLATION PROMPTS
    prompts.append(PromptTemplate(
        id="translation_technical_001",
        content="""Translate the following technical description to simple English 
        that a 12-year-old could understand:
        
        'The API endpoint utilizes OAuth 2.0 authentication protocol with JWT tokens 
        for stateless session management, implementing rate limiting through a 
        token bucket algorithm to prevent DDoS attacks.'""",
        category=EvaluationCategory.TRANSLATION,
        evaluation_criteria=["accuracy", "simplicity", "completeness"],
        difficulty=3
    ))
    
    # SUMMARIZATION PROMPTS
    prompts.append(PromptTemplate(
        id="summarization_article_001",
        content="""Summarize this article in 3 sentences:
        
        'Artificial Intelligence has made remarkable progress in recent years, 
        with large language models demonstrating capabilities that were once 
        thought to be decades away. However, these advances come with significant 
        challenges, including concerns about bias, misinformation, and the 
        environmental impact of training large models. Researchers are now 
        focusing on making AI more efficient, interpretable, and aligned with 
        human values, while also exploring new architectures that could overcome 
        current limitations. The next decade will likely see AI becoming more 
        integrated into daily life, requiring careful consideration of ethical 
        and societal implications.'""",
        category=EvaluationCategory.SUMMARIZATION,
        evaluation_criteria=["conciseness", "accuracy", "completeness"],
        difficulty=2
    ))
    
    return prompts

# ============================================================================
# Main Demo Functions
# ============================================================================

async def run_comprehensive_evaluation():
    """Run comprehensive evaluation across all models and prompts"""
    
    print("\n" + "="*60)
    print("🚀 MULTI-MODEL PROMPT ENGINEERING PLATFORM DEMO")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Get model configurations
    configs = get_model_configs()
    if not configs:
        print("❌ No models configured. Please set API keys in .env file")
        return
    
    print(f"\n📊 Configured {len(configs)} models for evaluation")
    
    # Initialize evaluation engine
    print("\n🔧 Initializing evaluation engine...")
    engine = MultiModelEvaluationEngine(configs)
    
    # Create evaluation prompts
    print("📝 Creating evaluation prompts...")
    prompts = create_evaluation_prompts()
    print(f"   Created {len(prompts)} prompts across {len(set(p.category for p in prompts))} categories")
    
    # Run evaluations
    print("\n🏃 Running evaluations...")
    print("   This may take a few minutes depending on API response times")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"   [{i}/{len(prompts)}] Evaluating: {prompt.id}")
        try:
            result = await engine.evaluate_prompt_async(prompt)
            results.append(result)
            
            # Show quick summary
            valid_responses = [r for r in result.responses if not r.error]
            print(f"      ✅ {len(valid_responses)}/{len(result.responses)} successful responses")
            print(f"      🏆 Best model: {result.best_response}")
            print(f"      🤝 Consensus: {result.consensus_score:.2f}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    return results, engine

def analyze_results(results):
    """Analyze and visualize evaluation results"""
    
    print("\n" + "="*60)
    print("📊 ANALYSIS & VISUALIZATION")
    print("="*60)
    
    # Convert results to DataFrame
    data = []
    for result in results:
        for response in result.responses:
            data.append({
                'prompt_id': result.prompt_id,
                'category': result.category.value,
                'model': response.model,
                'provider': response.provider.value,
                'response_time': response.response_time,
                'error': response.error,
                'quality_score': result.scores.get(response.model, {}).get('overall', 0),
                'bias_score': result.bias_analysis.get(response.model, {}).get('overall', 0),
                'consensus_score': result.consensus_score,
                'is_best': result.best_response == response.model
            })
    
    df = pd.DataFrame(data)
    
    # Performance Analysis
    print("\n📈 Performance Analysis")
    analyzer = PerformanceAnalyzer(df)
    
    # Model Rankings
    rankings = analyzer.get_model_rankings()
    print("\n🏆 Model Rankings:")
    print(rankings.to_string())
    
    # Statistical Comparison
    print("\n📊 Statistical Analysis (Quality Scores):")
    stats = analyzer.statistical_comparison('quality_score')
    print(f"   ANOVA F-statistic: {stats['anova']['f_statistic']:.4f}")
    print(f"   ANOVA p-value: {stats['anova']['p_value']:.4f}")
    
    if stats['anova']['p_value'] < 0.05:
        print("   ✅ Significant differences found between models!")
    else:
        print("   ℹ️ No significant differences found")
    
    # Cost Analysis
    print("\n💰 Cost Analysis:")
    cost_analyzer = CostAnalyzer()
    costs = cost_analyzer.cost_quality_analysis(df)
    print(costs[['model', 'total_cost', 'quality_score', 'quality_per_dollar']].to_string())
    
    # Bias Analysis
    print("\n🎯 Bias Analysis Summary:")
    bias_data = {}
    for result in results:
        for model, biases in result.bias_analysis.items():
            if model not in bias_data:
                bias_data[model] = []
            bias_data[model].append(biases.get('overall', 0))
    
    for model, bias_scores in bias_data.items():
        avg_bias = np.mean(bias_scores)
        print(f"   {model}: {avg_bias:.3f}")
    
    # Category Performance
    print("\n📂 Performance by Category:")
    category_perf = df.groupby('category').agg({
        'quality_score': 'mean',
        'response_time': 'mean',
        'error': lambda x: (x.notna()).mean()
    }).round(3)
    print(category_perf.to_string())
    
    return df, analyzer

def generate_reports(df, results):
    """Generate comprehensive reports"""
    
    print("\n" + "="*60)
    print("📄 REPORT GENERATION")
    print("="*60)
    
    # Initialize report generator
    report_gen = ReportGenerator(df)
    
    # Generate HTML report
    print("\n📊 Generating HTML report...")
    html_path = report_gen.generate_html_report("data/reports/evaluation_report.html")
    print(f"   ✅ HTML report saved to: {html_path}")
    
    # Generate Excel report
    print("\n📊 Generating Excel report...")
    excel_path = report_gen.generate_excel_report("data/reports/evaluation_report.xlsx")
    print(f"   ✅ Excel report saved to: {excel_path}")
    
    # Generate JSON results
    print("\n📊 Generating JSON results...")
    json_path = "data/reports/evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(
            [result.__dict__ for result in results],
            f,
            default=str,
            indent=2
        )
    print(f"   ✅ JSON results saved to: {json_path}")
    
    # Generate Markdown summary
    print("\n📊 Generating Markdown summary...")
    md_content = generate_markdown_summary(df, results)
    md_path = "data/reports/evaluation_summary.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"   ✅ Markdown summary saved to: {md_path}")

def generate_markdown_summary(df, results):
    """Generate executive summary in Markdown"""
    
    summary = ["# Multi-Model Evaluation Executive Summary\n\n"]
    summary.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Overview
    summary.append("## 📊 Overview\n\n")
    summary.append(f"- **Total Evaluations:** {len(results)}\n")
    summary.append(f"- **Models Tested:** {df['model'].nunique()}\n")
    summary.append(f"- **Categories Covered:** {df['category'].nunique()}\n")
    summary.append(f"- **Success Rate:** {(1 - df['error'].notna().mean()):.1%}\n\n")
    
    # Top Performers
    summary.append("## 🏆 Top Performers\n\n")
    top_quality = df.groupby('model')['quality_score'].mean().sort_values(ascending=False)
    for i, (model, score) in enumerate(top_quality.head(3).items(), 1):
        summary.append(f"{i}. **{model}**: {score:.3f} quality score\n")
    
    # Key Insights
    summary.append("\n## 💡 Key Insights\n\n")
    
    # Best for each category
    summary.append("### Best Model by Category:\n\n")
    best_by_category = df.groupby('category').apply(
        lambda x: x.loc[x['quality_score'].idxmax()]
    )[['model', 'quality_score']]
    
    for category, row in best_by_category.iterrows():
        summary.append(f"- **{category}**: {row['model']} ({row['quality_score']:.3f})\n")
    
    # Recommendations
    summary.append("\n## 🎯 Recommendations\n\n")
    
    # Quality leader
    quality_leader = top_quality.index[0]
    summary.append(f"1. **For highest quality**: Use {quality_leader}\n")
    
    # Speed leader
    speed_leader = df.groupby('model')['response_time'].mean().idxmin()
    summary.append(f"2. **For fastest responses**: Use {speed_leader}\n")
    
    # Cost efficiency
    cost_analyzer = CostAnalyzer()
    costs = cost_analyzer.cost_quality_analysis(df)
    if not costs.empty:
        best_value = costs.loc[costs['quality_per_dollar'].idxmax(), 'model']
        summary.append(f"3. **For best value**: Use {best_value}\n")
    
    return "".join(summary)

def create_visualizations(df):
    """Create and save visualization plots"""
    
    print("\n" + "="*60)
    print("📈 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Quality Score Distribution
    ax1 = axes[0, 0]
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax1.hist(model_df['quality_score'], alpha=0.5, label=model, bins=10)
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Quality Score Distribution by Model')
    ax1.legend()
    
    # 2. Response Time Comparison
    ax2 = axes[0, 1]
    df.boxplot(column='response_time', by='model', ax=ax2)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Response Time (seconds)')
    ax2.set_title('Response Time Distribution')
    plt.sca(ax2)
    plt.xticks(rotation=45)
    
    # 3. Category Performance Heatmap
    ax3 = axes[1, 0]
    pivot = df.pivot_table(values='quality_score', index='category', columns='model')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Quality Scores by Category and Model')
    
    # 4. Success Rate Bar Chart
    ax4 = axes[1, 1]
    success_rate = df.groupby('model').apply(
        lambda x: 1 - x['error'].notna().mean()
    )
    success_rate.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Success Rate by Model')
    ax4.set_ylim([0, 1])
    plt.sca(ax4)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, v in enumerate(success_rate):
        ax4.text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = "data/reports/evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Visualizations saved to: {plot_path}")
    
    # Show plot (optional)
    # plt.show()
    
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution function"""
    
    try:
        # Run comprehensive evaluation
        results, engine = await run_comprehensive_evaluation()
        
        if not results:
            print("\n❌ No evaluation results generated")
            return
        
        # Analyze results
        df, analyzer = analyze_results(results)
        
        # Generate reports
        generate_reports(df, results)
        
        # Create visualizations
        create_visualizations(df)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)
        
        print("\n📁 Generated Files:")
        print("   - data/reports/evaluation_report.html")
        print("   - data/reports/evaluation_report.xlsx")
        print("   - data/reports/evaluation_results.json")
        print("   - data/reports/evaluation_summary.md")
        print("   - data/reports/evaluation_plots.png")
        
        print("\n🚀 Next Steps:")
        print("   1. Review the HTML report for interactive analysis")
        print("   2. Open Excel report for detailed data exploration")
        print("   3. Run 'streamlit run dashboard.py' for live dashboard")
        print("   4. Check evaluation_summary.md for executive overview")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def run_demo():
    """Run the demo (wrapper for async main)"""
    asyncio.run(main())

if __name__ == "__main__":
    # Run the demo
    run_demo()
