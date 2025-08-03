"""
Command Line Interface for Data Expansion
"""

import click
import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
from data_expander import DataExpansionOrchestrator

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Travel Data Expansion Tool"""
    pass

@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--output-dir', '-o', default='expanded_data', help='Output directory')
@click.option('--scraping/--no-scraping', default=True, help='Enable/disable scraping')
@click.option('--synthetic/--no-synthetic', default=True, help='Enable/disable synthetic generation')
@click.option('--count', '-n', default=5000, help='Target number of synthetic samples')
@click.option('--model', default='llama3-8b-8192', help='Groq model to use for generation')
def expand(config, output_dir, scraping, synthetic, count, model):
    """Run the complete data expansion process"""
    
    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    # Override config with CLI options
    if 'scraping' not in config_data:
        config_data['scraping'] = {}
    if 'synthetic' not in config_data:
        config_data['synthetic'] = {}
    
    config_data['scraping']['enabled'] = scraping
    config_data['synthetic']['enabled'] = synthetic
    config_data['synthetic']['target_count'] = count
    config_data['synthetic']['model_name'] = model

    # Check API keys if synthetic generation is enabled
    if synthetic:
        if not os.getenv('GROQ_API_KEY'):
            click.echo("❌ GROQ_API_KEY not found!")
            click.echo("Please set it in .env file: GROQ_API_KEY=your_key_here")
            click.echo("Or export it: export GROQ_API_KEY=your_key_here")
            return
    
    # Create orchestrator
    orchestrator = DataExpansionOrchestrator(output_dir=output_dir)
    orchestrator.config.update(config_data)
    
    # Run expansion
    click.echo("🚀 Starting data expansion process...")
    try:
        final_file = orchestrator.run_full_expansion()
        click.echo(f"✅ Data expansion completed!")
        click.echo(f"📁 Final dataset: {final_file}")
    except Exception as e:
        click.echo(f"❌ Error during expansion: {e}")
        logger.error(f"Expansion failed: {e}", exc_info=True)

@cli.command()
@click.option('--output-file', '-o', default='scraped_data.jsonl', help='Output file')
@click.option('--reddit-limit', default=200, help='Limit per Reddit subreddit')
def scrape(output_file, reddit_limit):
    """Run only the scraping phase"""
    from scraper import TravelDataScraper
    
    click.echo("🕷️ Starting web scraping...")
    
    scraper = TravelDataScraper()
    count = scraper.run_scraping(output_file)
    
    click.echo(f"✅ Scraping completed! Collected {count} records")
    click.echo(f"📁 Data saved to: {output_file}")

@cli.command()
@click.option('--count', '-n', default=100, help='Number of samples to generate')
@click.option('--output-file', '-o', default='synthetic_data.jsonl', help='Output file')
@click.option('--model', default='llama3-8b-8192', help='Groq model to use')
def generate(count, output_file, model):
    """Run only the synthetic generation phase using LangChain + Groq"""
    from synthetic_generator import SyntheticDataGenerator

    # Check API keys
    if not os.getenv('GROQ_API_KEY'):
        click.echo("❌ GROQ_API_KEY not found!")
        click.echo("Please set it in .env file: GROQ_API_KEY=your_key_here")
        return

    click.echo(f"🤖 Generating {count} synthetic samples using LangChain + Groq ({model})...")

    generator = SyntheticDataGenerator(model_name=model)
    generated_count = generator.run_generation(count, output_file)

    click.echo(f"✅ Generation completed! Created {generated_count} records")
    click.echo(f"📁 Data saved to: {output_file}")

@cli.command()
@click.argument('input_file')
@click.option('--output-file', '-o', help='Output file (default: cleaned_<input_file>)')
def clean(input_file, output_file):
    """Clean and deduplicate a dataset"""
    from data_expander import DataQualityManager
    import json
    
    if not output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"cleaned_{input_path.name}"
    
    click.echo(f"🧹 Cleaning dataset: {input_file}")
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Clean data
    quality_manager = DataQualityManager()
    clean_data = quality_manager.deduplicate_and_validate(data)
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in clean_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    click.echo(f"✅ Cleaning completed!")
    click.echo(f"📊 Original: {len(data)} records")
    click.echo(f"📊 Cleaned: {len(clean_data)} records")
    click.echo(f"📁 Saved to: {output_file}")

@cli.command()
@click.argument('dataset_file')
def analyze(dataset_file):
    """Analyze a dataset and show statistics"""
    import json
    from collections import Counter
    
    click.echo(f"📊 Analyzing dataset: {dataset_file}")
    
    data = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not data:
        click.echo("❌ No valid data found!")
        return
    
    # Basic statistics
    click.echo(f"\n📈 Dataset Statistics:")
    click.echo(f"Total records: {len(data)}")
    
    # Prompt lengths
    prompt_lengths = [len(item.get('prompt', '')) for item in data]
    click.echo(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.1f} chars")
    
    # Response lengths
    response_lengths = [len(item.get('response', '')) for item in data]
    click.echo(f"Average response length: {sum(response_lengths) / len(response_lengths):.1f} chars")
    
    # Sources
    sources = [item.get('entities', {}).get('source', 'unknown') for item in data]
    source_counts = Counter(sources)
    
    click.echo(f"\n📋 Sources:")
    for source, count in source_counts.most_common():
        click.echo(f"  {source}: {count}")
    
    # Categories
    categories = [item.get('entities', {}).get('category', 'unknown') for item in data]
    category_counts = Counter(categories)
    
    click.echo(f"\n🏷️ Categories:")
    for category, count in category_counts.most_common():
        click.echo(f"  {category}: {count}")

@cli.command()
def setup():
    """Setup the data expansion environment"""
    click.echo("🔧 Setting up data expansion environment...")
    
    # Create directories
    dirs = ['expanded_data', 'logs', 'backups']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        click.echo(f"📁 Created directory: {dir_name}")
    
    # Check dependencies
    try:
        import requests
        import bs4
        import pandas
        click.echo("✅ Core dependencies installed")
    except ImportError as e:
        click.echo(f"❌ Missing dependency: {e}")
        click.echo("Run: pip install -r requirements.txt")
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        click.echo("✅ .env file found")
    else:
        click.echo("⚠️ .env file not found - creating template...")
        with open('.env', 'w') as f:
            f.write("""# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Configuration (optional alternative)
# OPENAI_API_KEY=your_openai_api_key_here

# Data Expansion Settings
USE_GROQ=true
MAX_RETRIES=3
REQUEST_DELAY=1.0

# Output Settings
OUTPUT_DIR=expanded_data
BACKUP_ENABLED=true
""")
        click.echo("📝 Created .env template - please add your API keys")

    # Check API keys
    if os.getenv('GROQ_API_KEY') and os.getenv('GROQ_API_KEY') != 'your_groq_api_key_here':
        click.echo("✅ GROQ_API_KEY configured")
    else:
        click.echo("⚠️ GROQ_API_KEY not configured in .env file")

    if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here':
        click.echo("✅ OPENAI_API_KEY configured")
    else:
        click.echo("⚠️ OPENAI_API_KEY not configured in .env file")
    
    click.echo("\n🎉 Setup completed!")

if __name__ == '__main__':
    cli()
