# Travel Data Expansion Suite

Advanced data expansion toolkit for travel AI systems using **LangChain + Groq**. This suite combines web scraping, synthetic data generation, and quality management to expand your travel dataset from 1K to 10K+ high-quality samples.

## 🚀 Features

### Web Scraping
- **Reddit Travel Communities**: Scrapes r/travel, r/solotravel, r/backpacking, etc.
- **Travel Forums**: TripAdvisor, Lonely Planet forums
- **Travel Blogs**: Popular travel blogs and guides
- **Smart Rate Limiting**: Respects robots.txt and implements delays
- **Content Deduplication**: Prevents duplicate content

### Synthetic Data Generation
- **LangChain + Groq**: Uses LangChain framework with Groq for fast, cost-effective generation
- **Multiple Models**: Support for llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
- **Diverse Scenarios**: 50+ destinations, 25+ traveler types
- **Realistic Parameters**: Budget ranges, trip durations, interests
- **Quality Control**: Automated validation and filtering

### Data Quality Management
- **Deduplication**: Content-based hash deduplication
- **Quality Filtering**: Length, keyword, and relevance checks
- **Format Validation**: Ensures consistent data structure
- **Statistics Tracking**: Detailed quality metrics

## 📦 Installation

1. **Clone and Setup**:
```bash
cd data_expansion
pip install -r requirements.txt
python cli.py setup
```

2. **Configure Groq API Key** (for LangChain + Groq synthetic generation):

Edit the `.env` file:
```bash
# Groq API Configuration (using LangChain)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

Or set environment variable:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

## 🎯 Quick Start

### Complete Data Expansion
```bash
# Run full expansion (scraping + synthetic generation)
python cli.py expand --count 5000 --output-dir expanded_data

# With custom configuration
python cli.py expand --config custom_config.yaml --count 10000
```

### Individual Components

**Web Scraping Only**:
```bash
python cli.py scrape --output-file scraped_travel.jsonl --reddit-limit 300
```

**Synthetic Generation Only**:
```bash
# Using LangChain + Groq (default model)
python cli.py generate --count 1000

# Using specific Groq model
python cli.py generate --count 1000 --model llama3-70b-8192
python cli.py generate --count 1000 --model mixtral-8x7b-32768
```

**Data Cleaning**:
```bash
python cli.py clean input_data.jsonl --output-file cleaned_data.jsonl
```

**Dataset Analysis**:
```bash
python cli.py analyze final_dataset.jsonl
```

## 📊 Expected Output

### Data Volume
- **Scraping**: 500-2000 records (depending on sources)
- **Synthetic**: 1000-10000+ records (configurable)
- **Final Dataset**: 10K+ high-quality travel Q&A pairs

### Data Quality
- Minimum 20 characters for questions
- Minimum 100 characters for answers
- Travel-keyword validation
- Duplicate removal
- Source attribution

### File Formats
- **JSONL**: Primary format for ML training
- **CSV**: For analysis and inspection
- **Statistics**: JSON metadata files

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
scraping:
  enabled: true
  sources:
    reddit:
      subreddits: ["travel", "solotravel", "backpacking"]
      limit_per_subreddit: 200

synthetic_generation:
  enabled: true
  target_count: 5000
  use_groq: true
  temperature: 0.7

quality_control:
  min_prompt_length: 20
  min_response_length: 100
  travel_keywords: ["travel", "trip", "vacation", ...]
```

## 🔧 Advanced Usage

### Custom Scraping Sources
```python
from scraper import TravelDataScraper

scraper = TravelDataScraper()
# Add custom scraping methods
scraper.scrape_custom_source("https://example-travel-site.com")
```

### Custom Synthetic Scenarios
```python
from synthetic_generator import SyntheticDataGenerator, TravelScenario

generator = SyntheticDataGenerator()
custom_scenario = TravelScenario(
    destination="Tokyo",
    duration_days=14,
    budget_usd=3000,
    traveler_type="digital nomad",
    interests=["technology", "food", "culture"]
)
```

### Quality Control Customization
```python
from data_expander import DataQualityManager

quality_manager = DataQualityManager()
# Customize quality checks
quality_manager.min_prompt_length = 30
quality_manager.travel_keywords.extend(["adventure", "explore"])
```

## 📈 Performance & Scaling

### Rate Limiting
- Reddit: 2-4 seconds between requests
- Forums: 3-5 seconds between requests
- Blogs: 2-4 seconds between requests

### API Usage
- **Groq**: ~$0.10 per 1000 samples (recommended)
- **OpenAI GPT-4**: ~$2.00 per 1000 samples

### Processing Speed
- Scraping: ~100-500 records/hour
- Synthetic: ~50-200 records/hour (API dependent)
- Cleaning: ~10,000 records/minute

## 🛡️ Best Practices

### Ethical Scraping
- Respects robots.txt
- Implements proper delays
- Uses appropriate user agents
- Doesn't overload servers

### Data Quality
- Multiple validation layers
- Content deduplication
- Source attribution
- Quality metrics tracking

### API Management
- Rate limiting compliance
- Error handling and retries
- Cost monitoring
- Backup generation methods

## 📁 Output Structure

```
expanded_data/
├── expanded_travel_data_20240119_143022.jsonl  # Final dataset
├── expanded_travel_data_20240119_143022.csv    # CSV version
├── expansion_stats_20240119_143022.json        # Statistics
├── scraped_raw.jsonl                           # Raw scraped data
├── synthetic_raw.jsonl                         # Raw synthetic data
└── backups/                                    # Periodic backups
```

## 🔍 Quality Metrics

The system tracks:
- Total records processed
- Duplicates removed
- Quality filters applied
- Source distribution
- Content length statistics
- Keyword coverage

## 🚨 Troubleshooting

### Common Issues

**API Key Errors**:
```bash
# Check if keys are set
echo $GROQ_API_KEY
echo $OPENAI_API_KEY
```

**Scraping Blocked**:
- Increase delays in config
- Check robots.txt compliance
- Use different user agents

**Quality Issues**:
- Adjust quality thresholds in config
- Review travel keywords list
- Check source-specific filters

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python cli.py expand --count 100
```

## 📚 Next Steps

After expansion:
1. **Data Analysis**: Use `analyze` command to review quality
2. **Fine-tuning Preparation**: Format for your ML framework
3. **Model Training**: Use expanded dataset for fine-tuning
4. **Evaluation**: Test model performance on held-out data

## 🤝 Contributing

1. Add new scraping sources in `scraper.py`
2. Enhance synthetic scenarios in `synthetic_generator.py`
3. Improve quality filters in `data_expander.py`
4. Submit pull requests with tests

## 📄 License

MIT License - see LICENSE file for details.
