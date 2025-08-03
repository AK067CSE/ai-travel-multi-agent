"""
Quick analysis of original data.jsonl file
"""

import json
from pathlib import Path

def analyze_original_data():
    """Analyze the original data.jsonl file"""
    
    data_file = Path("../data.jsonl")
    
    if not data_file.exists():
        print("❌ Original data.jsonl not found")
        return
    
    print("📊 Analyzing original data.jsonl...")
    print("=" * 50)
    
    valid_records = 0
    invalid_records = 0
    sample_records = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    valid_records += 1
                    
                    # Collect first 3 samples
                    if len(sample_records) < 3:
                        sample_records.append(record)
                        
                except json.JSONDecodeError as e:
                    invalid_records += 1
                    if invalid_records <= 5:  # Show first 5 errors
                        print(f"⚠️ Line {line_num}: {e}")
    
    print(f"\n📈 Statistics:")
    print(f"✅ Valid records: {valid_records}")
    print(f"❌ Invalid records: {invalid_records}")
    print(f"📊 Total lines processed: {valid_records + invalid_records}")
    
    if sample_records:
        print(f"\n📋 Sample Record Structure:")
        sample = sample_records[0]
        print(f"Keys: {list(sample.keys())}")
        
        if 'prompt' in sample:
            print(f"Prompt length: {len(sample['prompt'])} chars")
        if 'response' in sample:
            print(f"Response length: {len(sample['response'])} chars")
        
        print(f"\n📝 First Record Preview:")
        for key, value in sample.items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n✅ Original data analysis complete!")
    return valid_records, invalid_records

if __name__ == "__main__":
    analyze_original_data()
