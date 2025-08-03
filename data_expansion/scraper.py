"""
Advanced Travel Data Scraper
Scrapes travel Q&A from multiple sources to expand the dataset
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TravelQA:
    question: str
    answer: str
    source: str
    category: str
    location: str = ""
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.generate_id(),
            "prompt": self.question,
            "response": self.answer,
            "entities": {
                "source": self.source,
                "category": self.category,
                "location": self.location,
                "tags": self.tags or []
            }
        }
    
    def generate_id(self) -> str:
        """Generate unique ID based on content"""
        content = f"{self.question}{self.answer}{self.source}"
        return hashlib.md5(content.encode()).hexdigest()

class TravelDataScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_data = []
        self.seen_ids = set()
        
    def add_delays(self, min_delay=1, max_delay=3):
        """Add random delays to avoid being blocked"""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        return text
    
    def scrape_reddit_travel(self, subreddits=['travel', 'solotravel', 'backpacking'], limit=100):
        """Scrape travel Q&A from Reddit"""
        logger.info("Scraping Reddit travel data...")
        
        for subreddit in subreddits:
            try:
                # Use Reddit's JSON API
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    for post in posts:
                        post_data = post.get('data', {})
                        
                        # Skip if not a text post or no selftext
                        if not post_data.get('selftext') or post_data.get('selftext') == '[removed]':
                            continue
                        
                        question = self.clean_text(post_data.get('title', ''))
                        answer = self.clean_text(post_data.get('selftext', ''))
                        
                        if len(question) > 20 and len(answer) > 50:
                            qa = TravelQA(
                                question=question,
                                answer=answer,
                                source=f"reddit_r_{subreddit}",
                                category="general_travel",
                                tags=[subreddit, "reddit"]
                            )
                            
                            if qa.generate_id() not in self.seen_ids:
                                self.scraped_data.append(qa)
                                self.seen_ids.add(qa.generate_id())
                
                self.add_delays(2, 4)  # Longer delays for Reddit
                
            except Exception as e:
                logger.error(f"Error scraping Reddit r/{subreddit}: {e}")
    
    def scrape_travel_forums(self):
        """Scrape travel forums and Q&A sites"""
        logger.info("Scraping travel forums...")
        
        # TripAdvisor forum URLs (example structure)
        tripadvisor_forums = [
            "https://www.tripadvisor.com/ShowForum-g1-i10702-General_Travel.html",
            "https://www.tripadvisor.com/ShowForum-g1-i10703-Budget_Travel.html",
        ]
        
        for forum_url in tripadvisor_forums:
            try:
                response = self.session.get(forum_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find forum topics (this is a simplified example)
                    topics = soup.find_all('div', class_='topic')
                    
                    for topic in topics[:20]:  # Limit to avoid overloading
                        title_elem = topic.find('a')
                        if title_elem:
                            topic_title = self.clean_text(title_elem.get_text())
                            topic_url = urljoin(forum_url, title_elem.get('href', ''))
                            
                            # Get the topic content
                            topic_content = self.scrape_forum_topic(topic_url)
                            
                            if topic_content and len(topic_title) > 20:
                                qa = TravelQA(
                                    question=topic_title,
                                    answer=topic_content,
                                    source="tripadvisor_forum",
                                    category="travel_advice",
                                    tags=["tripadvisor", "forum"]
                                )
                                
                                if qa.generate_id() not in self.seen_ids:
                                    self.scraped_data.append(qa)
                                    self.seen_ids.add(qa.generate_id())
                
                self.add_delays(3, 5)
                
            except Exception as e:
                logger.error(f"Error scraping TripAdvisor forum {forum_url}: {e}")
    
    def scrape_forum_topic(self, topic_url: str) -> str:
        """Scrape individual forum topic content"""
        try:
            response = self.session.get(topic_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the main post content (simplified)
                content_elem = soup.find('div', class_='post-content')
                if content_elem:
                    return self.clean_text(content_elem.get_text())
                
        except Exception as e:
            logger.error(f"Error scraping topic {topic_url}: {e}")
        
        return ""
    
    def scrape_travel_blogs(self):
        """Scrape travel blogs for Q&A content"""
        logger.info("Scraping travel blogs...")
        
        # Popular travel blog URLs
        blog_urls = [
            "https://www.nomadicmatt.com/travel-blog/",
            "https://www.lonelyplanet.com/articles",
        ]
        
        for blog_url in blog_urls:
            try:
                response = self.session.get(blog_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find article links
                    article_links = soup.find_all('a', href=True)
                    
                    for link in article_links[:10]:  # Limit articles
                        href = link.get('href')
                        if href and ('travel' in href.lower() or 'guide' in href.lower()):
                            article_url = urljoin(blog_url, href)
                            article_content = self.scrape_blog_article(article_url)
                            
                            if article_content:
                                title = self.clean_text(link.get_text())
                                if len(title) > 20 and len(article_content) > 100:
                                    qa = TravelQA(
                                        question=f"Tell me about: {title}",
                                        answer=article_content,
                                        source="travel_blog",
                                        category="travel_guide",
                                        tags=["blog", "guide"]
                                    )
                                    
                                    if qa.generate_id() not in self.seen_ids:
                                        self.scraped_data.append(qa)
                                        self.seen_ids.add(qa.generate_id())
                
                self.add_delays(2, 4)
                
            except Exception as e:
                logger.error(f"Error scraping blog {blog_url}: {e}")
    
    def scrape_blog_article(self, article_url: str) -> str:
        """Scrape individual blog article content"""
        try:
            response = self.session.get(article_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article content (common selectors)
                content_selectors = [
                    'article', '.article-content', '.post-content', 
                    '.entry-content', 'main', '.content'
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        # Extract text and limit length
                        text = self.clean_text(content_elem.get_text())
                        return text[:2000]  # Limit to 2000 chars
                
        except Exception as e:
            logger.error(f"Error scraping article {article_url}: {e}")
        
        return ""
    
    def save_data(self, filename: str):
        """Save scraped data to JSONL file"""
        logger.info(f"Saving {len(self.scraped_data)} records to {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            for qa in self.scraped_data:
                json.dump(qa.to_dict(), f, ensure_ascii=False)
                f.write('\n')
    
    def run_scraping(self, output_file: str = "scraped_travel_data.jsonl"):
        """Run all scraping methods"""
        logger.info("Starting travel data scraping...")
        
        # Run all scraping methods
        self.scrape_reddit_travel()
        self.scrape_travel_forums()
        self.scrape_travel_blogs()
        
        # Save results
        self.save_data(output_file)
        
        logger.info(f"Scraping completed! Collected {len(self.scraped_data)} travel Q&A pairs")
        return len(self.scraped_data)

if __name__ == "__main__":
    scraper = TravelDataScraper()
    scraper.run_scraping()
