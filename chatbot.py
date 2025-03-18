import json
import re
import random
import logging
import os
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Simple text processing functions that don't rely on NLTK
def simple_tokenize(text):
    """Simple tokenization without relying on NLTK"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and split on spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace
    return text.split()

# Common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't"
}

# User intent categories
INTENT_CATEGORIES = {
    'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
    'farewell': ['bye', 'goodbye', 'see you', 'talk to you later', 'thanks', 'thank you'],
    'info_request': ['what is', 'how does', 'tell me about', 'explain', 'information on', 'details about'],
    'price_inquiry': ['how much', 'price', 'cost', 'pricing', 'fee', 'rates', 'charge'],
    'discount_inquiry': ['discount', 'offer', 'deal', 'promotion', 'cheaper', 'affordable', 'expensive', 'package', 'any offer'],
    'capability_question': ['can you', 'are you able', 'do you offer', 'is it possible', 'ability to'],
    'comparison': ['versus', 'compared to', 'difference between', 'better than', 'worse than', 'vs'],
    'opinion': ['what do you think', 'recommend', 'suggest', 'advise', 'opinion on', 'best'],
    'problem': ['issue', 'problem', 'trouble', 'error', 'difficulty', 'challenge', 'help'],
    'clarification': ['what do you mean', 'confused', 'don\'t understand', 'clarify', 'elaborate']
}

class ConversationContext:
    """Stores conversation history and context for more coherent responses"""
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        self.user_interests = Counter()
        self.session_start = time.time()
        self.topic_focus = None
        
    def add_exchange(self, user_input, bot_response):
        """Add a conversation exchange to history"""
        self.history.append({
            'user_input': user_input,
            'bot_response': bot_response,
            'timestamp': time.time()
        })
        
        # Keep history within max size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update user interests based on keywords in input
        tokens = simple_tokenize(user_input)
        interests = [token for token in tokens if len(token) > 3 and token not in STOPWORDS]
        self.user_interests.update(interests)
        
    def get_dominant_topic(self):
        """Get the most common topic from recent conversation"""
        if not self.user_interests:
            return None
        return self.user_interests.most_common(1)[0][0]
    
    def has_mentioned(self, term, within_exchanges=None):
        """Check if a term has been mentioned in the conversation history"""
        if not self.history:
            return False
            
        history_to_check = self.history
        if within_exchanges is not None:
            history_to_check = self.history[-within_exchanges:]
            
        for exchange in history_to_check:
            if term.lower() in exchange['user_input'].lower():
                return True
        return False
    
    def get_conversation_duration(self):
        """Get the duration of the current conversation in seconds"""
        return time.time() - self.session_start

class AphatorChatbot:
    def __init__(self):
        """Initialize the Aphator Tech Chatbot with company data."""
        # Use the predefined stopwords instead of NLTK
        self.stop_words = STOPWORDS
        self.company_data = self.load_company_data()
        self.vectorizer = TfidfVectorizer()
        
        # Initialize conversation context for tracking user intent and history
        self.context = ConversationContext(max_history=10)
        
        # Track learning from conversations
        self.learned_responses = {}
        self.topic_keywords = defaultdict(list)
        
        # Populate keyword-to-topic mapping for better understanding
        self.topic_keywords['blockchain'] = ['blockchain', 'smart contract', 'ethereum', 'token', 'web3', 'crypto', 'dapp', 'decentralized']
        self.topic_keywords['crypto_trading'] = ['trading', 'investment', 'portfolio', 'market', 'bot', 'price', 'coin', 'token', 'exchange']
        self.topic_keywords['application'] = ['app', 'application', 'software', 'mobile', 'web', 'development', 'platform', 'website', 'interface']
        self.topic_keywords['security'] = ['security', 'protection', 'hack', 'risk', 'breach', 'safe', 'secure', 'vulnerability', 'audit']
        self.topic_keywords['nft'] = ['nft', 'collectible', 'token', 'art', 'marketplace', 'mint', 'royalty', 'collection']
        
        # Engagement strategies to keep users interested
        self.engagement_prompts = [
            "Would you like to know more about how our {topic} solutions can benefit your business?",
            "Have you considered implementing {topic} technology in your organization?",
            "Many of our clients have seen significant results with our {topic} services. Would you like to hear about some case studies?",
            "What specific aspects of {topic} are you most interested in?",
            "Is there a particular challenge with {topic} that you're looking to solve?"
        ]
        
        # Prepare vectorizer with existing data
        self.prepare_vectors()
        
        # Greetings and fallbacks
        self.greetings = [
            "Hello! Welcome to Aphator Tech. How can I assist you today?",
            "Hi there! I'm the Aphator Tech assistant. What can I help you with?",
            "Welcome to Aphator Tech! I'm here to answer your questions about our crypto and tech services."
        ]
        
        self.farewells = [
            "Thank you for chatting with Aphator Tech. Have a great day!",
            "It was a pleasure assisting you. If you have more questions, feel free to ask anytime!",
            "Thank you for considering Aphator Tech. We hope to serve you soon!"
        ]
        
        self.fallbacks = [
            "I'm not sure I understand that query. Could you please rephrase it or ask about our crypto or tech services?",
            "I don't have information on that specific topic. Would you like to know about our crypto solutions or tech services instead?",
            "I'm still learning and don't have an answer for that yet. Can I tell you about Aphator Tech's expertise in blockchain or development services?"
        ]
        
        logger.info("Aphator Chatbot initialized successfully")
    
    def load_company_data(self):
        """Load company data from JSON file."""
        try:
            data_file = Path(os.path.dirname(os.path.abspath(__file__))) / 'company_data.json'
            
            # Load text data from data/companydata.txt if it exists
            text_data_file = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'companydata.txt'
            if text_data_file.exists():
                logger.info(f"Found additional company data at {text_data_file}")
                # We'll use this data later for enhanced responses
                with open(text_data_file, 'r') as tf:
                    self.text_company_data = tf.read()
            else:
                self.text_company_data = None
                
            # Load primary JSON data
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading company data: {str(e)}")
            return {
                "company_info": {
                    "name": "Aphator Tech",
                    "description": "A leading provider of crypto and tech solutions",
                    "expertise": ["Blockchain Development", "Cryptocurrency Consulting", "Tech Infrastructure", "Software Development"]
                },
                "faqs": [],
                "services": [],
                "products": []
            }
    
    def prepare_vectors(self):
        """Prepare TF-IDF vectors from company data."""
        # Extract all content for vectorization
        self.corpus = []
        self.responses = []
        
        # Add FAQ data
        for item in self.company_data.get("faqs", []):
            self.corpus.append(item.get("question", ""))
            self.responses.append(item.get("answer", ""))
        
        # Add service descriptions
        for service in self.company_data.get("services", []):
            self.corpus.append(service.get("name", "") + " " + service.get("description", ""))
            self.responses.append(self._generate_service_response(service))
        
        # Add product information
        for product in self.company_data.get("products", []):
            self.corpus.append(product.get("name", "") + " " + product.get("description", ""))
            self.responses.append(self._generate_product_response(product))
        
        # Add company info
        company_info = self.company_data.get("company_info", {})
        company_text = f"{company_info.get('name', '')} {company_info.get('description', '')} {' '.join(company_info.get('expertise', []))}"
        self.corpus.append("Tell me about Aphator Tech")
        self.responses.append(self._generate_company_response(company_info))
        
        self.corpus.append("What does Aphator Tech do")
        self.responses.append(self._generate_company_response(company_info))
        
        # Add application development specific responses
        self.corpus.append("Can you help me launch an application")
        self.responses.append("Yes, Aphator Tech specializes in developing and launching applications for various platforms. " +
                             "Our software development team can help you create mobile apps (iOS/Android), web applications, " +
                             "and enterprise software solutions. Pricing starts at $10,000 for full applications, with " +
                             "the exact cost depending on complexity and requirements. Would you like to discuss your specific app idea?")
        
        self.corpus.append("How much does app development cost")
        self.responses.append("At Aphator Tech, application development costs start at $10,000 for basic applications. " +
                             "The final price depends on factors like complexity, features, platform requirements, and timeline. " +
                             "We offer both native app development for iOS/Android and cross-platform solutions. " +
                             "We'd be happy to provide a detailed quote after understanding your specific requirements.")
        
        self.corpus.append("What kind of applications can you build")
        self.responses.append("Aphator Tech can develop a wide range of applications including: mobile apps for iOS and Android, " +
                             "web applications, enterprise software, blockchain dApps, cryptocurrency trading platforms, " +
                             "NFT marketplaces, fintech solutions, and custom software for specific business needs. " +
                             "Our development team is skilled in multiple technologies and frameworks to create reliable, " +
                             "scalable, and secure applications tailored to your requirements.")
                             
        # If we have content, create the vectors
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            logger.info(f"Prepared vectors for {len(self.corpus)} items")
        else:
            logger.warning("No content available for vectorization")
            self.tfidf_matrix = None
    
    def _preprocess_text(self, text):
        """Preprocess text by tokenizing and removing stopwords."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize using our simple tokenizer and remove stopwords
        tokens = simple_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        return " ".join(tokens)
    
    def _generate_service_response(self, service):
        """Generate a formatted response about a service."""
        return f"Aphator Tech offers {service.get('name', '')}. {service.get('description', '')}. The pricing starts at {service.get('pricing', 'competitive rates')}."
    
    def _generate_product_response(self, product):
        """Generate a formatted response about a product."""
        return f"Aphator Tech's {product.get('name', '')} is {product.get('description', '')}. It's available at {product.get('pricing', 'competitive rates')}."
    
    def _generate_company_response(self, company_info):
        """Generate a formatted response about the company."""
        expertise = ", ".join(company_info.get('expertise', []))
        return f"{company_info.get('name', 'Aphator Tech')} is {company_info.get('description', 'a leading provider of crypto and tech solutions')}. We specialize in {expertise}."
    
    def get_response(self, user_input):
        """Get a response based on user input with context awareness and learning."""
        # Check for learned responses first
        tokens = simple_tokenize(user_input)
        key_terms = [token for token in tokens if len(token) > 3 and token not in self.stop_words]
        
        if key_terms and len(key_terms) >= 2:
            # Look for matching patterns in learned responses
            pattern = ' '.join(sorted(key_terms[:3]))  # Use up to 3 key terms
            if pattern in self.learned_responses:
                logger.debug(f"Using learned response for pattern: {pattern}")
                learned_response = self.learned_responses[pattern]
                # Add engagement and learn from this interaction
                final_response = self._add_engagement_prompt(learned_response)
                self._learn_from_interaction(user_input, final_response)
                return final_response
                
        # Detect user intent
        intent = self._detect_intent(user_input)
        sentiment = self._analyze_sentiment(user_input)
        lower_input = user_input.lower()
        
        # Check for greetings
        if self._is_greeting(user_input):
            greeting = random.choice(self.greetings)
            # If this isn't the first interaction, personalize based on history
            if self.context.history:
                dominant_topic = self.context.get_dominant_topic()
                if dominant_topic:
                    greeting += f" I see you're interested in {dominant_topic}. How can I help you with that today?"
            
            self._learn_from_interaction(user_input, greeting)
            return greeting
        
        # Check for farewells
        if self._is_farewell(user_input):
            farewell = random.choice(self.farewells)
            self._learn_from_interaction(user_input, farewell)
            return farewell
        
        # General help response for broad questions
        if any(phrase in lower_input for phrase in ["how can you help", "what can you do", "assist me", "help me with"]):
            help_response = ("I can help you with information about Aphator Tech's products and services. " +
                           "We specialize in blockchain development, cryptocurrency trading solutions, Web3 integration, " +
                           "NFT development, cybersecurity for crypto, and custom software development. " +
                           "How can I assist you specifically today?")
            self._learn_from_interaction(user_input, help_response)
            return help_response

        # Handle "I want to" type requests with context awareness
        if "want to" in lower_input:
            if any(word in lower_input for word in ["build", "create", "develop", "launch"]):
                if any(word in lower_input for word in ["app", "application", "website", "software", "platform"]):
                    app_response = ("Aphator Tech can help you develop your application or software project. Our development team " +
                                  "creates custom solutions for various platforms including web, mobile, and enterprise systems. " +
                                  "Application development starts at $10,000, with the exact price depending on your specific requirements. " +
                                  "Would you like to tell me more about your project?")
                    self._learn_from_interaction(user_input, app_response)
                    return self._add_engagement_prompt(app_response, "application")
                    
                if any(word in lower_input for word in ["blockchain", "smart contract", "dapp", "token"]):
                    blockchain_response = ("Aphator Tech specializes in blockchain development. We can help you build custom blockchain solutions, " +
                                         "smart contracts, DApps, or handle tokenization services. Our blockchain services start at $5,000, " +
                                         "and we work with various blockchain protocols including Ethereum, Solana, and Binance Smart Chain. " +
                                         "What kind of blockchain project are you looking to develop?")
                    self._learn_from_interaction(user_input, blockchain_response)
                    return self._add_engagement_prompt(blockchain_response, "blockchain")
                    
                if any(word in lower_input for word in ["trade", "trading", "invest", "investment"]):
                    trading_response = ("For cryptocurrency trading and investment solutions, Aphator Tech offers custom trading bots, " +
                                      "market analysis tools, and portfolio management systems. Our TradeBotX product ($59.99/month) provides " +
                                      "automated trading capabilities with strategy building and risk management features. " +
                                      "Would you like more information about our trading solutions?")
                    self._learn_from_interaction(user_input, trading_response)
                    return self._add_engagement_prompt(trading_response, "crypto_trading")
                    
        # Check for more specific queries that can be handled directly from text data
        if hasattr(self, 'text_company_data') and self.text_company_data:
            # Check for product specific questions with context awareness
            if "cryptotracker" in lower_input or "crypto tracker" in lower_input:
                tracker_response = ("CryptoTracker Pro is Aphator Tech's all-in-one cryptocurrency portfolio tracking and management solution. " +
                                  "It offers multi-wallet support, real-time price updates, performance analytics, and tax reporting tools. " +
                                  "It's available for $29.99/month. Would you like more details about its features?")
                self._learn_from_interaction(user_input, tracker_response)
                return self._add_engagement_prompt(tracker_response, "crypto_trading")
            
            if "blocksecure" in lower_input or "block secure" in lower_input:
                security_response = ("BlockSecure is our comprehensive security solution for blockchain assets. It includes multi-signature " +
                                   "wallet implementation, automated security audits, threat detection and alerts, and secure backup solutions. " +
                                   "Available for $49.99/month. Would you like to learn more about how it can protect your crypto assets?")
                self._learn_from_interaction(user_input, security_response)
                return self._add_engagement_prompt(security_response, "security")
            
            if "smartcontract" in lower_input or "smart contract builder" in lower_input or "smart contract" in lower_input:
                contract_response = ("SmartContract Builder is Aphator Tech's no-code platform for creating smart contracts. It includes a template " +
                                   "library, visual contract builder, automated testing, and one-click deployment capabilities. " +
                                   "It's priced at $39.99/month. Would you like more information about how it simplifies smart contract development?")
                self._learn_from_interaction(user_input, contract_response)
                return self._add_engagement_prompt(contract_response, "blockchain")
            
            if "tradebotx" in lower_input or "trade bot" in lower_input or "trading bot" in lower_input:
                bot_response = ("TradeBotX is our automated cryptocurrency trading bot featuring a strategy builder, multi-exchange support, " +
                              "backtesting capabilities, and risk management tools. It's available for $59.99/month. " +
                              "Would you like to know more about how it can optimize your trading strategies?")
                self._learn_from_interaction(user_input, bot_response)
                return self._add_engagement_prompt(bot_response, "crypto_trading")
            
            # Check for services with context awareness
            if "blockchain" in lower_input or "dapp" in lower_input:
                blockchain_service = ("Aphator Tech offers comprehensive blockchain development services, including custom blockchain solutions, " +
                                    "smart contract creation and auditing, DApp development, tokenization services, and blockchain integration " +
                                    "with existing systems. Our team has extensive experience building secure and efficient blockchain applications " +
                                    "tailored to specific business needs.")
                self._learn_from_interaction(user_input, blockchain_service)
                return self._add_engagement_prompt(blockchain_service, "blockchain")
            
            if any(word in lower_input for word in ["trading", "crypto trading", "cryptocurrency trading"]):
                trading_service = ("Our Cryptocurrency Trading Solutions include custom trading bots, market analysis tools, portfolio management " +
                                 "systems, trading strategy implementation, and real-time market data integration. We can help optimize " +
                                 "your trading operations with cutting-edge technology and expertise in cryptocurrency markets.")
                self._learn_from_interaction(user_input, trading_service)
                return self._add_engagement_prompt(trading_service, "crypto_trading")
            
            if "web3" in lower_input:
                web3_service = ("Aphator Tech specializes in Web3 integration services, including wallet integration, decentralized authentication, " +
                              "smart contract interaction, cross-chain compatibility, and gas optimization. We can help connect your " +
                              "existing platforms to the decentralized web and blockchain ecosystems.")
                self._learn_from_interaction(user_input, web3_service)
                return self._add_engagement_prompt(web3_service, "blockchain")
            
            if "nft" in lower_input:
                nft_service = ("Our NFT development services include NFT marketplace development, collection smart contracts, minting tools " +
                             "and platforms, metadata management, and royalty implementation. We can help you create, launch, and " +
                             "manage NFT projects from concept to deployment.")
                self._learn_from_interaction(user_input, nft_service)
                return self._add_engagement_prompt(nft_service, "nft")
            
            if "security" in lower_input or "cybersecurity" in lower_input or "secure" in lower_input:
                security_service = ("Aphator Tech provides specialized Cybersecurity for Crypto services, including wallet security audits, " +
                                  "smart contract vulnerability analysis, penetration testing, security protocol implementation, and " +
                                  "secure key management solutions. We help protect your digital assets with advanced security measures.")
                self._learn_from_interaction(user_input, security_service)
                return self._add_engagement_prompt(security_service, "security")
            
            if any(word in lower_input for word in ["contact", "reach", "email", "phone", "call"]):
                contact_info = ("You can reach Aphator Tech through the following channels:\nEmail: info@aphatortech.com\n" +
                              "Support: support@aphatortech.com\nPhone: +1-555-APHATOR\nWebsite: www.aphatortech.com")
                self._learn_from_interaction(user_input, contact_info)
                return contact_info
                       
            # Application development questions
            if any(word in lower_input for word in ["app", "application", "software", "mobile app", "web app"]):
                app_dev_info = ("Aphator Tech provides comprehensive application development services, including mobile applications (iOS/Android), " + 
                              "web applications, enterprise systems, and custom software solutions. Our development process includes " +
                              "requirements analysis, design, development, testing, and deployment. Pricing starts at $10,000 for " +
                              "full applications, with the final cost depending on complexity, features, and timeline. Would you like " +
                              "to discuss your specific software needs?")
                self._learn_from_interaction(user_input, app_dev_info)
                return self._add_engagement_prompt(app_dev_info, "application")
            
            # Pricing questions
            if any(word in lower_input for word in ["price", "pricing", "cost", "how much", "fee", "payment"]):
                pricing_info = ("Aphator Tech offers various services and products with different pricing structures. " +
                              "Our software development services start at $10,000 for full applications. " +
                              "Blockchain development starts at $5,000. Our products include CryptoTracker Pro ($29.99/month), " +
                              "BlockSecure ($49.99/month), SmartContract Builder ($39.99/month), and TradeBotX ($59.99/month). " +
                              "We'd be happy to provide a detailed quote based on your specific requirements.")
                self._learn_from_interaction(user_input, pricing_info)
                return pricing_info
        
        # If we have vectorized data, find the best match
        if self.tfidf_matrix is not None and self.corpus:
            # Preprocess and vectorize the user input
            processed_input = self._preprocess_text(user_input)
            input_vector = self.vectorizer.transform([processed_input])
            
            # Calculate similarity with all corpus items
            similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
            
            # Find the best match
            best_match_idx = similarity_scores.argmax()
            max_similarity = similarity_scores[best_match_idx]
            
            logger.debug(f"Best match index: {best_match_idx}, similarity: {max_similarity}")
            
            # If we have a reasonable match, return the corresponding response
            if max_similarity > 0.2:  # Threshold for confidence
                response = self.responses[best_match_idx]
                
                # Determine topic to use with the engagement prompt
                topic = None
                for topic_name, keywords in self.topic_keywords.items():
                    if any(keyword in response.lower() for keyword in keywords):
                        topic = topic_name
                        break
                
                # Add engagement and learn from this interaction
                final_response = self._add_engagement_prompt(response, topic)
                self._learn_from_interaction(user_input, final_response)
                return final_response
        
        # Check for conversation continuity based on context
        if self.context.history:
            # Look for potential follow-up to previous conversation
            last_exchange = self.context.history[-1]
            last_response = last_exchange['bot_response'].lower()
            
            # If the previous response mentioned a service/product, try to continue that thread
            if any(term in last_response for term in ['would you like', 'more information', 'tell me', 'interested in']):
                # Detect positive response
                if any(word in lower_input for word in ['yes', 'sure', 'okay', 'please', 'interested', 'tell me']):
                    # Find what we were talking about
                    topics_mentioned = []
                    for topic, keywords in self.topic_keywords.items():
                        if any(keyword in last_response for keyword in keywords):
                            topics_mentioned.append(topic)
                    
                    if topics_mentioned:
                        primary_topic = topics_mentioned[0]
                        if primary_topic == 'blockchain':
                            follow_up = ("Our blockchain development team specializes in EVM-compatible chains like Ethereum, " +
                                       "Binance Smart Chain, and Polygon, as well as alternate protocols like Solana and Cosmos. " +
                                       "We can develop custom smart contracts, create DApps, handle token issuance, build NFT platforms, " +
                                       "and integrate existing applications with blockchain technology. Would you like to schedule " +
                                       "a consultation with one of our blockchain specialists?")
                            self._learn_from_interaction(user_input, follow_up)
                            return follow_up
                        
                        elif primary_topic == 'crypto_trading':
                            follow_up = ("Our trading solutions can be customized to your specific needs and trading style. " +
                                       "We can develop algorithmic bots that trade based on technical indicators, implement " +
                                       "specific strategies (trend-following, mean reversion, arbitrage, etc.), create real-time " +
                                       "portfolio trackers and analytics systems, and integrate with major exchanges. Would you like " +
                                       "to discuss what features would be most important for your trading needs?")
                            self._learn_from_interaction(user_input, follow_up)
                            return follow_up
                        
                        elif primary_topic == 'application':
                            follow_up = ("Our application development process begins with thorough requirements gathering " +
                                       "to ensure we build precisely what you need. We develop mobile apps (native or cross-platform), " +
                                       "web applications, enterprise systems, and custom software solutions. Our developers follow " +
                                       "industry best practices for secure, scalable, and maintainable code. Would you like to " +
                                       "tell me more about the specific application you're looking to build?")
                            self._learn_from_interaction(user_input, follow_up)
                            return follow_up
                        
                        elif primary_topic == 'nft':
                            follow_up = ("Our NFT development services cover the complete lifecycle from concept to marketplace. " +
                                       "This includes creating smart contracts for your collection, implementing minting functionality, " +
                                       "managing metadata and assets, building marketplace functionality, and ensuring proper royalty " +
                                       "distribution. We've helped launch several successful NFT projects in art, gaming, and utility tokens. " +
                                       "Would you like to discuss your specific NFT project ideas?")
                            self._learn_from_interaction(user_input, follow_up)
                            return follow_up
                        
                        elif primary_topic == 'security':
                            follow_up = ("Our security services include comprehensive audits of smart contracts and blockchain applications, " +
                                       "implementation of multi-signature solutions, secure key management systems, vulnerability assessment, " +
                                       "penetration testing, and ongoing security monitoring. We help protect your digital assets with " +
                                       "industry-leading security practices. Would you like more information about specific security " +
                                       "concerns or protocols?")
                            self._learn_from_interaction(user_input, follow_up)
                            return follow_up
        
        # Attempt to handle general questions based on intent
        if intent != "general_query":
            if intent == "opinion":
                opinion_response = ("Based on our extensive experience in the crypto and tech space, Aphator Tech " +
                                  "recommends a careful, strategic approach to implementing blockchain and crypto solutions. " +
                                  "Security should always be the priority, followed by scalability and user experience. " +
                                  "Our experts can provide more specific recommendations based on your unique requirements.")
                self._learn_from_interaction(user_input, opinion_response)
                return opinion_response
            
            elif intent == "comparison":
                comparison_response = ("When comparing solutions, Aphator Tech focuses on security, performance, cost-effectiveness, " +
                                     "and long-term maintainability. Each technology has its strengths - for example, Ethereum offers " +
                                     "robust security and widespread adoption but with higher gas fees, while alternatives like Solana " +
                                     "offer higher throughput at potentially lower costs. We can help you evaluate the best fit for your specific needs.")
                self._learn_from_interaction(user_input, comparison_response)
                return comparison_response
            
            elif intent == "problem":
                problem_response = ("Aphator Tech specializes in solving complex technical challenges in the crypto and blockchain space. " +
                                  "Common issues we address include smart contract vulnerabilities, blockchain integration difficulties, " +
                                  "scalability bottlenecks, and security concerns. Our team can analyze your specific problem and develop " +
                                  "a tailored solution. Could you tell me more about the specific challenge you're facing?")
                self._learn_from_interaction(user_input, problem_response)
                return problem_response
            
            elif intent == "clarification":
                clarification_response = ("I'd be happy to clarify any information about Aphator Tech's services or crypto technology in general. " +
                                       "We aim to make complex technical concepts accessible and understandable. Could you specify which " +
                                       "aspect you'd like me to explain in more detail?")
                self._learn_from_interaction(user_input, clarification_response)
                return clarification_response
        
        # General learning response for unhandled queries about learning capabilities
        if any(word in lower_input for word in ["learn", "train", "teaching", "model", "improve"]):
            learning_response = ("I'm designed to learn and improve as I interact with more questions. While I don't have training capabilities " +
                              "in the traditional sense, the Aphator Tech team regularly updates my knowledge base to better assist with questions " +
                              "about our crypto and tech services. Is there something specific about Aphator Tech you'd like to know?")
            self._learn_from_interaction(user_input, learning_response)
            return learning_response
        
        # Custom fallback with context awareness
        if sentiment == "negative":
            # If user seems frustrated, be more helpful
            apologetic_fallback = ("I apologize for not understanding your question correctly. " +
                                 "Aphator Tech specializes in blockchain development, crypto trading solutions, application development, " +
                                 "NFT platforms, and security services. Could you please rephrase your question or specify which " +
                                 "of our services you're interested in learning more about?")
            self._learn_from_interaction(user_input, apologetic_fallback)
            return apologetic_fallback
        else:
            # Standard fallback
            fallback = random.choice(self.fallbacks)
            self._learn_from_interaction(user_input, fallback)
            return fallback
    
    def _is_greeting(self, text):
        """Check if text contains a greeting."""
        greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        return any(pattern in text.lower() for pattern in greeting_patterns)
    
    def _is_farewell(self, text):
        """Check if text contains a farewell."""
        farewell_patterns = ['bye', 'goodbye', 'see you', 'talk to you later', 'thanks', 'thank you']
        return any(pattern in text.lower() for pattern in farewell_patterns)
        
    def _detect_intent(self, text):
        """Determine the intent of the user's message."""
        text_lower = text.lower()
        
        # Check against our intent categories
        for intent, patterns in INTENT_CATEGORIES.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
                
        # Check for known topics in our topic keywords
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return f"topic_{topic}"
                
        return "general_query"
        
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis to detect user mood."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like', 'helpful', 'useful']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'worst', 'hate', 'dislike', 'useless', 'disappointing', 'expensive']
        
        text_tokens = simple_tokenize(text)
        positive_count = sum(1 for word in text_tokens if word in positive_words)
        negative_count = sum(1 for word in text_tokens if word in negative_words)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"
        
    def _add_engagement_prompt(self, response, topic=None):
        """Add an engagement prompt to encourage further conversation."""
        # Don't add prompts to every response
        if random.random() > 0.7:  # 30% chance to add a prompt
            return response
            
        if not topic:
            # Try to determine topic from the response
            for topic_name, keywords in self.topic_keywords.items():
                if any(keyword in response.lower() for keyword in keywords):
                    topic = topic_name
                    break
                    
        if topic:
            # Format the prompt with the detected topic
            prompt = random.choice(self.engagement_prompts).format(topic=topic.replace('_', ' '))
            return f"{response}\n\n{prompt}"
        
        return response
        
    def _learn_from_interaction(self, user_input, response):
        """Store user input patterns to improve future responses."""
        # Extract key terms from user input
        tokens = simple_tokenize(user_input)
        key_terms = [token for token in tokens if len(token) > 3 and token not in self.stop_words]
        
        if key_terms and len(key_terms) >= 2:
            # Create a simple pattern from the key terms
            pattern = ' '.join(sorted(key_terms[:3]))  # Use up to 3 key terms
            
            # Store the response for this pattern if we don't already have one
            if pattern not in self.learned_responses:
                self.learned_responses[pattern] = response
                logger.debug(f"Learned new pattern: {pattern}")
                
        # Update conversation context
        self.context.add_exchange(user_input, response)
