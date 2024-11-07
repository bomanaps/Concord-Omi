from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DB_NAME')]
channels_collection = db['channels']

def extract_transcript_segments(memory_data: dict) -> List[str]:
    """Extract all transcript segments from the memory data."""
    segments = []
    for segment in memory_data.get('transcript_segments', []):
        if segment.get('text'):
            segments.append(segment['text'])
    return segments

def get_embedding(texts: List[str]) -> np.ndarray:
    """Generate BERT embedding for the combined text."""
    combined_text = ' '.join(texts)
    return model.encode([combined_text])[0]

def find_similar_channels(embedding: np.ndarray, threshold: float = 0.7) -> List[dict]:
    """Find channels with similar topics based on BERT embedding similarity."""
    similar_channels = []
    
    # Get all channels from MongoDB
    channels = channels_collection.find({}, {'_id': 1, 'name': 1, 'description': 1, 'topic_embedding': 1})
    
    for channel in channels:
        if 'topic_embedding' in channel:
            # Calculate cosine similarity
            similarity = np.dot(embedding, channel['topic_embedding']) / (
                np.linalg.norm(embedding) * np.linalg.norm(channel['topic_embedding'])
            )
            
            if similarity >= threshold:
                similar_channels.append({
                    'channel_id': str(channel['_id']),
                    'name': channel['name'],
                    'description': channel['description'],
                    'similarity_score': float(similarity)
                })
    
    # Sort by similarity score in descending order
    similar_channels.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar_channels

@app.route('/process-memory', methods=['POST'])
def process_memory():
    """
    Process incoming memory data from OMI and return similar channels.
    Expected query parameter: uid
    """
    try:
        # Get user ID from query parameters
        uid = request.args.get('uid')
        if not uid:
            return jsonify({'error': 'Missing uid parameter'}), 400

        # Get memory data from request body
        memory_data = request.json
        if not memory_data:
            return jsonify({'error': 'No memory data provided'}), 400

        # Extract transcript segments
        segments = extract_transcript_segments(memory_data)
        if not segments:
            return jsonify({'error': 'No transcript segments found'}), 400

        # Generate embedding for the conversation
        conversation_embedding = get_embedding(segments)

        # Find similar channels
        similar_channels = find_similar_channels(conversation_embedding)

        return jsonify({
            'uid': uid,
            'similar_channels': similar_channels,
            'segments_processed': len(segments)
        })

    except Exception as e:
        app.logger.error(f"Error processing memory: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/setup-complete', methods=['GET'])
def setup_complete():
    """
    Endpoint to check if the app is properly set up for a user.
    """
    return jsonify({'is_setup_completed': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))