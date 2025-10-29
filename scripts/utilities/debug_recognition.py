"""
Debug script to test face recognition with detailed output.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.recognizer import FaceRecognizer
from app.core.camera import CameraHandler
from app.core.database import SessionLocal
from app.models.database import Person, FaceEmbedding
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_recognition():
    print("\n" + "=" * 70)
    print("FACE RECOGNITION DEBUG")
    print("=" * 70 + "\n")

    # Initialize
    recognizer = FaceRecognizer()
    db = SessionLocal()

    # Get enrolled persons
    persons = db.query(Person).all()
    print(f"📋 Enrolled Persons: {len(persons)}")
    for p in persons:
        print(f"   - {p.name} (CNIC: {p.cnic}, ID: {p.id})")
    print()

    if len(persons) == 0:
        print("❌ No enrolled persons. Enroll someone first!")
        return

    # Get embeddings
    all_embeddings = db.query(FaceEmbedding).all()
    print(f"💾 Stored Embeddings: {len(all_embeddings)}")
    for emb in all_embeddings:
        person = db.query(Person).filter(Person.id == emb.person_id).first()
        print(f"   - Person: {person.name}, Source: {emb.source}, Confidence: {emb.confidence:.3f}")
    print()

    # Capture from camera
    print("📹 Capturing from camera...")
    camera = CameraHandler(use_main_stream=False)

    if not camera.connect():
        print("❌ Failed to connect to camera")
        return

    ret, frame = camera.read_frame()
    camera.disconnect()

    if not ret or frame is None:
        print("❌ Failed to capture frame")
        return

    print("✅ Frame captured\n")

    # Extract embedding from camera
    print("🔍 Extracting embedding from camera image...")
    result = recognizer.extract_embedding(frame)

    if result is None:
        print("❌ No face detected in camera frame")
        print("   Try:")
        print("   - Moving closer to camera")
        print("   - Better lighting")
        print("   - Looking directly at camera")
        return

    print(f"✅ Face detected!")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Embedding shape: {result.embedding.shape}")
    print()

    # Load all database embeddings
    print("🔄 Comparing with enrolled embeddings...")
    db_embeddings = []
    person_ids = []

    for emb in all_embeddings:
        embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)
        db_embeddings.append(embedding_vec)
        person_ids.append(emb.person_id)

    # Compute similarities with each
    print(f"\n📊 Similarity Scores:")
    print("-" * 70)
    for i, (db_emb, person_id) in enumerate(zip(db_embeddings, person_ids)):
        person = db.query(Person).filter(Person.id == person_id).first()
        similarity = recognizer.compare_embeddings(result.embedding, db_emb)

        status = "✅ MATCH!" if similarity >= 0.6 else "❌ No match"
        print(f"   {person.name:20s} | Similarity: {similarity:.4f} | {status}")

    print("-" * 70)

    # Find best match
    best_idx, best_similarity = recognizer.match_face(
        result.embedding,
        db_embeddings,
        threshold=0.6
    )

    print(f"\n🎯 Best Match:")
    if best_idx >= 0:
        best_person = db.query(Person).filter(Person.id == person_ids[best_idx]).first()
        print(f"   ✅ RECOGNIZED: {best_person.name}")
        print(f"   Similarity: {best_similarity:.4f}")
        print(f"   CNIC: {best_person.cnic}")
    else:
        print(f"   ❌ NOT RECOGNIZED")
        print(f"   Best similarity: {best_similarity:.4f} (threshold: 0.6)")
        print(f"   Difference: {0.6 - best_similarity:.4f} below threshold")

    print("\n" + "=" * 70)

    db.close()

if __name__ == "__main__":
    debug_recognition()
