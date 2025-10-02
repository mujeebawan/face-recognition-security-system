"""
Face recognition and enrollment API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional
import time

from app.core.recognizer import FaceRecognizer
from app.core.detector import FaceDetector
from app.core.camera import CameraHandler
from app.core.augmentation import FaceAugmentation
from app.core.database import get_db
from app.models.database import Person, FaceEmbedding, RecognitionLog
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recognition"])

# Initialize face recognizer (singleton - will be initialized on first use)
face_recognizer: Optional[FaceRecognizer] = None


def get_recognizer() -> FaceRecognizer:
    """Get or initialize face recognizer"""
    global face_recognizer
    if face_recognizer is None:
        logger.info("Initializing face recognizer...")
        face_recognizer = FaceRecognizer()
    return face_recognizer


@router.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    cnic: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Enroll a new person with their face image.

    Args:
        name: Person's name
        cnic: National ID number (unique)
        file: Face image file
        db: Database session

    Returns:
        Enrollment result with person ID
    """
    try:
        # Check if CNIC already exists
        existing = db.query(Person).filter(Person.cnic == cnic).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Person with CNIC {cnic} already enrolled")

        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Extract face embedding
        recognizer = get_recognizer()
        result = recognizer.extract_embedding(image)

        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Create person record
        person = Person(
            name=name,
            cnic=cnic,
            reference_image_path=f"data/images/{cnic}_{file.filename}"
        )
        db.add(person)
        db.flush()  # Get the person.id

        # Store face embedding
        embedding_data = FaceRecognizer.serialize_embedding(result.embedding)
        face_embedding = FaceEmbedding(
            person_id=person.id,
            embedding=embedding_data,
            source='original',
            confidence=result.confidence
        )
        db.add(face_embedding)
        db.commit()

        # Save reference image
        import os
        os.makedirs("data/images", exist_ok=True)
        cv2.imwrite(person.reference_image_path, image)

        logger.info(f"Enrolled person: {name} (CNIC: {cnic}, ID: {person.id})")

        return {
            "success": True,
            "message": f"Person {name} enrolled successfully",
            "person_id": person.id,
            "cnic": cnic,
            "confidence": result.confidence,
            "embedding_dimension": len(result.embedding)
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error enrolling person: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize")
async def recognize_face(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Recognize a face from uploaded image.

    Args:
        file: Face image file
        db: Database session

    Returns:
        Recognition result with person details
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Extract face embedding
        recognizer = get_recognizer()
        result = recognizer.extract_embedding(image)

        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Get all enrolled embeddings
        all_embeddings = db.query(FaceEmbedding).all()

        if len(all_embeddings) == 0:
            return {
                "success": False,
                "message": "No enrolled persons in database",
                "matched": False
            }

        # Prepare database embeddings
        db_embeddings = []
        person_ids = []
        for emb in all_embeddings:
            embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)
            db_embeddings.append(embedding_vec)
            person_ids.append(emb.person_id)

        # Match face
        best_idx, similarity = recognizer.match_face(
            result.embedding,
            db_embeddings,
            threshold=settings.face_recognition_threshold
        )

        # Log recognition attempt
        log = RecognitionLog(
            person_id=person_ids[best_idx] if best_idx >= 0 else None,
            confidence=similarity,
            matched=1 if best_idx >= 0 else 0,
            camera_source="upload"
        )
        db.add(log)
        db.commit()

        if best_idx >= 0:
            # Found a match
            person = db.query(Person).filter(Person.id == person_ids[best_idx]).first()

            logger.info(f"Face recognized: {person.name} (similarity: {similarity:.3f})")

            return {
                "success": True,
                "matched": True,
                "person": {
                    "id": person.id,
                    "name": person.name,
                    "cnic": person.cnic
                },
                "similarity": round(similarity, 3),
                "confidence": round(result.confidence, 3)
            }
        else:
            logger.info(f"Face not recognized (best similarity: {similarity:.3f})")

            return {
                "success": True,
                "matched": False,
                "message": "Face not recognized",
                "best_similarity": round(similarity, 3)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recognize/camera")
async def recognize_from_camera(db: Session = Depends(get_db)):
    """
    Recognize face from live camera feed.

    Args:
        db: Database session

    Returns:
        Recognition result
    """
    try:
        # Capture from camera
        camera = CameraHandler(use_main_stream=False)

        if not camera.connect():
            raise HTTPException(status_code=503, detail="Failed to connect to camera")

        ret, frame = camera.read_frame()
        camera.disconnect()

        if not ret or frame is None:
            raise HTTPException(status_code=503, detail="Failed to capture frame")

        # Extract embedding
        recognizer = get_recognizer()
        result = recognizer.extract_embedding(frame)

        if result is None:
            return {
                "success": False,
                "message": "No face detected in camera",
                "matched": False
            }

        # Get all enrolled embeddings
        all_embeddings = db.query(FaceEmbedding).all()

        if len(all_embeddings) == 0:
            return {
                "success": False,
                "message": "No enrolled persons in database",
                "matched": False
            }

        # Match face
        db_embeddings = []
        person_ids = []
        for emb in all_embeddings:
            embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)
            db_embeddings.append(embedding_vec)
            person_ids.append(emb.person_id)

        best_idx, similarity = recognizer.match_face(
            result.embedding,
            db_embeddings,
            threshold=settings.face_recognition_threshold
        )

        # Log recognition
        log = RecognitionLog(
            person_id=person_ids[best_idx] if best_idx >= 0 else None,
            confidence=similarity,
            matched=1 if best_idx >= 0 else 0,
            camera_source=f"hikvision_{settings.camera_ip}"
        )
        db.add(log)
        db.commit()

        if best_idx >= 0:
            person = db.query(Person).filter(Person.id == person_ids[best_idx]).first()

            return {
                "success": True,
                "matched": True,
                "person": {
                    "id": person.id,
                    "name": person.name,
                    "cnic": person.cnic
                },
                "similarity": round(similarity, 3),
                "timestamp": datetime.utcnow()
            }
        else:
            return {
                "success": True,
                "matched": False,
                "message": "Face not recognized",
                "best_similarity": round(similarity, 3),
                "timestamp": datetime.utcnow()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in camera recognition: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persons")
async def list_persons(db: Session = Depends(get_db)):
    """List all enrolled persons"""
    persons = db.query(Person).all()

    return {
        "total": len(persons),
        "persons": [
            {
                "id": p.id,
                "name": p.name,
                "cnic": p.cnic,
                "enrolled_at": p.created_at
            }
            for p in persons
        ]
    }


@router.delete("/persons/{person_id}")
async def delete_person(person_id: int, db: Session = Depends(get_db)):
    """Delete an enrolled person"""
    person = db.query(Person).filter(Person.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    db.delete(person)
    db.commit()

    return {
        "success": True,
        "message": f"Person {person.name} deleted successfully"
    }


@router.post("/enroll/multiple")
async def enroll_person_multiple_images(
    name: str = Form(...),
    cnic: str = Form(...),
    files: List[UploadFile] = File(...),
    use_augmentation: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Enroll a person with multiple face images for better accuracy.

    Args:
        name: Person's name
        cnic: National ID number (unique)
        files: List of face image files (2-10 images recommended)
        use_augmentation: Apply augmentation to each image
        db: Database session

    Returns:
        Enrollment result with total embeddings stored
    """
    try:
        # Check if CNIC already exists
        existing = db.query(Person).filter(Person.cnic == cnic).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Person with CNIC {cnic} already enrolled")

        if len(files) < 1:
            raise HTTPException(status_code=400, detail="At least one image required")

        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

        recognizer = get_recognizer()
        augmentor = FaceAugmentation()

        # Create person record
        person = Person(
            name=name,
            cnic=cnic,
            reference_image_path=f"data/images/{cnic}_multiple"
        )
        db.add(person)
        db.flush()

        all_embeddings = []
        saved_images = []

        # Process each uploaded image
        for idx, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.warning(f"Invalid image file: {file.filename}")
                continue

            # Extract embedding from original
            result = recognizer.extract_embedding(image)

            if result is None:
                logger.warning(f"No face detected in {file.filename}")
                continue

            # Store original embedding
            embedding_data = FaceRecognizer.serialize_embedding(result.embedding)
            face_embedding = FaceEmbedding(
                person_id=person.id,
                embedding=embedding_data,
                source=f'original_{idx+1}',
                confidence=result.confidence
            )
            db.add(face_embedding)
            all_embeddings.append(result.embedding)

            # Save original image
            import os
            os.makedirs("data/images", exist_ok=True)
            img_path = f"data/images/{cnic}_img{idx+1}.jpg"
            cv2.imwrite(img_path, image)
            saved_images.append(img_path)

            # Apply augmentation if requested
            if use_augmentation and len(files) < 5:  # Only augment if few images
                variations = augmentor.generate_variations(image, num_variations=5)

                for var_idx, var_img in enumerate(variations[1:], 1):  # Skip first (original)
                    var_result = recognizer.extract_embedding(var_img)

                    if var_result is not None:
                        var_embedding_data = FaceRecognizer.serialize_embedding(var_result.embedding)
                        var_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=var_embedding_data,
                            source=f'augmented_{idx+1}_{var_idx}',
                            confidence=var_result.confidence
                        )
                        db.add(var_face_embedding)
                        all_embeddings.append(var_result.embedding)

        if len(all_embeddings) == 0:
            db.rollback()
            raise HTTPException(status_code=400, detail="No valid faces detected in any image")

        db.commit()

        logger.info(f"Enrolled {name} with {len(all_embeddings)} embeddings from {len(files)} images")

        return {
            "success": True,
            "message": f"Person {name} enrolled successfully with multiple images",
            "person_id": person.id,
            "cnic": cnic,
            "images_processed": len(files),
            "total_embeddings": len(all_embeddings),
            "augmentation_used": use_augmentation
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error enrolling person with multiple images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/camera")
async def enroll_from_camera(
    name: str = Form(...),
    cnic: str = Form(...),
    num_captures: int = Form(5),
    use_augmentation: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Enroll a person by capturing multiple images from camera.

    Args:
        name: Person's name
        cnic: National ID number (unique)
        num_captures: Number of frames to capture (3-10)
        use_augmentation: Apply augmentation to captured images
        db: Database session

    Returns:
        Enrollment result
    """
    try:
        # Check if CNIC already exists
        existing = db.query(Person).filter(Person.cnic == cnic).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Person with CNIC {cnic} already enrolled")

        if num_captures < 3 or num_captures > 10:
            raise HTTPException(status_code=400, detail="num_captures must be between 3 and 10")

        recognizer = get_recognizer()
        augmentor = FaceAugmentation()
        camera = CameraHandler(use_main_stream=False)

        if not camera.connect():
            raise HTTPException(status_code=503, detail="Failed to connect to camera")

        # Create person record
        person = Person(
            name=name,
            cnic=cnic,
            reference_image_path=f"data/images/{cnic}_camera"
        )
        db.add(person)
        db.flush()

        all_embeddings = []
        captured_frames = []

        # Capture multiple frames
        logger.info(f"Capturing {num_captures} frames for enrollment...")

        for i in range(num_captures * 3):  # Capture more, keep best
            ret, frame = camera.read_frame()

            if ret and frame is not None:
                result = recognizer.extract_embedding(frame)

                if result is not None and result.confidence > 0.7:  # Good quality frame
                    captured_frames.append((frame, result))

                    if len(captured_frames) >= num_captures:
                        break

        camera.disconnect()

        if len(captured_frames) == 0:
            db.rollback()
            raise HTTPException(status_code=400, detail="No faces detected in camera")

        # Process captured frames
        import os
        os.makedirs("data/images", exist_ok=True)

        for idx, (frame, result) in enumerate(captured_frames):
            # Store original embedding
            embedding_data = FaceRecognizer.serialize_embedding(result.embedding)
            face_embedding = FaceEmbedding(
                person_id=person.id,
                embedding=embedding_data,
                source=f'camera_{idx+1}',
                confidence=result.confidence
            )
            db.add(face_embedding)
            all_embeddings.append(result.embedding)

            # Save captured frame
            img_path = f"data/images/{cnic}_camera{idx+1}.jpg"
            cv2.imwrite(img_path, frame)

            # Apply augmentation
            if use_augmentation and len(captured_frames) < 5:
                variations = augmentor.generate_variations(frame, num_variations=3)

                for var_idx, var_img in enumerate(variations[1:], 1):
                    var_result = recognizer.extract_embedding(var_img)

                    if var_result is not None:
                        var_embedding_data = FaceRecognizer.serialize_embedding(var_result.embedding)
                        var_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=var_embedding_data,
                            source=f'camera_aug_{idx+1}_{var_idx}',
                            confidence=var_result.confidence
                        )
                        db.add(var_face_embedding)
                        all_embeddings.append(var_result.embedding)

        db.commit()

        logger.info(f"Enrolled {name} from camera with {len(all_embeddings)} embeddings")

        return {
            "success": True,
            "message": f"Person {name} enrolled from camera successfully",
            "person_id": person.id,
            "cnic": cnic,
            "frames_captured": len(captured_frames),
            "total_embeddings": len(all_embeddings),
            "augmentation_used": use_augmentation
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error enrolling from camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_video_stream(db: Session):
    """
    Generate MJPEG video stream with real-time face recognition.
    Uses MediaPipe for fast detection, InsightFace only for recognition.

    Yields:
        JPEG frames with recognition overlay
    """
    detector = FaceDetector()
    recognizer = get_recognizer()
    camera = CameraHandler(use_main_stream=False)

    if not camera.connect():
        logger.error("Failed to connect to camera for streaming")
        return

    # Load all enrolled persons and embeddings
    all_embeddings = db.query(FaceEmbedding).all()

    if len(all_embeddings) == 0:
        logger.warning("No enrolled persons in database")

    # Build embedding database
    db_embeddings = []
    person_ids = []

    for emb in all_embeddings:
        embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)
        db_embeddings.append(embedding_vec)
        person_ids.append(emb.person_id)

    logger.info(f"Streaming started with {len(db_embeddings)} embeddings from {len(set(person_ids))} persons")

    frame_count = 0
    last_recognition = {}  # Cache last recognition per face
    last_detection_bbox = None  # Cache last detection bbox

    try:
        while True:
            ret, frame = camera.read_frame()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Skip frames to reduce processing load (process every 2nd frame)
            if frame_count % 2 != 0:
                # Use cached detection for skipped frames
                if last_detection_bbox is not None and last_recognition and 'best_idx' in last_recognition:
                    x, y, w, h = last_detection_bbox
                    best_idx = last_recognition['best_idx']
                    similarity = last_recognition['similarity']

                    if best_idx >= 0:
                        person_id = last_recognition['person_id']
                        person = db.query(Person).filter(Person.id == person_id).first()

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 255, 0), -1)
                        cv2.putText(frame, f"KNOWN: {person.name}", (x + 5, y - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(frame, f"Match: {similarity:.2f}", (x + 5, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 0, 255), -1)
                        cv2.putText(frame, "UNKNOWN PERSON", (x + 5, y - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Sim: {similarity:.2f}", (x + 5, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Encode and yield quickly
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            # Use MediaPipe for fast face detection on processed frames
            detections = detector.detect_faces(frame)

            if detections and len(detections) > 0:
                # Face detected with MediaPipe
                detection = detections[0]  # First face
                bbox = detection.bbox
                x, y, w, h = bbox

                # Cache bbox for skipped frames
                last_detection_bbox = (x, y, w, h)

                # Only run recognition every 10th frame
                if frame_count % 20 == 0 and len(db_embeddings) > 0:
                    # Use InsightFace for recognition
                    result = recognizer.extract_embedding(frame)

                    if result is not None:
                        best_idx, similarity = recognizer.match_face(
                            result.embedding,
                            db_embeddings,
                            threshold=settings.face_recognition_threshold
                        )

                        # Cache result
                        last_recognition = {
                            'best_idx': best_idx,
                            'similarity': similarity,
                            'person_id': person_ids[best_idx] if best_idx >= 0 else None
                        }

                # Draw box with cached or current recognition
                if last_recognition and 'best_idx' in last_recognition:
                    best_idx = last_recognition['best_idx']
                    similarity = last_recognition['similarity']

                    if best_idx >= 0:
                        # Known person
                        person_id = last_recognition['person_id']
                        person = db.query(Person).filter(Person.id == person_id).first()

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 255, 0), -1)
                        cv2.putText(frame, f"KNOWN: {person.name}", (x + 5, y - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(frame, f"Match: {similarity:.2f}", (x + 5, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    else:
                        # Unknown person
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 0, 255), -1)
                        cv2.putText(frame, "UNKNOWN PERSON", (x + 5, y - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Sim: {similarity:.2f}", (x + 5, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    # Just detected, no recognition yet
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, "Detecting...", (x + 5, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # No face detected - clear cache
                last_recognition = {}
                last_detection_bbox = None

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except GeneratorExit:
        logger.info("Streaming stopped by client")
    finally:
        camera.disconnect()
        logger.info("Camera disconnected")


@router.get("/stream/live")
async def live_stream(db: Session = Depends(get_db)):
    """
    Live video stream with real-time face recognition overlay.

    Returns:
        MJPEG video stream showing Known/Unknown labels
    """
    return StreamingResponse(
        generate_video_stream(db),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
