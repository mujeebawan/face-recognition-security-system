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
import threading
from queue import Queue, Empty

from app.core.recognizer import FaceRecognizer
from app.core.detector import FaceDetector
from app.core.camera import CameraHandler
from app.core.augmentation import FaceAugmentation
from app.core.alerts import AlertManager
from app.core.database import get_db
from app.models.database import Person, FaceEmbedding, RecognitionLog, User
from app.core.auth import get_current_user
from app.config import settings
import torch


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recognition"])

# Initialize face recognizer, detector, alert manager, and camera (singletons)
face_recognizer: Optional[FaceRecognizer] = None
face_detector: Optional[FaceDetector] = None
alert_manager: Optional[AlertManager] = None
camera_handler: Optional[CameraHandler] = None


def get_recognizer() -> FaceRecognizer:
    """Get or initialize face recognizer"""
    global face_recognizer
    if face_recognizer is None:
        logger.info("Initializing face recognizer...")
        face_recognizer = FaceRecognizer()
    return face_recognizer


def get_detector() -> FaceDetector:
    """Get or initialize face detector (singleton to avoid slow SCRFD reload)"""
    global face_detector
    if face_detector is None:
        logger.info("Initializing SCRFD detector (one-time GPU initialization)...")
        face_detector = FaceDetector()
    return face_detector


def get_alert_manager() -> AlertManager:
    """Get or initialize alert manager"""
    global alert_manager
    if alert_manager is None:
        logger.info("Initializing alert manager...")
        alert_manager = AlertManager()
    return alert_manager


def get_camera() -> CameraHandler:
    """Get or initialize camera handler (singleton to avoid multiple RTSP connections)"""
    global camera_handler
    if camera_handler is None:
        logger.info("Initializing camera handler (singleton)...")
        camera_handler = CameraHandler(use_main_stream=False)  # Use sub-stream as requested
        if not camera_handler.connect():
            logger.error("Failed to connect to camera on startup")
            camera_handler = None
            raise HTTPException(status_code=503, detail="Camera connection failed")
    elif not camera_handler.is_connected:
        # Reconnect if disconnected
        logger.info("Reconnecting to camera...")
        if not camera_handler.connect():
            logger.error("Failed to reconnect to camera")
            raise HTTPException(status_code=503, detail="Camera reconnection failed")
    return camera_handler


@router.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    cnic: str = Form(...),
    file: UploadFile = File(...),
    use_sd_augmentation: bool = Form(False),
    use_controlnet: bool = Form(False),
    use_liveportrait: bool = Form(False),
    num_sd_variations: int = Form(5),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enroll a new person with their face image.

    Args:
        name: Person's name
        cnic: National ID number (unique)
        file: Face image file
        use_sd_augmentation: Use Stable Diffusion to generate additional angles (default: False)
        use_controlnet: Use ControlNet for better pose control (default: False, requires use_sd_augmentation=True)
        use_liveportrait: Use LivePortrait for 3D-aware pose generation (default: False)
        num_sd_variations: Number of variations to generate if augmentation enabled (default: 5, max: 10)
        db: Database session

    Returns:
        Enrollment result with person ID and total embeddings
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

        # Extract face embedding from original
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

        # Store original face embedding
        embedding_data = FaceRecognizer.serialize_embedding(result.embedding)
        face_embedding = FaceEmbedding(
            person_id=person.id,
            embedding=embedding_data,
            source='original',
            confidence=result.confidence
        )
        db.add(face_embedding)

        # Save reference image
        import os
        os.makedirs("data/images", exist_ok=True)
        cv2.imwrite(person.reference_image_path, image)

        total_embeddings = 1
        augmentation_time = 0

        # Generate LivePortrait augmented faces if requested (takes priority over SD)
        if use_liveportrait:
            try:
                import time
                from app.core.liveportrait_augmentation import LivePortraitAugmentor

                # Limit variations
                num_variations = min(max(1, num_sd_variations), 10)

                logger.info(f"Generating {num_variations} LivePortrait pose variations for {name}...")

                # Initialize LivePortrait augmentor
                augmentor = LivePortraitAugmentor(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_fp16=True
                )
                augmentation_type = "liveportrait"

                # Generate variations
                start_time = time.time()
                generated_images = augmentor.generate_face_angles(
                    reference_image=image,
                    num_variations=num_variations
                )
                augmentation_time = time.time() - start_time

                logger.info(f"LivePortrait generation completed in {augmentation_time:.2f}s ({len(generated_images)} images)")

                # Process each generated image
                for idx, gen_img in enumerate(generated_images):
                    # Extract embedding from generated image
                    gen_result = recognizer.extract_embedding(gen_img)

                    if gen_result is not None:
                        # Store embedding
                        gen_embedding_data = FaceRecognizer.serialize_embedding(gen_result.embedding)
                        gen_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=gen_embedding_data,
                            source=f'liveportrait_augmented_{idx+1}',
                            confidence=gen_result.confidence
                        )
                        db.add(gen_face_embedding)
                        total_embeddings += 1

                        # Save generated image
                        gen_img_path = f"data/images/{cnic}_liveportrait_gen_{idx+1}.jpg"
                        cv2.imwrite(gen_img_path, gen_img)

                # Cleanup
                del augmentor

                logger.info(f"✅ LivePortrait augmentation complete: {total_embeddings} total embeddings")

            except Exception as aug_e:
                logger.error(f"LivePortrait augmentation failed: {aug_e}, continuing with original only")
                import traceback
                traceback.print_exc()
                # Continue with enrollment even if augmentation fails

        # Generate SD augmented faces if requested (only if LivePortrait not used)
        elif use_sd_augmentation:
            try:
                import time

                # Limit variations
                num_variations = min(max(1, num_sd_variations), 10)

                # Check GPU memory before loading SD models
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # Free memory in GB
                    logger.info(f"Available GPU memory: {free_mem:.2f} GB")

                    # SD models need ~6-8GB minimum
                    if free_mem < 6.0:
                        raise HTTPException(
                            status_code=507,
                            detail=f"Insufficient GPU memory for Stable Diffusion. Available: {free_mem:.1f}GB, Required: ~6GB. Please use LivePortrait instead (requires only ~2GB) or free up GPU memory."
                        )

                # Choose augmentation method: ControlNet or img2img
                if use_controlnet:
                    from app.core.controlnet_augmentation import ControlNetFaceAugmentor
                    logger.info(f"Generating {num_variations} ControlNet augmented angles for {name}...")

                    # Initialize ControlNet augmentor
                    augmentor = ControlNetFaceAugmentor(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=True
                    )
                    augmentation_type = "controlnet"
                else:
                    from app.core.generative_augmentation import StableDiffusionAugmentor
                    logger.info(f"Generating {num_variations} SD img2img augmented angles for {name}...")

                    # Initialize SD img2img augmentor
                    augmentor = StableDiffusionAugmentor(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=True
                    )
                    augmentation_type = "img2img"

                # Load model
                if not augmentor.load_model():
                    logger.error(f"Failed to load {augmentation_type} model, skipping augmentation")
                else:
                    # Generate variations
                    start_time = time.time()

                    # Prepare parameters based on augmentation type
                    gen_params = {
                        'reference_image': image,
                        'num_variations': num_variations,
                        'num_inference_steps': 30 if use_controlnet else 20,  # ControlNet needs more steps
                        'guidance_scale': 7.5
                    }

                    # Add ControlNet-specific parameter
                    if use_controlnet:
                        gen_params['controlnet_scale'] = 0.9

                    generated_images = augmentor.generate_face_angles(**gen_params)
                    sd_generation_time = time.time() - start_time

                    logger.info(f"{augmentation_type.upper()} generation completed in {sd_generation_time:.2f}s ({len(generated_images)} images)")

                    # Process each generated image
                    for idx, gen_img in enumerate(generated_images):
                        # Extract embedding from generated image
                        gen_result = recognizer.extract_embedding(gen_img)

                        if gen_result is not None:
                            # Store embedding
                            gen_embedding_data = FaceRecognizer.serialize_embedding(gen_result.embedding)
                            gen_face_embedding = FaceEmbedding(
                                person_id=person.id,
                                embedding=gen_embedding_data,
                                source=f'{augmentation_type}_augmented_{idx+1}',
                                confidence=gen_result.confidence
                            )
                            db.add(gen_face_embedding)
                            total_embeddings += 1

                            # Save generated image
                            gen_img_path = f"data/images/{cnic}_{augmentation_type}_gen_{idx+1}.jpg"
                            cv2.imwrite(gen_img_path, gen_img)

                    # Unload model to free GPU memory
                    augmentor.unload_model()

                    logger.info(f"✅ {augmentation_type.upper()} augmentation complete: {total_embeddings} total embeddings")

            except Exception as aug_e:
                logger.error(f"Augmentation failed: {aug_e}, continuing with original only")
                import traceback
                traceback.print_exc()
                # Continue with enrollment even if augmentation fails

        db.commit()

        logger.info(f"Enrolled person: {name} (CNIC: {cnic}, ID: {person.id}, Embeddings: {total_embeddings})")

        response = {
            "success": True,
            "message": f"Person {name} enrolled successfully",
            "person_id": person.id,
            "cnic": cnic,
            "confidence": result.confidence,
            "total_embeddings": total_embeddings,
            "liveportrait_used": use_liveportrait,
            "sd_augmentation_used": use_sd_augmentation,
            "controlnet_used": use_controlnet if use_sd_augmentation else False
        }

        if augmentation_time > 0:
            response["generation_time"] = round(augmentation_time, 2)
            if use_liveportrait:
                response["avg_time_per_image"] = round(augmentation_time / num_sd_variations, 2)
                response["augmentation_method"] = "LivePortrait"
            elif use_sd_augmentation:
                response["avg_time_per_image"] = round(augmentation_time / num_sd_variations, 2)
                response["augmentation_method"] = "ControlNet" if use_controlnet else "img2img"

        return response

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
        camera = CameraHandler(use_main_stream=False)  # Use sub-stream as requested

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
async def list_persons(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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


@router.get("/persons/{person_id}")
async def get_person_details(
    person_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a person including all their images and case history"""
    from app.models.database import Alert
    person = db.query(Person).filter(Person.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Get all embeddings with their sources
    embeddings = db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person_id).all()

    # Get all alerts/cases for this person
    alerts = db.query(Alert).filter(Alert.person_id == person_id).order_by(Alert.timestamp.desc()).all()

    # Get recognition log count
    recognition_count = db.query(RecognitionLog).filter(
        RecognitionLog.person_id == person_id,
        RecognitionLog.matched == 1
    ).count()

    # Collect all image paths
    import os
    from pathlib import Path
    images = []

    # Add original image
    if person.reference_image_path and os.path.exists(person.reference_image_path):
        images.append({
            "source": "original",
            "path": person.reference_image_path,
            "url": f"/api/image/{person.cnic}/original"
        })

    # Add AI-generated images (ControlNet, img2img, and LivePortrait)
    for emb in embeddings:
        # Handle all augmentation types
        if any(emb.source.startswith(prefix) for prefix in ['sd_augmented_', 'controlnet_augmented_', 'img2img_augmented_', 'liveportrait_augmented_']):
            idx = emb.source.split('_')[-1]

            # Determine augmentation type
            if emb.source.startswith('controlnet_augmented_'):
                aug_type = 'controlnet'
            elif emb.source.startswith('img2img_augmented_'):
                aug_type = 'img2img'
            elif emb.source.startswith('liveportrait_augmented_'):
                aug_type = 'liveportrait'
            else:  # sd_augmented (legacy)
                aug_type = 'sd'

            img_path = f"data/images/{person.cnic}_{aug_type}_gen_{idx}.jpg"
            if os.path.exists(img_path):
                images.append({
                    "source": emb.source,
                    "path": img_path,
                    "url": f"/api/image/{person.cnic}/{aug_type}_gen_{idx}",
                    "confidence": emb.confidence
                })

    return {
        "success": True,
        "person": {
            "id": person.id,
            "name": person.name,
            "cnic": person.cnic,
            "enrolled_at": person.created_at,
            "updated_at": person.updated_at
        },
        "embeddings": [
            {
                "id": emb.id,
                "source": emb.source,
                "confidence": emb.confidence,
                "created_at": emb.created_at
            }
            for emb in embeddings
        ],
        "images": images,
        "alerts": [
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "event_type": alert.event_type,
                "confidence": alert.confidence,
                "acknowledged": alert.acknowledged,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at,
                "notes": alert.notes,
                "snapshot_path": alert.snapshot_path
            }
            for alert in alerts
        ],
        "total_embeddings": len(embeddings),
        "total_images": len(images),
        "total_alerts": len(alerts),
        "recognition_count": recognition_count
    }


@router.delete("/persons/{person_id}")
async def delete_person(
    person_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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


@router.get("/image/{cnic}/{image_type}")
async def get_person_image(
    cnic: str,
    image_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Serve person images (original, SD, ControlNet, img2img, or LivePortrait generated)"""
    import os
    from fastapi.responses import FileResponse

    # Determine image path based on type
    if image_type == "original":
        # Get actual path from database instead of assuming filename
        person = db.query(Person).filter(Person.cnic == cnic).first()
        if not person or not person.reference_image_path:
            raise HTTPException(status_code=404, detail="Person or original image not found")
        image_path = person.reference_image_path
    elif any(image_type.startswith(prefix) for prefix in ["sd_gen_", "controlnet_gen_", "img2img_gen_", "liveportrait_gen_"]):
        # AI-generated images (all types)
        image_path = f"data/images/{cnic}_{image_type}.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Check if file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpeg")


@router.post("/enroll/multiple")
async def enroll_person_multiple_images(
    name: str = Form(...),
    cnic: str = Form(...),
    files: List[UploadFile] = File(...),
    use_augmentation: bool = Form(True),
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
        camera = CameraHandler(use_main_stream=False)  # Use sub-stream as requested

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


def create_error_frame(message: str, width: int = 640, height: int = 480):
    """Create an error frame with a message"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background

    # Add error message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 0, 255)  # Red text

    # Split message into lines
    lines = message.split('\n')
    y_start = (height - len(lines) * 40) // 2

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = y_start + i * 40
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)

    return frame


def generate_video_stream(db: Session):
    """
    Generate MJPEG video stream with real-time face recognition.
    Uses SCRFD (GPU) for fast detection, InsightFace for recognition.
    Triggers alerts for unknown persons.

    Recognition processing runs in a background thread to avoid blocking the stream.

    Yields:
        JPEG frames with recognition overlay
    """
    # Single viewer mode - simple camera access
    logger.info("New viewer connected to stream")

    try:
        detector = get_detector()  # Singleton - avoid recreating SCRFD!
        recognizer = get_recognizer()
        alert_mgr = get_alert_manager()
        camera = get_camera()  # Use singleton camera to avoid multiple RTSP connections

        # Load all enrolled persons and embeddings
        all_embeddings = db.query(FaceEmbedding).all()

        if len(all_embeddings) == 0:
            logger.warning("No enrolled persons in database")

        # Build embedding database and cache person info
        db_embeddings = []
        person_ids = []
        person_cache = {}  # Cache person info to avoid DB queries on every frame

        for emb in all_embeddings:
            embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)
            db_embeddings.append(embedding_vec)
            person_ids.append(emb.person_id)

            # Cache person info
            if emb.person_id not in person_cache:
                person = db.query(Person).filter(Person.id == emb.person_id).first()
                if person:
                    person_cache[emb.person_id] = {
                        'id': person.id,
                    'name': person.name,
                    'cnic': person.cnic
                }

        logger.info(f"Streaming started with {len(db_embeddings)} embeddings from {len(set(person_ids))} persons")

        frame_count = 0
        last_recognitions = {}  # Cache last recognition per face (dict keyed by face index)
        last_detections = []  # Cache last detection bboxes
        last_logged = {}  # Track when each person was last logged (person_id: timestamp)

        # Queue for sending frames to recognition thread
        recognition_queue = Queue(maxsize=2)  # Small queue to avoid lag
        recognition_results_lock = threading.Lock()

        def recognition_worker():
            """Background worker that processes recognition without blocking stream"""
            while True:
                try:
                    task = recognition_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break

                    frame_for_recognition, detections_list, frame_num = task

                    # Extract embeddings for ALL detected faces
                    face_results = recognizer.extract_multiple_embeddings(frame_for_recognition)

                    # Process matches
                    for face_idx, detection in enumerate(detections_list):
                        mp_bbox = detection.bbox
                        mp_x, mp_y, mp_w, mp_h = mp_bbox

                        # Find matching InsightFace result by bbox overlap
                        best_match_result = None
                        best_iou = 0.3

                        for if_result in face_results:
                            if_x, if_y, if_w, if_h = if_result.bbox
                            # Calculate IoU
                            xi1 = max(mp_x, if_x)
                            yi1 = max(mp_y, if_y)
                            xi2 = min(mp_x + mp_w, if_x + if_w)
                            yi2 = min(mp_y + mp_h, if_y + if_h)
                            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                            union_area = (mp_w * mp_h) + (if_w * if_h) - inter_area
                            iou = inter_area / union_area if union_area > 0 else 0

                            if iou > best_iou:
                                best_iou = iou
                                best_match_result = if_result

                        if best_match_result is not None:
                            best_idx, similarity = recognizer.match_face(
                                best_match_result.embedding,
                                db_embeddings,
                                threshold=settings.face_recognition_threshold
                            )

                            # Update shared recognition cache (thread-safe)
                            face_key = f"{mp_x//50}_{mp_y//50}"
                            with recognition_results_lock:
                                last_recognitions[face_key] = {
                                    'best_idx': best_idx,
                                    'similarity': similarity,
                                    'person_id': person_ids[best_idx] if best_idx >= 0 else None,
                                    'bbox': (mp_x, mp_y, mp_w, mp_h)
                                }

                            # Log and alert (non-blocking)
                            current_time = time.time()
                            log_key = f"{person_ids[best_idx] if best_idx >= 0 else 'unknown'}_{face_key}"

                            if log_key not in last_logged or (current_time - last_logged[log_key]) > 10:
                                try:
                                    log_entry = RecognitionLog(
                                        person_id=person_ids[best_idx] if best_idx >= 0 else None,
                                        timestamp=datetime.utcnow(),
                                        confidence=similarity,
                                        matched=1 if best_idx >= 0 else 0,
                                        camera_source=settings.camera_ip
                                    )
                                    db.add(log_entry)
                                    db.commit()

                                    last_logged[log_key] = current_time

                                    # Trigger alerts
                                    try:
                                        if best_idx >= 0:
                                            person_info = person_cache.get(person_ids[best_idx], {'name': 'Unknown'})
                                            logger.warning(f"🔔 KNOWN PERSON: {person_info['name']} (Confidence: {similarity:.2f})")
                                            alert_mgr.create_alert(
                                                db=db,
                                                event_type='known_person',
                                                person_id=person_ids[best_idx],
                                                person_name=person_info['name'],
                                                confidence=similarity,
                                                num_faces=len(detections_list),
                                                frame=frame_for_recognition.copy()
                                            )
                                        else:
                                            logger.warning(f"⚠️  ALERT: UNKNOWN PERSON DETECTED - Confidence: {similarity:.2f}")
                                            alert_mgr.create_alert(
                                                db=db,
                                                event_type='unknown_person',
                                                person_id=None,
                                                person_name=None,
                                                confidence=similarity,
                                                num_faces=len(detections_list),
                                                frame=frame_for_recognition.copy()
                                            )
                                    except Exception as alert_e:
                                        logger.error(f"Failed to create alert: {alert_e}")
                                except Exception as e:
                                    logger.error(f"Failed to log recognition event: {e}")

                except Empty:
                    # Queue timeout - this is normal, just continue
                    pass
                except Exception as e:
                    logger.error(f"Recognition worker error: {type(e).__name__}: {str(e)}", exc_info=True)

        # Start background recognition thread
        recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        recognition_thread.start()

        try:
            while True:
                # Flush buffer to reduce latency in FFMPEG mode
                ret, frame = camera.read_frame(flush_buffer=True)

                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Skip frames to reduce processing load (process every 2nd frame)
                if frame_count % 2 != 0:
                    # Use cached detections for skipped frames
                    for bbox in last_detections:
                        x, y, w, h = bbox

                        # Find matching cached recognition by position (thread-safe)
                        face_key = f"{x//50}_{y//50}"
                        recog = None

                        with recognition_results_lock:
                            if face_key in last_recognitions:
                                recog = last_recognitions[face_key]
                            else:
                                # Try nearby positions
                                for cached_key, cached_recog in last_recognitions.items():
                                    cached_bbox = cached_recog.get('bbox')
                                    if cached_bbox:
                                        cached_x, cached_y, cached_w, cached_h = cached_bbox
                                        if abs(cached_x - x) < 100 and abs(cached_y - y) < 100:
                                            recog = cached_recog
                                            break

                        if recog:
                            best_idx = recog['best_idx']
                            similarity = recog['similarity']

                            if best_idx >= 0:
                                person_id = recog['person_id']
                                person_info = person_cache.get(person_id, {'name': 'Unknown'})

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 255, 0), -1)
                                cv2.putText(frame, f"KNOWN: {person_info['name']}", (x + 5, y - 28),
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

                    # Encode and yield with balanced quality for smooth streaming
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    continue

                # Use SCRFD (GPU) for fast face detection on processed frames
                detections = detector.detect_faces(frame)

                if detections and len(detections) > 0:
                    # Process all detected faces
                    current_detections = []

                    # Run recognition every 5th frame - NON-BLOCKING using background thread
                    if frame_count % 5 == 0 and len(db_embeddings) > 0:
                        # Submit frame for recognition in background thread (non-blocking)
                        try:
                            # Don't block if queue is full - just skip this recognition cycle
                            recognition_queue.put_nowait((frame.copy(), detections.copy(), frame_count))
                        except:
                            pass  # Queue full, skip this cycle

                    for face_idx, detection in enumerate(detections):
                        bbox = detection.bbox
                        x, y, w, h = bbox
                        current_detections.append((x, y, w, h))

                        # Find matching cached recognition by position (thread-safe)
                        face_key = f"{x//50}_{y//50}"
                        recog = None

                        with recognition_results_lock:
                            # Try exact match first
                            if face_key in last_recognitions:
                                recog = last_recognitions[face_key]
                            else:
                                # Try nearby positions (in case face moved slightly)
                                for cached_key, cached_recog in last_recognitions.items():
                                    cached_bbox = cached_recog.get('bbox')
                                    if cached_bbox:
                                        cached_x, cached_y, cached_w, cached_h = cached_bbox
                                        # Check if bboxes overlap significantly
                                        if abs(cached_x - x) < 100 and abs(cached_y - y) < 100:
                                            recog = cached_recog
                                            break

                        # Draw box with cached or current recognition
                        if recog:
                            best_idx = recog['best_idx']
                            similarity = recog['similarity']

                            if best_idx >= 0:
                                # Known person
                                person_id = recog['person_id']
                                person_info = person_cache.get(person_id, {'name': 'Unknown'})

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.rectangle(frame, (x, y - 45), (x + w, y), (0, 255, 0), -1)
                                cv2.putText(frame, f"KNOWN: {person_info['name']}", (x + 5, y - 28),
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

                    # Update cached detections
                    last_detections = current_detections
                else:
                    # No face detected - clear cache
                    last_recognitions = {}
                    last_detections = []

                # Encode frame to JPEG with balanced quality for smooth streaming
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                # No artificial delay - let it run at natural camera speed for smooth video

        except GeneratorExit:
            logger.info("Streaming stopped by client")
        except Exception as e:
            logger.error(f"Stream error in main loop: {e}")

    except GeneratorExit:
        logger.info("Viewer disconnected from stream")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        logger.info("Stream connection closed")


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


@router.get("/stream/preview")
async def preview_stream():
    """
    Fast preview stream - raw camera frames with NO processing.
    Used for camera capture modal in admin panel.

    Returns:
        MJPEG video stream (raw frames, ~15-20 FPS)
    """
    def generate_preview_stream():
        """Generate raw camera frames without any processing"""
        import time
        # Use the SAME camera instance as snapshot to ensure sync
        camera = get_camera()  # Use singleton camera handler

        # Camera is already connected via get_camera(), no need to connect again

        try:
            while True:
                # Flush buffer for low latency in FFMPEG mode
                success, frame = camera.read_frame(crop_osd=False, flush_buffer=True)

                if not success or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Encode frame as JPEG with balanced quality for smooth preview
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()

                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # No artificial delay - natural camera speed for smooth preview

        except GeneratorExit:
            logger.info("Preview stream disconnected by client")
        except Exception as e:
            logger.error(f"Preview stream error: {e}")
        finally:
            # Clean up camera connection when stream ends
            camera.disconnect()
            logger.info("Preview stream camera disconnected")

    return StreamingResponse(
        generate_preview_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/access-logs")
async def get_access_logs(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get recent access logs (door entry events).
    Shows last 50 recognition events by default.
    """
    try:
        logs = db.query(RecognitionLog).order_by(RecognitionLog.timestamp.desc()).limit(limit).all()

        results = []
        for log in logs:
            person_name = "Unknown"
            person_cnic = None

            if log.person_id:
                person = db.query(Person).filter(Person.id == log.person_id).first()
                if person:
                    person_name = person.name
                    person_cnic = person.cnic

            results.append({
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "person_name": person_name,
                "person_cnic": person_cnic,
                "matched": bool(log.matched),
                "confidence": round(log.confidence, 3),
                "camera_source": log.camera_source
            })

        return {
            "total": len(results),
            "logs": results
        }

    except Exception as e:
        logger.error(f"Error fetching access logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
