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
    use_traditional: bool = Form(False),
    use_multi_model: bool = Form(False),
    num_sd_variations: int = Form(5),
    # Watchlist fields
    watchlist_status: str = Form("none"),
    threat_level: str = Form("none"),
    criminal_notes: str = Form(""),
    notes: str = Form(""),
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
        use_traditional: Use traditional augmentation (rotation, brightness, etc.) (default: False)
        use_multi_model: Use ALL augmentation models sequentially for critical cases (default: False)
        num_sd_variations: Number of variations to generate if augmentation enabled (default: 5, max: 10)
        db: Database session

    Returns:
        Enrollment result with person ID and total embeddings
    """
    try:
        import os
        import re

        # Sanitize person name for folder creation (remove special characters)
        folder_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
        person_folder = f"data/person_images/{folder_name}"

        # Check if folder already exists (name conflict)
        if os.path.exists(person_folder):
            raise HTTPException(
                status_code=400,
                detail=f"A person with similar name already exists. Please use a different name to avoid conflicts. Suggested: {name}_2 or {name}_{cnic[-4:]}"
            )

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

        # Create person-specific folder
        os.makedirs(person_folder, exist_ok=True)
        logger.info(f"Created person folder: {person_folder}")

        # Create person record with watchlist fields
        original_image_path = f"{person_folder}/original_{file.filename}"
        person = Person(
            name=name,
            cnic=cnic,
            reference_image_path=original_image_path,
            watchlist_status=watchlist_status,
            threat_level=threat_level,
            criminal_notes=criminal_notes if criminal_notes else None,
            added_to_watchlist_at=datetime.now() if watchlist_status != 'none' else None
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

        # Save reference image to person folder
        cv2.imwrite(original_image_path, image)
        logger.info(f"Saved original image: {original_image_path}")

        total_embeddings = 1
        augmentation_time = 0
        augmentation_methods_used = []

        # Multi-model mode: Use ALL augmentation models sequentially for maximum accuracy
        if use_multi_model:
            logger.info(f"MULTI-MODEL AUGMENTATION MODE activated for {name} - Using ALL models sequentially")
            use_traditional = True
            use_liveportrait = True
            # SD will be handled separately in Step 4 for multi-model
            # Note: We'll use img2img for multi-model to save GPU memory, not ControlNet

        # If user selected multiple individual methods, they can all run
        # (e.g., user can select both LivePortrait AND ControlNet)

        # Step 1: Traditional Augmentation (if requested or multi-model)
        if use_traditional or use_multi_model:
            try:
                from app.core.augmentation import FaceAugmentation

                logger.info(f"Generating traditional augmentations for {name}...")
                augmentor = FaceAugmentation()

                start_time = time.time()
                variations = augmentor.generate_variations(image, num_variations=8)
                trad_time = time.time() - start_time
                augmentation_time += trad_time

                logger.info(f"Traditional augmentation completed in {trad_time:.2f}s ({len(variations)} images)")

                # Process each variation
                for idx, var_img in enumerate(variations[1:], 1):  # Skip first (original)
                    var_result = recognizer.extract_embedding(var_img)

                    if var_result is not None:
                        # Store embedding
                        var_embedding_data = FaceRecognizer.serialize_embedding(var_result.embedding)
                        var_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=var_embedding_data,
                            source=f'traditional_augmented_{idx}',
                            confidence=var_result.confidence
                        )
                        db.add(var_face_embedding)
                        total_embeddings += 1

                        # Save generated image to person folder
                        var_img_path = f"{person_folder}/traditional_aug_{idx}.jpg"
                        cv2.imwrite(var_img_path, var_img)

                augmentation_methods_used.append("Traditional")
                logger.info(f"Traditional augmentation complete: {total_embeddings} total embeddings")

            except Exception as aug_e:
                logger.error(f"Traditional augmentation failed: {aug_e}, continuing with other methods")
                import traceback
                traceback.print_exc()

        # Step 2: Generate LivePortrait augmented faces if requested
        if use_liveportrait or use_multi_model:
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

                        # Save generated image to person folder
                        gen_img_path = f"{person_folder}/liveportrait_gen_{idx+1}.jpg"
                        cv2.imwrite(gen_img_path, gen_img)

                # Cleanup and free GPU memory
                del augmentor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Freed GPU memory after LivePortrait")

                augmentation_methods_used.append("LivePortrait")
                logger.info(f"LivePortrait augmentation complete: {total_embeddings} total embeddings")

            except Exception as aug_e:
                logger.error(f"LivePortrait augmentation failed: {aug_e}, continuing with other methods")
                import traceback
                traceback.print_exc()
                # Continue with enrollment even if augmentation fails

        # Step 3: Generate SD augmented faces if requested (runs independently or in multi-model)
        if (use_sd_augmentation or use_controlnet) and not use_multi_model:
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

                            # Save generated image to person folder
                            gen_img_path = f"{person_folder}/{augmentation_type}_gen_{idx+1}.jpg"
                            cv2.imwrite(gen_img_path, gen_img)

                    # Unload model and free GPU memory aggressively
                    augmentor.unload_model()
                    del augmentor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        logger.info(f"Freed GPU memory after {augmentation_type.upper()}")

                    augmentation_methods_used.append(augmentation_type.upper())
                    logger.info(f"{augmentation_type.upper()} augmentation complete: {total_embeddings} total embeddings")

            except Exception as aug_e:
                logger.error(f"Augmentation failed: {aug_e}, continuing with other methods")
                import traceback
                traceback.print_exc()
                # Continue with enrollment even if augmentation fails
                # Make sure to free GPU memory even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Step 4: Multi-model SD generation (for multi_model mode only)
        if use_multi_model:
            try:
                import time
                from app.core.generative_augmentation import StableDiffusionAugmentor

                # In multi-model, use img2img (lighter than ControlNet)
                logger.info(f"Multi-Model Mode: Generating SD img2img augmented angles for {name}...")

                # Check GPU memory
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                    logger.info(f"Available GPU memory before SD: {free_mem:.2f} GB")

                    if free_mem < 4.0:
                        logger.warning(f"Low GPU memory ({free_mem:.1f}GB), skipping SD augmentation in multi-model mode")
                        raise HTTPException(
                            status_code=507,
                            detail=f"Insufficient GPU memory for Stable Diffusion in multi-model mode. Available: {free_mem:.1f}GB, Required: ~4GB"
                        )

                # Initialize SD img2img augmentor
                augmentor = StableDiffusionAugmentor(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_fp16=True
                )

                # Load model
                if augmentor.load_model():
                    # Generate variations (fewer in multi-model to save time)
                    num_variations = min(3, num_sd_variations)
                    start_time = time.time()

                    generated_images = augmentor.generate_face_angles(
                        reference_image=image,
                        num_variations=num_variations,
                        num_inference_steps=15,  # Faster for multi-model
                        guidance_scale=7.0
                    )
                    sd_time = time.time() - start_time
                    augmentation_time += sd_time

                    logger.info(f"SD generation completed in {sd_time:.2f}s ({len(generated_images)} images)")

                    # Process each generated image
                    for idx, gen_img in enumerate(generated_images):
                        gen_result = recognizer.extract_embedding(gen_img)

                        if gen_result is not None:
                            gen_embedding_data = FaceRecognizer.serialize_embedding(gen_result.embedding)
                            gen_face_embedding = FaceEmbedding(
                                person_id=person.id,
                                embedding=gen_embedding_data,
                                source=f'img2img_augmented_{idx+1}',
                                confidence=gen_result.confidence
                            )
                            db.add(gen_face_embedding)
                            total_embeddings += 1

                            # Save generated image to person folder
                            gen_img_path = f"{person_folder}/img2img_gen_{idx+1}.jpg"
                            cv2.imwrite(gen_img_path, gen_img)

                    # Unload and free memory
                    augmentor.unload_model()
                    del augmentor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        logger.info("Freed GPU memory after SD (multi-model)")

                    augmentation_methods_used.append("SD-IMG2IMG")
                    logger.info(f"Multi-model SD complete: {total_embeddings} total embeddings")

            except HTTPException:
                raise
            except Exception as aug_e:
                logger.error(f"SD augmentation in multi-model failed: {aug_e}")
                import traceback
                traceback.print_exc()
                # Free memory even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        db.commit()

        logger.info(f"Enrolled person: {name} (CNIC: {cnic}, ID: {person.id}, Embeddings: {total_embeddings}, Folder: {person_folder})")

        response = {
            "success": True,
            "message": f"Person {name} enrolled successfully with {total_embeddings} face embeddings",
            "person_id": person.id,
            "name": name,
            "cnic": cnic,
            "person_folder": person_folder,
            "confidence": result.confidence,
            "total_embeddings": total_embeddings,
            "multi_model_used": use_multi_model,
            "augmentation_methods": augmentation_methods_used,
            "traditional_used": use_traditional and not use_multi_model,
            "liveportrait_used": use_liveportrait and not use_multi_model,
            "sd_augmentation_used": use_sd_augmentation and not use_multi_model,
            "controlnet_used": use_controlnet and use_sd_augmentation and not use_multi_model
        }

        if augmentation_time > 0:
            response["total_augmentation_time"] = round(augmentation_time, 2)
            if len(augmentation_methods_used) > 0:
                response["augmentation_summary"] = f"Used {len(augmentation_methods_used)} methods: {', '.join(augmentation_methods_used)}"

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
                "timestamp": datetime.now()
            }
        else:
            return {
                "success": True,
                "matched": False,
                "message": "Face not recognized",
                "best_similarity": round(similarity, 3),
                "timestamp": datetime.now()
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
    import re
    from pathlib import Path
    images = []

    # Determine person folder
    folder_name = re.sub(r'[^\w\s-]', '', person.name).strip().replace(' ', '_')
    person_folder = f"data/person_images/{folder_name}"

    # Check if person folder exists (new structure) or use old structure
    use_new_structure = os.path.exists(person_folder)

    # Add original image
    if person.reference_image_path and os.path.exists(person.reference_image_path):
        images.append({
            "source": "original",
            "path": person.reference_image_path,
            "url": f"/api/image/{person.id}/original"
        })

    # Add AI-generated and traditional augmented images
    for emb in embeddings:
        # Handle all augmentation types (new and legacy)
        source = emb.source

        # Skip original embedding
        if source == 'original':
            continue

        # Determine image path based on structure
        if use_new_structure:
            # New folder structure
            if source.startswith('traditional_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"{person_folder}/traditional_aug_{idx}.jpg"
            elif source.startswith('liveportrait_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"{person_folder}/liveportrait_gen_{idx}.jpg"
            elif source.startswith('controlnet_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"{person_folder}/controlnet_gen_{idx}.jpg"
            elif source.startswith('img2img_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"{person_folder}/img2img_gen_{idx}.jpg"
            else:
                continue
        else:
            # Legacy flat structure
            if source.startswith('controlnet_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"data/images/{person.cnic}_controlnet_gen_{idx}.jpg"
            elif source.startswith('img2img_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"data/images/{person.cnic}_img2img_gen_{idx}.jpg"
            elif source.startswith('liveportrait_augmented_'):
                idx = source.split('_')[-1]
                img_path = f"data/images/{person.cnic}_liveportrait_gen_{idx}.jpg"
            else:
                continue

        if os.path.exists(img_path):
            images.append({
                "source": source,
                "path": img_path,
                "url": f"/api/image/{person.id}/{os.path.basename(img_path)}",
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
    """Delete an enrolled person and all associated files"""
    import os
    import re
    import shutil

    person = db.query(Person).filter(Person.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Delete person folder and all its contents from SD card
    folder_name = re.sub(r'[^\w\s-]', '', person.name).strip().replace(' ', '_')
    person_folder = f"data/person_images/{folder_name}"

    if os.path.exists(person_folder):
        try:
            shutil.rmtree(person_folder)
            logger.info(f"Deleted person folder: {person_folder}")
        except Exception as e:
            logger.error(f"Failed to delete folder {person_folder}: {e}")
            # Continue with database deletion even if folder deletion fails

    # Also delete legacy reference image if it exists
    if person.reference_image_path and os.path.exists(person.reference_image_path):
        try:
            os.remove(person.reference_image_path)
            logger.info(f"Deleted reference image: {person.reference_image_path}")
        except Exception as e:
            logger.error(f"Failed to delete reference image: {e}")

    # Delete from database (cascade will delete embeddings and logs)
    db.delete(person)
    db.commit()

    return {
        "success": True,
        "message": f"Person {person.name} deleted successfully from database and storage"
    }


@router.get("/image/{person_id}/{image_filename}")
async def get_person_image(
    person_id: int,
    image_filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Serve person images (original and augmented images from person folder or legacy structure)"""
    import os
    import re
    from fastapi.responses import FileResponse

    # Get person
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Determine folder structure
    folder_name = re.sub(r'[^\w\s-]', '', person.name).strip().replace(' ', '_')
    person_folder = f"data/person_images/{folder_name}"

    # Handle special case: "original"
    if image_filename == "original":
        if not person.reference_image_path or not os.path.exists(person.reference_image_path):
            raise HTTPException(status_code=404, detail="Original image not found")
        return FileResponse(person.reference_image_path, media_type="image/jpeg")

    # Try new folder structure first
    if os.path.exists(person_folder):
        image_path = f"{person_folder}/{image_filename}"
        if os.path.exists(image_path):
            return FileResponse(image_path, media_type="image/jpeg")

    # Fall back to legacy structure
    legacy_path = f"data/images/{person.cnic}_{image_filename}"
    if os.path.exists(legacy_path):
        return FileResponse(legacy_path, media_type="image/jpeg")

    # Image not found
    raise HTTPException(status_code=404, detail=f"Image not found: {image_filename}")


@router.post("/enroll/multiple")
async def enroll_person_multiple_images(
    name: str = Form(...),
    cnic: str = Form(...),
    files: List[UploadFile] = File(...),
    use_augmentation: bool = Form(True),
    # Watchlist fields
    watchlist_status: str = Form("none"),
    threat_level: str = Form("none"),
    criminal_notes: str = Form(""),
    notes: str = Form(""),
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

        # Create person record with watchlist fields
        person = Person(
            name=name,
            cnic=cnic,
            reference_image_path=f"data/images/{cnic}_multiple",
            watchlist_status=watchlist_status,
            threat_level=threat_level,
            criminal_notes=criminal_notes if criminal_notes else None,
            added_to_watchlist_at=datetime.now() if watchlist_status != 'none' else None
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
                                        timestamp=datetime.now(),
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
                                            logger.warning(f"[DETECTED] KNOWN PERSON: {person_info['name']} (Confidence: {similarity:.2f})")
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
