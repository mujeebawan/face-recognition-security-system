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
from typing import List, Optional, AsyncGenerator
import time
import threading
from queue import Queue, Empty
from collections import deque
import json
import asyncio

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
        # CameraHandler will automatically use settings from database
        camera_handler = CameraHandler()
        if not camera_handler.connect():
            logger.error("Failed to connect to camera on startup")
            camera_handler = None
            raise HTTPException(status_code=503, detail="Camera connection failed")
    elif not camera_handler.is_connected:
        # Reconnect if disconnected (this happens when stream settings change)
        logger.info("Reconnecting to camera...")
        if not camera_handler.connect():
            logger.error("Failed to reconnect to camera")
            raise HTTPException(status_code=503, detail="Camera reconnection failed")
    return camera_handler


def sse_message(data: dict) -> str:
    """Format data as Server-Sent Event message"""
    return f"data: {json.dumps(data)}\n\n"


async def enroll_person_generator(
    name: str,
    cnic: str,
    image: np.ndarray,
    use_sd_augmentation: bool,
    use_controlnet: bool,
    use_liveportrait: bool,
    use_traditional: bool,
    use_multi_model: bool,
    num_sd_variations: int,
    watchlist_status: str,
    threat_level: str,
    criminal_notes: str,
    notes: str,
    db: Session
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE progress events during enrollment.

    Yields SSE messages with format:
    {
        "progress": 0-100,
        "status": "message",
        "stage": "stage_name",
        "detail": "additional_info"
    }
    """
    try:
        import os
        import re

        # Initial progress
        yield sse_message({"progress": 0, "status": "Starting enrollment", "stage": "init"})
        await asyncio.sleep(0.1)  # Allow event to be sent

        # Sanitize person name for folder creation
        folder_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
        person_folder = f"data/person_images/{folder_name}"

        # Check if folder already exists
        if os.path.exists(person_folder):
            yield sse_message({
                "progress": 0,
                "status": "Error: Person already exists",
                "stage": "error",
                "error": f"A person with similar name already exists. Please use a different name."
            })
            return

        # Check if CNIC already exists
        existing = db.query(Person).filter(Person.cnic == cnic).first()
        if existing:
            yield sse_message({
                "progress": 0,
                "status": "Error: CNIC already enrolled",
                "stage": "error",
                "error": f"Person with CNIC {cnic} already enrolled"
            })
            return

        yield sse_message({"progress": 5, "status": "Extracting face embedding", "stage": "extract"})
        await asyncio.sleep(0.1)

        # Extract face embedding from original
        recognizer = get_recognizer()
        result = recognizer.extract_embedding(image)

        if result is None:
            yield sse_message({
                "progress": 0,
                "status": "Error: No face detected",
                "stage": "error",
                "error": "No face detected in image"
            })
            return

        yield sse_message({"progress": 10, "status": "Creating person record", "stage": "create"})
        await asyncio.sleep(0.1)

        # Create person-specific folder
        os.makedirs(person_folder, exist_ok=True)
        logger.info(f"Created person folder: {person_folder}")

        # Create person record with watchlist fields
        original_image_path = f"{person_folder}/original.jpg"
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

        # Save reference image
        cv2.imwrite(original_image_path, image)
        logger.info(f"Saved original image: {original_image_path}")

        # Commit person and original embedding immediately to avoid long transaction locks
        db.commit()
        logger.info(f"Committed person {person.id} and original embedding to database")

        total_embeddings = 1
        augmentation_methods_used = []

        # Calculate progress distribution based on what's enabled
        stages = []
        if use_traditional or use_multi_model:
            stages.append("traditional")
        if use_liveportrait or use_multi_model:
            stages.append("liveportrait")
        if (use_sd_augmentation or use_controlnet) and not use_multi_model:
            stages.append("sd")
        if use_multi_model:
            stages.append("sd_multi")

        # If no augmentation, complete now
        if not stages:
            yield sse_message({"progress": 90, "status": "Finalizing enrollment", "stage": "finalize"})
            await asyncio.sleep(0.1)
            # Already committed above
            yield sse_message({
                "progress": 100,
                "status": "Enrollment complete",
                "stage": "complete",
                "person_id": person.id,
                "total_embeddings": total_embeddings
            })
            return

        # Progress ranges: 15-90% for augmentation stages
        base_progress = 15
        progress_per_stage = 75 / len(stages)
        current_stage_idx = 0

        # Traditional Augmentation
        if use_traditional or use_multi_model:
            stage_start = base_progress + (current_stage_idx * progress_per_stage)
            yield sse_message({
                "progress": int(stage_start),
                "status": "Generating traditional augmentations",
                "stage": "traditional",
                "detail": "Creating rotated and brightness variations"
            })
            await asyncio.sleep(0.1)

            try:
                from app.core.augmentation import FaceAugmentation
                augmentor = FaceAugmentation()
                variations = augmentor.generate_variations(image, num_variations=8)

                for idx, var_img in enumerate(variations[1:], 1):
                    var_result = recognizer.extract_embedding(var_img)
                    if var_result is not None:
                        var_embedding_data = FaceRecognizer.serialize_embedding(var_result.embedding)
                        var_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=var_embedding_data,
                            source=f'traditional_augmented_{idx}',
                            confidence=var_result.confidence
                        )
                        db.add(var_face_embedding)
                        total_embeddings += 1
                        var_img_path = f"{person_folder}/traditional_aug_{idx}.jpg"
                        cv2.imwrite(var_img_path, var_img)

                    # Update progress within this stage
                    progress_in_stage = (idx / len(variations)) * progress_per_stage
                    yield sse_message({
                        "progress": int(stage_start + progress_in_stage),
                        "status": f"Processing traditional variation {idx}/{len(variations)-1}",
                        "stage": "traditional"
                    })
                    await asyncio.sleep(0.05)

                augmentation_methods_used.append("Traditional")
                logger.info(f"Traditional augmentation complete: {total_embeddings} total embeddings")

                # Commit traditional embeddings to avoid long transaction locks
                db.commit()
                logger.info("Committed traditional augmentation embeddings")

            except Exception as aug_e:
                logger.error(f"Traditional augmentation failed: {aug_e}")
                db.rollback()

            current_stage_idx += 1

        # LivePortrait Augmentation
        if use_liveportrait or use_multi_model:
            stage_start = base_progress + (current_stage_idx * progress_per_stage)
            num_variations = min(max(1, num_sd_variations), 10)

            yield sse_message({
                "progress": int(stage_start),
                "status": f"Loading LivePortrait model",
                "stage": "liveportrait",
                "detail": "Initializing 3D-aware face pose generation"
            })
            await asyncio.sleep(0.1)

            try:
                from app.core.liveportrait_augmentation import LivePortraitAugmentor

                augmentor = LivePortraitAugmentor(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_fp16=True
                )

                yield sse_message({
                    "progress": int(stage_start + progress_per_stage * 0.2),
                    "status": f"Generating {num_variations} LivePortrait variations",
                    "stage": "liveportrait",
                    "detail": "This may take 1-2 minutes"
                })
                await asyncio.sleep(0.1)

                generated_images = augmentor.generate_face_angles(
                    reference_image=image,
                    num_variations=num_variations
                )

                for idx, gen_img in enumerate(generated_images):
                    gen_result = recognizer.extract_embedding(gen_img)
                    if gen_result is not None:
                        gen_embedding_data = FaceRecognizer.serialize_embedding(gen_result.embedding)
                        gen_face_embedding = FaceEmbedding(
                            person_id=person.id,
                            embedding=gen_embedding_data,
                            source=f'liveportrait_augmented_{idx+1}',
                            confidence=gen_result.confidence
                        )
                        db.add(gen_face_embedding)
                        total_embeddings += 1
                        gen_img_path = f"{person_folder}/liveportrait_gen_{idx+1}.jpg"
                        cv2.imwrite(gen_img_path, gen_img)

                    # Update progress
                    progress_in_stage = 0.2 + ((idx + 1) / num_variations) * 0.8
                    yield sse_message({
                        "progress": int(stage_start + progress_in_stage * progress_per_stage),
                        "status": f"Generated LivePortrait variation {idx+1}/{num_variations}",
                        "stage": "liveportrait"
                    })
                    await asyncio.sleep(0.05)

                del augmentor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                augmentation_methods_used.append("LivePortrait")
                logger.info(f"LivePortrait augmentation complete: {total_embeddings} total embeddings")

                # Commit LivePortrait embeddings to avoid long transaction locks
                db.commit()
                logger.info("Committed LivePortrait augmentation embeddings")

            except Exception as aug_e:
                logger.error(f"LivePortrait augmentation failed: {aug_e}")
                db.rollback()

            current_stage_idx += 1

        # SD/ControlNet Augmentation (non-multi-model)
        if (use_sd_augmentation or use_controlnet) and not use_multi_model:
            stage_start = base_progress + (current_stage_idx * progress_per_stage)
            num_variations = min(max(1, num_sd_variations), 10)
            augmentation_type = "ControlNet" if use_controlnet else "Stable Diffusion"

            yield sse_message({
                "progress": int(stage_start),
                "status": f"Loading {augmentation_type} model",
                "stage": "sd",
                "detail": "Loading model from SD card (first time may take 1-2 minutes)"
            })
            await asyncio.sleep(0.1)

            try:
                # Check GPU memory
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                    if free_mem < 6.0:
                        yield sse_message({
                            "progress": int(stage_start),
                            "status": "Error: Insufficient GPU memory",
                            "stage": "error",
                            "error": f"Need 6GB GPU memory, only {free_mem:.1f}GB available"
                        })
                        raise HTTPException(status_code=507, detail="Insufficient GPU memory")

                if use_controlnet:
                    from app.core.controlnet_augmentation import ControlNetFaceAugmentor
                    augmentor = ControlNetFaceAugmentor(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=True
                    )
                else:
                    from app.core.generative_augmentation import StableDiffusionAugmentor
                    augmentor = StableDiffusionAugmentor(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=True
                    )

                if not augmentor.load_model():
                    logger.error("Failed to load model")
                else:
                    yield sse_message({
                        "progress": int(stage_start + progress_per_stage * 0.15),
                        "status": f"Generating {num_variations} {augmentation_type} variations",
                        "stage": "sd",
                        "detail": "This will take 2-5 minutes depending on settings"
                    })
                    await asyncio.sleep(0.1)

                    gen_params = {
                        'reference_image': image,
                        'num_variations': num_variations,
                        'num_inference_steps': 30 if use_controlnet else 20,
                        'guidance_scale': 7.5
                    }
                    if use_controlnet:
                        gen_params['controlnet_scale'] = 0.9

                    generated_images = augmentor.generate_face_angles(**gen_params)

                    for idx, gen_img in enumerate(generated_images):
                        gen_result = recognizer.extract_embedding(gen_img)
                        if gen_result is not None:
                            gen_embedding_data = FaceRecognizer.serialize_embedding(gen_result.embedding)
                            source_name = 'controlnet' if use_controlnet else 'img2img'
                            gen_face_embedding = FaceEmbedding(
                                person_id=person.id,
                                embedding=gen_embedding_data,
                                source=f'{source_name}_augmented_{idx+1}',
                                confidence=gen_result.confidence
                            )
                            db.add(gen_face_embedding)
                            total_embeddings += 1
                            gen_img_path = f"{person_folder}/{source_name}_gen_{idx+1}.jpg"
                            cv2.imwrite(gen_img_path, gen_img)

                        # Update progress
                        progress_in_stage = 0.15 + ((idx + 1) / num_variations) * 0.85
                        yield sse_message({
                            "progress": int(stage_start + progress_in_stage * progress_per_stage),
                            "status": f"Generated {augmentation_type} variation {idx+1}/{num_variations}",
                            "stage": "sd"
                        })
                        await asyncio.sleep(0.05)

                    augmentor.unload_model()
                    del augmentor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()

                    augmentation_methods_used.append(augmentation_type.upper())
                    logger.info(f"{augmentation_type} augmentation complete: {total_embeddings} total embeddings")

                    # Commit SD/ControlNet embeddings to avoid long transaction locks
                    db.commit()
                    logger.info(f"Committed {augmentation_type} augmentation embeddings")

            except HTTPException:
                raise
            except Exception as aug_e:
                logger.error(f"SD augmentation failed: {aug_e}")
                db.rollback()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            current_stage_idx += 1

        # Multi-model SD generation
        if use_multi_model:
            stage_start = base_progress + (current_stage_idx * progress_per_stage)

            yield sse_message({
                "progress": int(stage_start),
                "status": "Multi-Model: Loading SD img2img",
                "stage": "sd_multi",
                "detail": "Final augmentation stage"
            })
            await asyncio.sleep(0.1)

            try:
                from app.core.generative_augmentation import StableDiffusionAugmentor

                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                    if free_mem < 4.0:
                        logger.warning(f"Low GPU memory, skipping SD in multi-model")
                    else:
                        augmentor = StableDiffusionAugmentor(
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            use_fp16=True
                        )

                        if augmentor.load_model():
                            num_variations = min(3, num_sd_variations)

                            yield sse_message({
                                "progress": int(stage_start + progress_per_stage * 0.2),
                                "status": f"Generating {num_variations} SD variations (multi-model)",
                                "stage": "sd_multi"
                            })
                            await asyncio.sleep(0.1)

                            generated_images = augmentor.generate_face_angles(
                                reference_image=image,
                                num_variations=num_variations,
                                num_inference_steps=15,
                                guidance_scale=7.0
                            )

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
                                    gen_img_path = f"{person_folder}/img2img_gen_{idx+1}.jpg"
                                    cv2.imwrite(gen_img_path, gen_img)

                                progress_in_stage = 0.2 + ((idx + 1) / num_variations) * 0.8
                                yield sse_message({
                                    "progress": int(stage_start + progress_in_stage * progress_per_stage),
                                    "status": f"Generated SD variation {idx+1}/{num_variations}",
                                    "stage": "sd_multi"
                                })
                                await asyncio.sleep(0.05)

                            augmentor.unload_model()
                            del augmentor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()

                            augmentation_methods_used.append("SD-IMG2IMG")

                            # Commit multi-model SD embeddings
                            db.commit()
                            logger.info("Committed multi-model SD augmentation embeddings")

            except Exception as aug_e:
                logger.error(f"Multi-model SD failed: {aug_e}")
                db.rollback()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Finalize
        yield sse_message({"progress": 95, "status": "Finalizing enrollment", "stage": "finalize"})
        await asyncio.sleep(0.1)

        # All embeddings already committed in chunks above
        # No need for final commit unless there were no augmentations

        yield sse_message({
            "progress": 100,
            "status": "Enrollment complete!",
            "stage": "complete",
            "person_id": person.id,
            "name": name,
            "total_embeddings": total_embeddings,
            "augmentation_methods": augmentation_methods_used
        })

        logger.info(f"Enrolled {name} with {total_embeddings} embeddings using {len(augmentation_methods_used)} methods")

    except HTTPException as he:
        yield sse_message({
            "progress": 0,
            "status": f"Error: {he.detail}",
            "stage": "error",
            "error": he.detail
        })
    except Exception as e:
        db.rollback()
        logger.error(f"Enrollment error: {e}")
        import traceback
        traceback.print_exc()
        yield sse_message({
            "progress": 0,
            "status": f"Error: {str(e)}",
            "stage": "error",
            "error": str(e)
        })


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
    Enroll a new person with their face image using Server-Sent Events for real-time progress.

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
        StreamingResponse with Server-Sent Events containing progress updates
    """
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Return SSE stream
    return StreamingResponse(
        enroll_person_generator(
            name=name,
            cnic=cnic,
            image=image,
            use_sd_augmentation=use_sd_augmentation,
            use_controlnet=use_controlnet,
            use_liveportrait=use_liveportrait,
            use_traditional=use_traditional,
            use_multi_model=use_multi_model,
            num_sd_variations=num_sd_variations,
            watchlist_status=watchlist_status,
            threat_level=threat_level,
            criminal_notes=criminal_notes,
            notes=notes,
            db=db
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


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
        best_idx, similarity, embedding_id = recognizer.match_face(
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
        camera = CameraHandler()  # Uses settings from database

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

        best_idx, similarity, embedding_id = recognizer.match_face(
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
        camera = CameraHandler()  # Uses settings from database

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


def generate_video_stream(db: Session, quality_mode: str = "smooth"):
    """
    Generate MJPEG video stream with real-time face recognition.
    Uses SCRFD (GPU) for fast detection, InsightFace for recognition.
    Triggers alerts for unknown persons.

    Recognition processing runs in a background thread to avoid blocking the stream.

    OPTIMIZATIONS:
    - FAISS GPU for <1ms embedding search (vs 100-200ms)
    - IoU-based face tracking (vs grid hashing)
    - Recognition every 5 frames
    - Queue size 10 (vs 2)
    - Dynamic quality modes for optimal performance

    Quality modes:
    - smooth: 720p @ 65 quality (~25-30 FPS) - Recommended
    - balanced: 720p @ 75 quality (~23-27 FPS)
    - quality: 1080p @ 70 quality (~20-25 FPS)
    - max: 2K @ 80 quality (~15-20 FPS)

    Yields:
        JPEG frames with recognition overlay
    """
    from app.core.settings_manager import get_setting
    from app.core.faiss_cache import FaceRecognitionCache, find_matching_face

    # Quality mode settings
    quality_settings = {
        'smooth': {'width': 1280, 'height': 720, 'jpeg_quality': 65},
        'balanced': {'width': 1280, 'height': 720, 'jpeg_quality': 75},
        'quality': {'width': 1920, 'height': 1080, 'jpeg_quality': 70},
        'max': {'width': 2560, 'height': 1440, 'jpeg_quality': 80},
    }

    # Get settings for selected quality mode
    settings = quality_settings.get(quality_mode, quality_settings['smooth'])
    target_width = settings['width']
    target_height = settings['height']
    jpeg_quality = settings['jpeg_quality']

    # Single viewer mode - simple camera access
    logger.info(f"New viewer connected - Quality mode: {quality_mode} ({target_width}x{target_height} @ {jpeg_quality})")

    try:
        detector = get_detector()  # Singleton - avoid recreating SCRFD!
        recognizer = get_recognizer()
        alert_mgr = get_alert_manager()
        camera = get_camera()  # Use singleton camera to avoid multiple RTSP connections

        # Load all enrolled persons and embeddings
        all_embeddings = db.query(FaceEmbedding).all()

        if len(all_embeddings) == 0:
            logger.warning("No enrolled persons in database")

        # Build FAISS GPU index for fast similarity search
        embeddings_dict = {}  # {person_id: [embedding1, embedding2, ...]}
        person_cache = {}  # Cache person info to avoid DB queries on every frame

        for emb in all_embeddings:
            embedding_vec = FaceRecognizer.deserialize_embedding(emb.embedding)

            if emb.person_id not in embeddings_dict:
                embeddings_dict[emb.person_id] = []
            embeddings_dict[emb.person_id].append(embedding_vec)

            # Cache person info
            if emb.person_id not in person_cache:
                person = db.query(Person).filter(Person.id == emb.person_id).first()
                if person:
                    person_cache[emb.person_id] = {
                        'id': person.id,
                        'name': person.name,
                        'cnic': person.cnic
                    }

        # Initialize FAISS GPU cache
        faiss_cache = FaceRecognitionCache(embedding_dim=512, use_gpu=True)
        faiss_cache.build_index(embeddings_dict, person_cache)

        stats = faiss_cache.get_stats()
        logger.info(f"✅ FAISS {stats['device']} index built: {stats['embedding_count']} embeddings "
                   f"from {stats['person_count']} persons")

        # Get frame skip setting dynamically
        frame_skip_setting = get_setting("frame_skip", 0)
        logger.info(f"Using frame skip setting: {frame_skip_setting} (0 = process all frames)")

        frame_count = 0
        last_recognitions = {}  # Cache last recognition per face (dict keyed by bbox tuple)
        last_detections = []  # Cache last detection bboxes
        last_logged = {}  # Track when each person was last logged (person_id: timestamp)

        # Video recording state
        frame_buffer = deque(maxlen=45)  # Buffer for ~3 seconds at 15 FPS (for "before" footage)
        video_recording_state = {}  # Track active video recordings: {alert_id: {'frames': [], 'target_frame_count': int}}
        video_recording_lock = threading.Lock()

        # Queue for sending frames to recognition thread - INCREASED from 2 to 10
        recognition_queue = Queue(maxsize=10)  # Larger queue for better throughput
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
                            # Use FAISS GPU for ultra-fast similarity search (<1ms vs 100-200ms!)
                            person_id, similarity, person_name = faiss_cache.search(
                                best_match_result.embedding,
                                threshold=settings.face_recognition_threshold
                            )

                            # Update shared recognition cache (thread-safe) - Use bbox as key for IoU matching
                            bbox_tuple = (mp_x, mp_y, mp_w, mp_h)
                            with recognition_results_lock:
                                last_recognitions[bbox_tuple] = {
                                    'person_id': person_id,
                                    'similarity': similarity,
                                    'person_name': person_name,
                                    'bbox': bbox_tuple,
                                    'matched': person_id is not None
                                }

                            # Log and alert (non-blocking)
                            current_time = time.time()
                            log_key = f"{person_id if person_id else 'unknown'}_{mp_x}_{mp_y}"

                            if log_key not in last_logged or (current_time - last_logged[log_key]) > 10:
                                try:
                                    log_entry = RecognitionLog(
                                        person_id=person_id,
                                        timestamp=datetime.now(),
                                        confidence=similarity,
                                        matched=1 if person_id is not None else 0,
                                        camera_source=settings.camera_ip
                                    )
                                    db.add(log_entry)
                                    db.commit()

                                    last_logged[log_key] = current_time

                                    # Trigger alerts
                                    try:
                                        # Prepare video frames from buffer (for video recording)
                                        video_frames_for_alert = None
                                        with video_recording_lock:
                                            if len(frame_buffer) > 0:
                                                # Copy buffered frames for video
                                                video_frames_for_alert = list(frame_buffer)

                                        if person_id is not None:
                                            logger.warning(f"[DETECTED] KNOWN PERSON: {person_name} (Confidence: {similarity:.2f})")
                                            alert_mgr.create_alert(
                                                db=db,
                                                event_type='known_person',
                                                person_id=person_id,
                                                person_name=person_name,
                                                confidence=similarity,
                                                num_faces=len(detections_list),
                                                frame=frame_for_recognition.copy(),
                                                video_frames=video_frames_for_alert
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
                                                frame=frame_for_recognition.copy(),
                                                video_frames=video_frames_for_alert
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

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0.0

        try:
            while True:
                loop_start = time.time()

                # Flush buffer to reduce latency in FFMPEG mode
                read_start = time.time()
                ret, frame = camera.read_frame(flush_buffer=True)
                read_time = (time.time() - read_start) * 1000

                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                fps_frame_count += 1

                # Calculate FPS every 30 frames
                if fps_frame_count % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    fps_display = 30 / elapsed if elapsed > 0 else 0
                    logger.info(f"📊 Stream FPS: {fps_display:.1f} | Frame read: {read_time:.1f}ms | Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    fps_start_time = time.time()
                    fps_frame_count = 0

                # Resize frame based on quality mode
                # Keep original for high-res recording
                original_frame = frame.copy()
                stream_height, stream_width = frame.shape[:2]

                if stream_width > target_width or stream_height > target_height:
                    # Calculate scaling to fit within target while maintaining aspect ratio
                    scale = min(target_width / stream_width, target_height / stream_height)
                    new_width = int(stream_width * scale)
                    new_height = int(stream_height * scale)

                    # Resize for streaming (fast interpolation)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    scale_factor = scale
                else:
                    scale_factor = 1.0

                # Add ORIGINAL high-res frame to buffer for video recording (thread-safe)
                with video_recording_lock:
                    frame_buffer.append(original_frame)

                # Skip frames to reduce processing load based on settings
                # frame_skip=0 means process all frames, frame_skip=1 means skip 1 frame (process every 2nd), etc.
                should_skip_frame = False
                if frame_skip_setting > 0:
                    # Skip frame if not at the right interval
                    if (frame_count - 1) % (frame_skip_setting + 1) != 0:
                        should_skip_frame = True

                if should_skip_frame:
                    # Use cached detections for skipped frames
                    for bbox in last_detections:
                        x, y, w, h = bbox

                        # Find matching cached recognition using IoU (thread-safe)
                        recog = None
                        with recognition_results_lock:
                            recog = find_matching_face((x, y, w, h), last_recognitions, iou_threshold=0.5)

                        if recog and recog.get('matched'):
                            person_id = recog['person_id']
                            similarity = recog['similarity']
                            person_name = recog.get('person_name', 'Unknown')

                            if person_id is not None:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                                cv2.rectangle(frame, (x, y - 70), (x + w, y), (0, 255, 0), -1)
                                cv2.putText(frame, f"KNOWN: {person_name}", (x + 8, y - 40),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                                cv2.putText(frame, f"Match: {similarity:.2f}", (x + 8, y - 12),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
                            else:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.rectangle(frame, (x, y - 70), (x + w, y), (0, 0, 255), -1)
                                cv2.putText(frame, "UNKNOWN PERSON", (x + 8, y - 40),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                                cv2.putText(frame, f"Sim: {similarity:.2f}", (x + 8, y - 12),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

                    # Encode and yield with quality from settings
                    encode_start = time.time()
                    _, buffer = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 0   # Disable optimization for speed
                    ])
                    encode_time = (time.time() - encode_start) * 1000

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    continue

                # Use SCRFD (GPU) for fast face detection
                # Detect every 5th frame to maximize FPS while maintaining accuracy
                detections = None
                detect_time = 0
                if frame_count % 5 == 0:  # Detect every 5 frames (was every 2)
                    detect_start = time.time()
                    detections = detector.detect_faces(frame)
                    detect_time = (time.time() - detect_start) * 1000
                    if frame_count % 30 == 0:
                        logger.info(f"⏱️  Detection: {detect_time:.1f}ms")

                # Update detections cache if new detections were made
                if detections and len(detections) > 0:
                    # Process all detected faces
                    current_detections = []

                    # Run recognition every 5th frame (same as detection) - OPTIMIZED!
                    # With FAISS GPU, we can afford to run recognition frequently (<1ms vs 100-200ms)
                    if frame_count % 5 == 0:
                        # Submit frame for recognition in background thread (non-blocking)
                        try:
                            # With larger queue (10 vs 2), we're less likely to drop frames
                            recognition_queue.put_nowait((frame.copy(), detections.copy(), frame_count))
                        except:
                            # Log when queue is full (helps debugging)
                            logger.debug(f"Recognition queue full at frame {frame_count}, skipping")

                    for face_idx, detection in enumerate(detections):
                        bbox = detection.bbox
                        x, y, w, h = bbox
                        current_detections.append((x, y, w, h))

                    # Update cached detections with NEW detections
                    last_detections = current_detections
                elif detections is not None and len(detections) == 0:
                    # Detected but found no faces - clear cache
                    last_detections = []
                    last_recognitions = {}

                # ALWAYS draw boxes using cached detections (for smooth display on ALL frames)
                for bbox in last_detections:
                    x, y, w, h = bbox

                    # Find matching cached recognition using IoU (thread-safe)
                    recog = None
                    with recognition_results_lock:
                        recog = find_matching_face((x, y, w, h), last_recognitions, iou_threshold=0.5)

                    # Draw box with cached or current recognition
                    if recog and recog.get('matched'):
                        person_id = recog['person_id']
                        similarity = recog['similarity']
                        person_name = recog.get('person_name', 'Unknown')

                        if person_id is not None:
                            # Known person
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                            cv2.rectangle(frame, (x, y - 70), (x + w, y), (0, 255, 0), -1)
                            cv2.putText(frame, f"KNOWN: {person_name}", (x + 8, y - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                            cv2.putText(frame, f"Match: {similarity:.2f}", (x + 8, y - 12),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
                        else:
                            # Unknown person
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                            cv2.rectangle(frame, (x, y - 70), (x + w, y), (0, 0, 255), -1)
                            cv2.putText(frame, "UNKNOWN PERSON", (x + 8, y - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                            cv2.putText(frame, f"Sim: {similarity:.2f}", (x + 8, y - 12),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                    else:
                        # Just detected, no recognition yet
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)
                        cv2.putText(frame, "Detecting...", (x + 8, y - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

                # Encode frame to JPEG with quality based on selected mode
                encode_start = time.time()
                _, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0   # Disable optimization for speed
                ])
                encode_time = (time.time() - encode_start) * 1000

                if frame_count % 30 == 0:
                    logger.info(f"⏱️  JPEG encode: {encode_time:.1f}ms (quality {jpeg_quality})")

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
async def live_stream(quality: str = "smooth", db: Session = Depends(get_db)):
    """
    Live video stream with real-time face recognition overlay.

    Args:
        quality: Performance mode - smooth, balanced, quality, or max

    Returns:
        MJPEG video stream showing Known/Unknown labels
    """
    return StreamingResponse(
        generate_video_stream(db, quality_mode=quality),
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
