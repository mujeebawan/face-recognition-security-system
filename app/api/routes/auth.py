"""
Authentication API endpoints.
Handles user login, logout, token refresh, and user management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator
from typing import Optional
from datetime import timedelta
import logging

from app.core.auth import (
    authenticate_user,
    create_access_token,
    hash_password,
    get_current_user,
    get_current_active_admin,
    verify_admin_password,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.models.database import User
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])
security = HTTPBearer()


# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: dict


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "operator"

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username must be alphanumeric (underscores allowed)')
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

    @validator('role')
    def valid_role(cls, v):
        if v not in ['admin', 'operator', 'viewer']:
            raise ValueError('Role must be admin, operator, or viewer')
        return v


class VerifyPasswordRequest(BaseModel):
    username: str
    password: str


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token.

    Args:
        login_data: Username and password
        db: Database session

    Returns:
        JWT token and user info

    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(db, login_data.username, login_data.password)

    if not user:
        logger.warning(f"Failed login attempt for username: {login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )

    logger.info(f"User '{user.username}' logged in successfully")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.

    Args:
        current_user: Current user from JWT token

    Returns:
        User information
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user (client-side token deletion).

    Args:
        current_user: Current user from JWT token

    Returns:
        Success message
    """
    logger.info(f"User '{current_user.username}' logged out")
    return {"message": "Logged out successfully"}


@router.post("/verify-password")
async def verify_password(
    verify_data: VerifyPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    Verify user password (for critical operations like delete).

    Args:
        verify_data: Username and password
        db: Database session

    Returns:
        Success/failure status
    """
    is_valid = verify_admin_password(db, verify_data.username, verify_data.password)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password or insufficient permissions"
        )

    return {"verified": True}


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: CreateUserRequest,
    current_admin: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """
    Create a new user (admin only).

    Args:
        user_data: User creation data
        current_admin: Current admin user
        db: Database session

    Returns:
        Created user information

    Raises:
        HTTPException: If username already exists
    """
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # Check if email already exists
    if user_data.email:
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )

    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        password_hash=hash_password(user_data.password),
        role=user_data.role,
        is_active=True
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"User '{new_user.username}' created by admin '{current_admin.username}'")

    return new_user


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_admin: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only).

    Args:
        current_admin: Current admin user
        db: Database session

    Returns:
        List of all users
    """
    users = db.query(User).all()
    return users


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    """
    Delete a user (admin only).

    Args:
        user_id: User ID to delete
        current_admin: Current admin user
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If user not found or trying to delete self
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    db.delete(user)
    db.commit()

    logger.info(f"User '{user.username}' deleted by admin '{current_admin.username}'")

    return {"message": f"User '{user.username}' deleted successfully"}
