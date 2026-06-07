"""
app/utils/ephemeral_storage.py
Context manager for secure, ephemeral file storage.
Uses Linux RAM-disk (/dev/shm) when available, otherwise falls back to the system temp directory,
and ensures files are overwritten with zeros (shredded) before deletion.
"""
import os
import tempfile
import logging
import uuid
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_ephemeral_dir() -> str:
    """
    Returns the path to the most secure ephemeral directory available.
    Prefers /dev/shm on Linux systems.
    """
    dev_shm = "/dev/shm"
    if os.path.exists(dev_shm) and os.path.isdir(dev_shm) and os.access(dev_shm, os.W_OK):
        logger.info("Using Linux shared memory RAM-disk (/dev/shm) for ephemeral storage.")
        return dev_shm
    
    # Fallback to standard system temp directory
    temp_dir = tempfile.gettempdir()
    logger.info("Using system temporary directory (%s) for ephemeral storage.", temp_dir)
    return temp_dir

def secure_shred(file_path: str) -> None:
    """
    Overwrites the contents of a file with zeros and forces synchronization
    to physical disk before removing it to prevent data recovery.
    """
    if not os.path.exists(file_path):
        return
        
    try:
        size = os.path.getsize(file_path)
        # Open file in write-binary mode without buffering and overwrite
        with open(file_path, "wb") as f:
            if size > 0:
                f.write(b"\x00" * size)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass # Some systems don't support fsync on temporary mounts
        os.remove(file_path)
        logger.info("File %s successfully shredded and deleted.", file_path)
    except Exception as exc:
        logger.error("Failed to secure-shred file %s: %s. Performing standard remove.", file_path, exc)
        try:
            os.remove(file_path)
        except Exception:
            pass

@contextmanager
def ephemeral_file(content: bytes, suffix: str = ".tmp"):
    """
    Context manager that creates a temporary file with the given content,
    yields the absolute file path, and securely shreds it on exit.
    """
    ephemeral_dir = get_ephemeral_dir()
    unique_filename = f"mindscan_{uuid.uuid4().hex}{suffix}"
    file_path = os.path.join(ephemeral_dir, unique_filename)
    
    try:
        # Write content securely
        with open(file_path, "wb") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
                
        logger.info("Created ephemeral file: %s (size: %d bytes)", file_path, len(content))
        yield file_path
    finally:
        secure_shred(file_path)
